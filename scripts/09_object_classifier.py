"""Classify LiDAR footprint candidates as building / not building.

Candidates are the connected footprints from 03_extract_footprints.py before
the NDVI rule (height and roughness steps unchanged). Each gets size, height,
surface, greenness, shape and context features, and a gradient-boosting model
predicts whether it is a building.

Training labels come from the ground truth: a candidate is a building if more
of its pixels lie on scored OS buildings than outside any OS building.
Candidates lying mostly in ignore zones are left out of training.

Validation follows the project's split: the model is trained on the
calibration quadrants (SW + NE) with spatial cross-validation by 1 km tile, and
the probability threshold is chosen on those out-of-fold predictions. SE + NW
are held out. In the written mask, calibration candidates use their
out-of-fold predictions, so every quadrant's evaluation stays honest.

Writes outputs/footprints/lidar_classified_mask.tif and
outputs/diagnostics/candidates.parquet (features and probabilities).
"""

import importlib

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import config
import geopandas as gpd
from groundtruth import load_ground_truth

ext = importlib.import_module("03_extract_footprints")

MASK_OUT = config.OUTPUTS / "footprints" / "lidar_classified_mask.tif"
TABLE_OUT = config.OUTPUTS / "diagnostics" / "candidates.parquet"
CALIBRATION = ["sw", "ne"]
HOLDOUT = ["se", "nw"]
FEATURES = ["area", "h_mean", "h_max", "h_std", "slenderness", "rough_mean", "rough_median",
            "smooth_frac", "smooth_core", "ndvi_mean", "ndvi_max", "ndvi_min", "circularity",
            "rectangularity", "solidity", "elongation", "neighbours_50m", "nearest_m"]


def candidate_features(cand, ndsm, rough, ndvi, p):
    labels, n = ndi.label(cand)
    idx = np.arange(1, n + 1)
    df = pd.DataFrame({"label": idx})
    df["area"] = np.bincount(labels.ravel(), minlength=n + 1)[1:] * config.RESOLUTION ** 2
    df["h_mean"] = ndi.mean(ndsm, labels, idx)
    df["h_max"] = ndi.maximum(ndsm, labels, idx)
    df["h_std"] = ndi.standard_deviation(ndsm, labels, idx)
    df["slenderness"] = df["h_mean"] / np.sqrt(df["area"])
    df["rough_mean"] = ndi.mean(rough, labels, idx)
    df["rough_median"] = ndi.median(rough, labels, idx)
    df["smooth_frac"] = ndi.mean(rough < p["max_roughness"], labels, idx)
    df["smooth_core"] = ext.largest_smooth_patch(cand, labels, n, rough, p)
    df["ndvi_mean"] = ndi.mean(ndvi, labels, idx)
    df["ndvi_max"] = ndi.maximum(ndvi, labels, idx)
    df["ndvi_min"] = ndi.minimum(ndvi, labels, idx)

    props = pd.DataFrame(measure.regionprops_table(
        labels, properties=("label", "perimeter_crofton", "solidity", "orientation",
                            "axis_major_length", "axis_minor_length", "centroid")))
    df = df.merge(props, on="label")
    df["circularity"] = (4 * np.pi * df["area"] / df["perimeter_crofton"] ** 2).clip(0, 1.2)
    df["elongation"] = df["axis_minor_length"] / df["axis_major_length"].replace(0, np.nan)

    # Rectangularity: area / bounding rectangle aligned with the principal axis
    rows, cols = np.nonzero(labels)
    lab = labels[rows, cols]
    theta = df.set_index("label")["orientation"].reindex(np.arange(n + 1)).fillna(0).values
    t = theta[lab]
    u = rows * np.cos(t) + cols * np.sin(t)
    v = -rows * np.sin(t) + cols * np.cos(t)
    stats = {}
    for name, arr in (("u", u), ("v", v)):
        lo = np.full(n + 1, np.inf); hi = np.full(n + 1, -np.inf)
        np.minimum.at(lo, lab, arr); np.maximum.at(hi, lab, arr)
        stats[name] = (hi - lo + 1)[1:]
    df["rectangularity"] = (df["area"] / (stats["u"] * stats["v"])).clip(0, 1)

    # Context: neighbouring candidates by centroid
    xy = df[["centroid-1", "centroid-0"]].values * config.RESOLUTION
    tree = cKDTree(xy)
    df["neighbours_50m"] = [len(x) - 1 for x in tree.query_ball_point(xy, 50)]
    dist, _ = tree.query(xy, k=2)
    df["nearest_m"] = dist[:, 1]
    return labels, df


def pixel_f1(df, keep, totals):
    """Pixel F1 per quadrant from per-candidate pixel counts (no re-rasterising)."""
    out = {}
    for q, scored_total in totals.items():
        d = df[(df["quadrant"] == q) & keep]
        tp, fp = d["scored_px"].sum(), d["fp_px"].sum()
        fn = scored_total - tp
        out[q] = dict(f1=2 * tp / (2 * tp + fp + fn), precision=tp / max(tp + fp, 1),
                      recall=tp / scored_total)
    return out


def main():
    p = dict(ext.PARAMS)
    with rasterio.open(config.NDSM_PATH) as src:
        ndsm, transform, profile = src.read(1), src.transform, src.profile
    with rasterio.open(config.DSM_PATH) as src:
        rough = ext.roughness(src.read(1))
    ndvi = ext.load_ndvi(ndsm.shape)
    print("Extracting candidates (NDVI rule off)...", flush=True)
    cand = ext.extract_mask(ndsm, None, dict(p, max_object_ndvi=1.0), rough=rough)

    labels, df = candidate_features(cand, ndsm, rough, ndvi, p)
    print(f"{len(df):,} candidates", flush=True)

    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()
    scored, ignored = load_ground_truth(aoi)
    ras = lambda g: rasterize(((x, 1) for x in g), out_shape=ndsm.shape, transform=transform,
                              dtype="uint8").astype(bool)
    gt = ras(scored.geometry)
    ign = ras(ignored.geometry) & ~gt
    n = len(df)
    df["scored_px"] = np.bincount(labels.ravel(), weights=gt.ravel(), minlength=n + 1)[1:]
    df["ignore_px"] = np.bincount(labels.ravel(), weights=ign.ravel(), minlength=n + 1)[1:]
    df["fp_px"] = df["area"] - df["scored_px"] - df["ignore_px"]
    df["is_building"] = (df["scored_px"] > df["fp_px"]).astype(int)
    df["trainable"] = df["ignore_px"] < 0.5 * df["area"]

    half = 5000 / config.RESOLUTION
    df["quadrant"] = (np.where(df["centroid-0"] < half, "n", "s")
                      + np.where(df["centroid-1"] < half, "w", "e"))
    df["tile_1km"] = ((df["centroid-1"] // 1000).astype(int) * 100
                      + (df["centroid-0"] // 1000).astype(int))
    # Scored GT pixels per quadrant (for recall), same centroid-free split as tuning
    rows_n = slice(0, int(half)); rows_s = slice(int(half), None)
    cols_w = slice(0, int(half)); cols_e = slice(int(half), None)
    totals = {"sw": gt[rows_s, cols_w].sum(), "ne": gt[rows_n, cols_e].sum(),
              "se": gt[rows_s, cols_e].sum(), "nw": gt[rows_n, cols_w].sum()}

    cal = df["quadrant"].isin(CALIBRATION)
    train = cal & df["trainable"]
    X, y, w = df[FEATURES], df["is_building"], df["area"]

    def model():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=31, l2_regularization=1.0,
                                              random_state=0)

    # Out-of-fold probabilities for calibration candidates (spatial CV by 1 km tile)
    df["prob"] = np.nan
    cal_idx = np.flatnonzero(cal)
    groups = df["tile_1km"].values[cal_idx]
    for fold_train, fold_test in GroupKFold(n_splits=5).split(cal_idx, groups=groups):
        tr = cal_idx[fold_train]; tr = tr[df["trainable"].values[tr]]
        m = model().fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        te = cal_idx[fold_test]
        df.loc[df.index[te], "prob"] = m.predict_proba(X.iloc[te])[:, 1]

    # Threshold: best mean F1 over the calibration quadrants, out-of-fold
    best_t, best = 0.5, -1
    for t in np.arange(0.2, 0.81, 0.025):
        f = pixel_f1(df, df["prob"].fillna(0) >= t, {q: totals[q] for q in CALIBRATION})
        score = np.mean([f[q]["f1"] for q in CALIBRATION])
        if score > best:
            best_t, best = t, score
    print(f"Threshold {best_t:.3f} (calibration out-of-fold mean F1 {best:.3f})", flush=True)

    final = model().fit(X[train], y[train], sample_weight=w[train])
    hold = ~cal
    df.loc[hold, "prob"] = final.predict_proba(X[hold])[:, 1]
    df["keep"] = df["prob"] >= best_t

    # Current rule for comparison: NDVI > 0.7 on footprints < 400 m2 is dropped
    rule = ~((df["ndvi_mean"] > p["max_object_ndvi"]) & (df["area"] < p["ndvi_max_area_m2"]))
    res = {"no NDVI rule": pixel_f1(df, np.ones(len(df), bool), totals),
           "current NDVI rule": pixel_f1(df, rule, totals),
           "classifier": pixel_f1(df, df["keep"], totals)}
    print(f"\n{'method':18s} | {'SW':>5} {'NE':>5} | {'SE':>5} {'NW':>5} | mean  | "
          "precision SE/NW | recall SE/NW")
    for name, r in res.items():
        f = [r[q]["f1"] for q in ("sw", "ne", "se", "nw")]
        print(f"{name:18s} | {f[0]:.3f} {f[1]:.3f} | {f[2]:.3f} {f[3]:.3f} | {np.mean(f):.3f} | "
              f"{r['se']['precision']:.3f} / {r['nw']['precision']:.3f}   | "
              f"{r['se']['recall']:.3f} / {r['nw']['recall']:.3f}")
    print("(SW/NE: out-of-fold; SE/NW: held out)")

    keep_lab = np.concatenate([[False], df.sort_values("label")["keep"].values])
    out = keep_lab[labels]
    profile.update(dtype="uint8", nodata=0, predictor=1)
    with rasterio.open(MASK_OUT, "w", **profile) as dst:
        dst.write(out.astype("uint8"), 1)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df["x"] = transform.c + (df["centroid-1"] + 0.5) * config.RESOLUTION
    df["y"] = transform.f - (df["centroid-0"] + 0.5) * config.RESOLUTION
    df.to_parquet(TABLE_OUT, index=False)
    print(f"\n{int(df['keep'].sum()):,} of {len(df):,} candidates kept -> {MASK_OUT}")


if __name__ == "__main__":
    main()
