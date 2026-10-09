"""Classify LiDAR footprint candidates as building / not building.

Candidates are the connected footprints from 03_extract_footprints.py before
the NDVI rule (height and roughness steps unchanged). Each gets size, height,
surface, greenness, shape and context features, and a gradient-boosting model
predicts whether it is a building.

Training labels come from the ground truth: a candidate is a building if more
of its pixels lie on scored OS buildings than outside any OS building.
Candidates lying mostly in ignore zones are left out of training.

Validation is leave-one-quadrant-out: for each of the four 5 km quadrants, a
model is trained and tuned on the other three and predicts the fourth. Every
quadrant is therefore scored by a model that never saw it, and the written
mask uses those predictions, so the evaluation stays honest everywhere.
Within the training quadrants, the probability threshold and the bridge rule
are chosen on out-of-fold predictions (5-fold spatial cross-validation by 1 km
tile), never on the quadrant being predicted.

For comparison the previous scheme is also run: train on SW + NE only and
predict SE + NW. Finally one model is trained and tuned on all four quadrants
and saved (outputs/models/), for mapping areas outside this AOI.

Bridge rule: candidates with road or railway centreline running through them
(OS Open Roads, and OS Open Zoomstack rail from prepare_rail.py) are bridges,
footbridges and the like, which look like flat roofs to the LiDAR. A candidate
is dropped when the centreline inside it (1 m in from its edge) is at least
`ratio` x the square root of its area, and it is under 5,000 m2 (large
stations and shopping centres do have roads and tracks through them). The
ratio is chosen with the threshold, after the classifier, on the training
quadrants; the size limit is fixed (see BRIDGE_MAX_AREAS).

Writes outputs/footprints/lidar_classified_mask.tif,
outputs/diagnostics/candidates.parquet (features and probabilities) and the
classifier report in outputs/eval/:
  classifier_pixels.csv      pixel precision / recall / F1 per quadrant: the NDVI
                             rule the classifier replaced, the previous SW + NE
                             scheme, the classifier and the final decision
                             (classifier + bridge rule + minimum size)
  classifier_candidates.csv  per quadrant, candidate level: ROC AUC, average
                             precision and the confusion counts at the chosen
                             threshold
  classifier_features.csv    permutation importance, averaged over the four
                             held-out quadrants: drop in average precision when
                             a feature is shuffled
  classifier_settings.json   threshold and bridge rule chosen in each fold and
                             for the saved all-quadrant model
"""

import importlib
import json

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import config
import geopandas as gpd
from groundtruth import load_ground_truth

ext = importlib.import_module("03_extract_footprints")

MASK_OUT = config.OUTPUTS / "footprints" / "lidar_classified_mask.tif"
TABLE_OUT = config.OUTPUTS / "diagnostics" / "candidates.parquet"
EVAL_DIR = config.OUTPUTS / "eval"
MODEL_OUT = config.OUTPUTS / "models" / "building_classifier.joblib"
QUADRANTS = ["sw", "ne", "se", "nw"]
PREVIOUS_TRAINING = ["sw", "ne"]  # the earlier calibration / hold-out split
THRESHOLDS = np.arange(0.2, 0.81, 0.025)
FEATURES = ["area", "h_mean", "h_max", "h_std", "slenderness", "rough_mean", "rough_median",
            "smooth_frac", "smooth_core", "ndvi_mean", "ndvi_max", "ndvi_min", "circularity",
            "rectangularity", "solidity", "elongation", "neighbours_50m", "nearest_m"]
BRIDGE_RATIOS = [0.8, 1.0, 1.2, 1.5, 2.0]
# The size limit is a design choice, not tuned: below large stations and shopping centres
# (Exeter St Davids is ~21,800 m2, with track through it) and above road bridges. With only
# a few dozen bridges, tuning it per fold picked limits by noise (one fold chose 2,000 m2
# and kept the 3,400 m2 A30 bridge over the M5)
BRIDGE_MAX_AREAS = [5000]


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


def line_ratio(labels, n, area, transform):
    """Road and railway centreline inside each candidate, relative to its size (sqrt area)."""
    minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
    lines = pd.concat([gpd.read_file(config.ROADS_PATH, bbox=(minx, miny, maxx, maxy)).geometry,
                       gpd.read_file(config.RAIL_PATH).geometry])
    on_line = rasterize(((g, 1) for g in lines), out_shape=labels.shape, transform=transform,
                        dtype="uint8").astype(bool)
    # 1 m in from the edge, so a road running along a footprint's side doesn't count
    inner = ndi.binary_erosion(labels > 0) & on_line
    line_m = np.bincount(labels[inner], minlength=n + 1)[1:] * config.RESOLUTION
    return line_m / np.sqrt(area)


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


def candidate_report(df):
    """Candidate-level quality per quadrant: how well the probabilities rank buildings
    (ROC AUC, average precision) and the confusion counts at the chosen threshold.
    Probabilities come from the leave-one-quadrant-out models, so no quadrant is
    scored by a model that saw its labels."""
    rows = []
    for q in QUADRANTS:
        d = df[(df["quadrant"] == q) & df["trainable"]]
        y, prob, pred = d["is_building"], d["prob"], d["keep_classifier"]
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        rows.append(dict(
            quadrant=q,
            candidates=len(d), buildings=int(y.sum()),
            roc_auc=roc_auc_score(y, prob), average_precision=average_precision_score(y, prob),
            # Area-weighted: large footprints matter more for the area scores
            roc_auc_area=roc_auc_score(y, prob, sample_weight=d["area"]),
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
            f1=2 * tp / max(2 * tp + fp + fn, 1)))
    return pd.DataFrame(rows)


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def apply_rule(df, keep, rule):
    """Keep decisions after the bridge rule (ratio, max area); rule None = off."""
    if rule is None:
        return keep
    return keep & ~((df["line_ratio"] >= rule[0]) & (df["area"] < rule[1]))


def fit_and_tune(df, quads, totals):
    """Train on the given quadrants and choose the threshold and bridge rule there.

    Out-of-fold probabilities (5-fold spatial CV by 1 km tile, within these
    quadrants) choose the threshold and then the bridge rule, both by mean pixel
    F1 over the quadrants. The returned model is then fitted on all of them.
    Returns (model, threshold, bridge rule or None, out-of-fold probabilities).
    """
    X, y, w = df[FEATURES], df["is_building"], df["area"]
    inside = df["quadrant"].isin(quads).values
    idx = np.flatnonzero(inside)
    oof = pd.Series(np.nan, index=df.index)
    for fold_train, fold_test in GroupKFold(n_splits=5).split(idx, groups=df["tile_1km"].values[idx]):
        tr = idx[fold_train]; tr = tr[df["trainable"].values[tr]]
        m = model().fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        oof.iloc[idx[fold_test]] = m.predict_proba(X.iloc[idx[fold_test]])[:, 1]

    sub = {q: totals[q] for q in quads}

    def mean_f1(keep):
        return np.mean([f["f1"] for f in pixel_f1(df, keep, sub).values()])

    scores = [mean_f1(oof.fillna(0) >= t) for t in THRESHOLDS]
    threshold = float(THRESHOLDS[int(np.argmax(scores))])
    keep = oof.fillna(0) >= threshold
    rule, best = None, mean_f1(keep)
    for ratio in BRIDGE_RATIOS:
        for max_area in BRIDGE_MAX_AREAS:
            score = mean_f1(apply_rule(df, keep, (ratio, max_area)))
            if score > best + 1e-6:
                rule, best = (ratio, float(max_area)), score

    train = inside & df["trainable"].values
    final = model().fit(X[train], y[train], sample_weight=w[train])
    return final, threshold, rule, oof


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

    df["line_ratio"] = line_ratio(labels, n, df["area"].values, transform)
    X, y, w = df[FEATURES], df["is_building"], df["area"]

    # Leave-one-quadrant-out: each quadrant is predicted by a model trained and tuned
    # on the other three
    df["prob"] = np.nan
    df["keep_classifier"] = False
    df["bridge"] = False
    folds, importances = {}, []
    for q in QUADRANTS:
        others = [o for o in QUADRANTS if o != q]
        m, threshold, rule, _ = fit_and_tune(df, others, totals)
        here = (df["quadrant"] == q).values
        df.loc[here, "prob"] = m.predict_proba(X[here])[:, 1]
        df.loc[here, "keep_classifier"] = df.loc[here, "prob"] >= threshold
        everything = pd.Series(True, index=df.index[here])
        df.loc[here, "bridge"] = (~apply_rule(df[here], everything, rule)).values
        folds[q] = dict(trained_on=others, threshold=round(threshold, 3),
                        bridge_rule=None if rule is None else dict(min_line_ratio=rule[0], max_area_m2=rule[1]))
        held = here & df["trainable"].values
        imp = permutation_importance(m, X[held], y[held], sample_weight=w[held],
                                     scoring="average_precision", n_repeats=5, random_state=0)
        importances.append(imp.importances_mean)
        print(f"{q}: trained on {'+'.join(others)}, threshold {threshold:.3f}, bridge rule {rule}", flush=True)
    df["keep"] = df["keep_classifier"] & ~df["bridge"] & (df["area"] >= config.MIN_BUILDING_M2)

    # The previous scheme, for comparison: tuned on SW + NE (out-of-fold there),
    # SE + NW predicted by the model trained on SW + NE
    m_prev, t_prev, rule_prev, oof_prev = fit_and_tune(df, PREVIOUS_TRAINING, totals)
    prob_prev = oof_prev.copy()
    rest = ~df["quadrant"].isin(PREVIOUS_TRAINING).values
    prob_prev[rest] = m_prev.predict_proba(X[rest])[:, 1]
    keep_prev = apply_rule(df, prob_prev >= t_prev, rule_prev) & (df["area"] >= config.MIN_BUILDING_M2)

    # The NDVI rule the classifier replaced: NDVI > 0.7 on footprints < 400 m2 is dropped
    rule = ~((df["ndvi_mean"] > p["max_object_ndvi"]) & (df["area"] < p["ndvi_max_area_m2"]))
    res = {"no NDVI rule": pixel_f1(df, np.ones(len(df), bool), totals),
           "NDVI rule": pixel_f1(df, rule, totals),
           "previous scheme (SW+NE)": pixel_f1(df, keep_prev, totals),
           "classifier": pixel_f1(df, df["keep_classifier"], totals),
           "final": pixel_f1(df, df["keep"], totals)}
    print(f"\n{'method':24s} |    SW    NE    SE    NW | mean")
    for name, r in res.items():
        f = [r[q]["f1"] for q in QUADRANTS]
        print(f"{name:24s} | {f[0]:.3f} {f[1]:.3f} {f[2]:.3f} {f[3]:.3f} | {np.mean(f):.3f}")
    print("(classifier and final: leave-one-quadrant-out; previous scheme: SW/NE out-of-fold, "
          "SE/NW held out)")

    # The model for new areas: trained and tuned on all four quadrants
    m_all, t_all, rule_all, oof_all = fit_and_tune(df, QUADRANTS, totals)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=m_all, features=FEATURES, threshold=t_all, bridge_rule=rule_all,
                     min_building_m2=config.MIN_BUILDING_M2), MODEL_OUT)
    print(f"All-quadrant model: threshold {t_all:.3f}, bridge rule {rule_all} -> {MODEL_OUT}")

    # Classifier report (see module docstring)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(method=m, quadrant=q, **v) for m, r in res.items() for q, v in r.items()]
                 ).round(4).to_csv(EVAL_DIR / "classifier_pixels.csv", index=False)
    cand = candidate_report(df)
    cand.round(4).to_csv(EVAL_DIR / "classifier_candidates.csv", index=False)
    print("\nCandidate level (trainable candidates):\n" + cand.round(3).to_string(index=False))
    imp = np.vstack(importances)
    feats = pd.DataFrame({"feature": FEATURES, "importance": imp.mean(0), "min_over_folds": imp.min(0),
                          "max_over_folds": imp.max(0)}).sort_values("importance", ascending=False)
    feats.round(4).to_csv(EVAL_DIR / "classifier_features.csv", index=False)
    print("\nPermutation importance, mean over the four held-out quadrants "
          "(drop in average precision):\n" + feats.round(4).to_string(index=False))
    settings = dict(
        validation="leave-one-quadrant-out", folds=folds,
        all_quadrant_model=dict(threshold=round(t_all, 3), bridge_rule=None if rule_all is None else
                                dict(min_line_ratio=rule_all[0], max_area_m2=rule_all[1]),
                                file=MODEL_OUT.relative_to(config.ROOT).as_posix()),
        min_building_m2=config.MIN_BUILDING_M2, candidates=len(df), kept=int(df["keep"].sum()),
        dropped_by_bridge_rule=int((df["keep_classifier"] & df["bridge"]).sum()))
    (EVAL_DIR / "classifier_settings.json").write_text(json.dumps(settings, indent=2))

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
