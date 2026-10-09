"""Add building heights to the footprints and check them against OS heights.

For every footprint in outputs/footprints/lidar_classified.gpkg, the heights
above ground of the 1 m nDSM pixels inside it give:
  height_median_m  typical roof height (eaves and ridge mixed)
  height_p95_m     near-top of the roof, robust to a stray tree branch or mast
  height_max_m     highest point, comparable with OS height_relativemax_m
Together with the footprint that is a simple block (LoD1) building model.

Check: a footprint is compared with an OS building when the two match one to
one (IoU >= 0.5, the same rule as the object-level evaluation), so merged
footprints covering several buildings are left out. Reports the error of
height_max_m and height_p95_m against OS height_relativemax_m overall, per
quadrant and per building class: mean (bias), median and mean absolute error,
RMSE and the share within 1 m and 2 m.

Writes the height columns back into lidar_classified.gpkg, and
outputs/eval/heights.csv (summary) and heights_pairs.csv (matched pairs).
"""

import importlib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize

import config
from groundtruth import load_ground_truth

ev = importlib.import_module("04_evaluate")
FOOTPRINTS = config.OUTPUTS / "footprints" / "lidar_classified.gpkg"


def footprint_heights(fp: gpd.GeoDataFrame) -> pd.DataFrame:
    """Median, 95th percentile and maximum nDSM height inside each footprint."""
    with rasterio.open(config.NDSM_PATH) as src:
        ndsm, transform = src.read(1), src.transform
    # Footprints don't overlap, so one label raster holds them all
    labels = rasterize(((g, i + 1) for i, g in enumerate(fp.geometry)), out_shape=ndsm.shape,
                       transform=transform, dtype="int32")
    inside = labels > 0
    order = np.argsort(labels[inside], kind="stable")
    lab = labels[inside][order]
    val = ndsm[inside][order]
    starts = np.searchsorted(lab, np.arange(1, len(fp) + 1))
    ends = np.searchsorted(lab, np.arange(1, len(fp) + 1), side="right")
    rows = []
    for s, e in zip(starts, ends, strict=True):
        v = val[s:e]
        v = v[np.isfinite(v)]
        rows.append((np.median(v), np.percentile(v, 95), v.max()) if len(v) else (np.nan,) * 3)
    return pd.DataFrame(rows, columns=["height_median_m", "height_p95_m", "height_max_m"],
                        index=fp.index).round(2)


def matched_pairs(fp: gpd.GeoDataFrame, gt: gpd.GeoDataFrame) -> pd.DataFrame:
    """Footprint - OS building pairs that match one to one at IoU >= 0.5."""
    pairs = gpd.sjoin(fp[["geometry"]], gt[["geometry"]], predicate="intersects")
    a = fp.geometry.values[pairs.index.values]
    b = gt.geometry.values[pairs["index_right"].values]
    iou = shapely.area(shapely.intersection(a, b)) / shapely.area(shapely.union(a, b))
    hits = pairs[iou >= 0.5]
    # IoU >= 0.5 is one to one for non-overlapping shapes
    return pd.DataFrame({"footprint": hits.index.values, "os": hits["index_right"].values,
                         "iou": iou[iou >= 0.5]})


def errors(diff: pd.Series) -> dict[str, float]:
    a = diff.abs()
    return dict(n=len(diff), bias_m=diff.mean(), median_abs_error_m=a.median(), mean_abs_error_m=a.mean(),
                rmse_m=float(np.sqrt((diff ** 2).mean())), within_1m_pct=100 * (a <= 1).mean(),
                within_2m_pct=100 * (a <= 2).mean())


def main():
    fp = gpd.read_file(FOOTPRINTS)
    heights = footprint_heights(fp)
    fp = fp.drop(columns=[c for c in heights.columns if c in fp.columns]).join(heights)
    fp.to_file(FOOTPRINTS, layer=FOOTPRINTS.stem)
    print(f"Heights added to {len(fp):,} footprints -> {FOOTPRINTS}")

    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()
    gt, _ = load_ground_truth(aoi)
    lookup = {d: c for c, ds in ev.CLASSES.items() for d in ds}
    gt["class"] = gt["description"].map(lookup).fillna("public/other")
    pairs = matched_pairs(fp, gt)
    pairs = pairs.join(fp[["height_median_m", "height_p95_m", "height_max_m"]], on="footprint")
    pairs = pairs.join(gt[["height_relativemax_m", "class", "description"]], on="os")
    point = shapely.point_on_surface(fp.geometry.values[pairs["footprint"].values])
    pairs["quadrant"] = np.select([shapely.within(point, b) for b in ev.QUADRANTS.values()],
                                  list(ev.QUADRANTS), default="")
    pairs = pairs.dropna(subset=["height_max_m"])
    pairs["error_max_m"] = pairs["height_max_m"] - pairs["height_relativemax_m"]
    pairs["error_p95_m"] = pairs["height_p95_m"] - pairs["height_relativemax_m"]

    rows = []
    for measure in ("max", "p95"):
        col = f"error_{measure}_m"
        rows.append(dict(measure=measure, group="all", **errors(pairs[col])))
        for key in ("quadrant", "class"):
            for name, g in pairs.groupby(key):
                rows.append(dict(measure=measure, group=f"{key}: {name}", **errors(g[col])))
    summary = pd.DataFrame(rows)
    ev.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(ev.EVAL_DIR / "heights.csv", index=False)
    pairs.round(3).to_csv(ev.EVAL_DIR / "heights_pairs.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{len(pairs):,} footprints match one OS building (IoU >= 0.5); "
          f"correlation of maximum heights {pairs['height_max_m'].corr(pairs['height_relativemax_m']):.3f}")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
