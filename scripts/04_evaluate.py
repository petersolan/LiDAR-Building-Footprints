"""Evaluate footprint datasets against the OS ground truth.

Three levels, all computed on vector geometry within the analysis AOI. Ground
truth buildings filtered out by groundtruth.load_ground_truth (excluded types,
below 3.5 m, no height) become ignore zones: removed from the scoring area, and
predicted blocks lying mostly inside them are not counted.

1. Area:    precision, recall, F1 and IoU of the dissolved footprint area.
            Unaffected by how each dataset splits buildings. Also reported per
            quadrant: SW (urban) and NE (rural) were used to calibrate the LiDAR
            method; SE and NW are held out.
2. Object:  ground truth and predictions are dissolved into blocks (touching
            buildings merge, e.g. a terrace becomes one block) and matched at
            IoU >= 0.5.
3. Detail:  each ground truth building's coverage by the prediction; a building
            counts as detected at >= 50% coverage. Summarised by building class
            and size band.

Writes CSVs to outputs/eval/ and per-building coverage to
outputs/eval/gt_coverage.gpkg.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

import config
from groundtruth import load_ground_truth

EVAL_DIR = config.OUTPUTS / "eval"
DATASETS = {"lidar_baseline": config.OUTPUTS / "footprints" / "lidar_baseline.gpkg",
            **config.BENCHMARKS}

minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
QUADRANTS = {
    "sw (calibration)": box(minx, miny, midx, midy),
    "se (hold-out)": box(midx, miny, maxx, midy),
    "nw (hold-out)": box(minx, midy, midx, maxy),
    "ne (calibration)": box(midx, midy, maxx, maxy),
}

CLASSES = {
    "terraced/semi house": ["Mid-Terrace House", "End-Of-Terrace House", "Semi-Detached House"],
    "detached house": ["Detached House"],
    "flats/other residential": ["Multiple Residential Accommodation", "Residential Building",
                                "Static Caravan Or Mobile Home",
                                "Temporary Or Holiday Accommodation Building"],
    "outbuilding/unknown": ["Domestic Outbuilding", "Unknown Building", "Ancillary Building"],
    "commercial/mixed use": ["Commercial Building", "Mixed Use Building"],
    "small utility": ["Electricity Sub Station", "Gas Governor", "Pumping Station",
                      "Utility Building", "Waste Water Treatment Works",
                      "Water Distribution Facility", "Electricity Distribution Facility"],
}
SIZE_BINS = [0, 20, 50, 200, 1000, np.inf]
SIZE_LABELS = ["<20", "20-50", "50-200", "200-1000", ">1000"]


def polygons(geoms):
    """Valid, single-part polygons from any geometry array."""
    parts = shapely.get_parts(shapely.make_valid(np.asarray(geoms)))
    parts = parts[shapely.get_type_id(parts) == 3]  # Polygon
    return parts[shapely.area(parts) > 0]


def load(path, aoi):
    gdf = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(config.CRS)
    gdf = gdf[gdf.intersects(aoi)]
    return gdf


def blocks(geoms):
    """Dissolve to non-overlapping blocks; touching polygons merge."""
    return polygons([shapely.union_all(polygons(geoms))])


def area_metrics(gt_u, pr_u, region):
    g = shapely.intersection(gt_u, region)
    p = shapely.intersection(pr_u, region)
    tp = shapely.intersection(g, p).area
    precision, recall = tp / p.area, tp / g.area
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall),
                iou=tp / shapely.union(g, p).area,
                gt_km2=g.area / 1e6, pred_km2=p.area / 1e6)


def overlap_fraction(geoms, others):
    """Share of each geometry's area covered by `others` (non-overlapping polygons)."""
    i, j = shapely.STRtree(others).query(geoms, predicate="intersects")
    covered = np.bincount(i, shapely.area(shapely.intersection(geoms[i], others[j])), len(geoms))
    return covered / shapely.area(geoms)


def object_metrics(gt_blocks, pr_blocks, ignore_parts):
    gt = gpd.GeoDataFrame(geometry=gt_blocks, crs=config.CRS)
    in_ignore = overlap_fraction(pr_blocks, ignore_parts)
    pr = gpd.GeoDataFrame(geometry=pr_blocks[in_ignore <= 0.5], crs=config.CRS)
    pairs = gpd.sjoin(gt, pr, predicate="intersects")
    g = gt.geometry.values[pairs.index.values]
    p = pr.geometry.values[pairs["index_right"].values]
    iou = shapely.area(shapely.intersection(g, p)) / shapely.area(shapely.union(g, p))
    hits = pairs[iou >= 0.5]
    # IoU >= 0.5 implies a one-to-one match between non-overlapping blocks
    tp = hits.index.nunique()
    precision, recall = tp / len(pr), tp / len(gt)
    return dict(gt_blocks=len(gt), pred_blocks=len(pr), matched=tp,
                obj_precision=precision, obj_recall=recall,
                obj_f1=2 * precision * recall / (precision + recall) if tp else 0.0)


def coverage(gt, pr_blocks):
    """Fraction of each ground truth building's area covered by the prediction."""
    pr = gpd.GeoDataFrame(geometry=pr_blocks, crs=config.CRS)
    pieces = gpd.overlay(gt[["gt_id", "geometry"]], pr, how="intersection", keep_geom_type=True)
    covered = pieces.assign(a=pieces.area).groupby("gt_id")["a"].sum()
    return (covered.reindex(gt["gt_id"]).fillna(0).values / gt.area.values).clip(0, 1)


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()

    gt, ignored = load_ground_truth(aoi)
    gt["gt_id"] = np.arange(len(gt))
    lookup = {d: c for c, ds in CLASSES.items() for d in ds}
    gt["class"] = gt["description"].map(lookup).fillna("public/other")
    gt["size_band"] = pd.cut(gt.area, SIZE_BINS, labels=SIZE_LABELS, right=False)
    gt_blocks = blocks(shapely.intersection(gt.geometry.values, aoi))
    gt_u = shapely.union_all(gt_blocks)
    ignore_u = shapely.difference(shapely.union_all(polygons(ignored.geometry.values)), gt_u)
    ignore_parts = polygons([ignore_u])
    region = shapely.difference(aoi, ignore_u)
    quadrants = {q: shapely.intersection(b, region) for q, b in QUADRANTS.items()}
    print(f"Ground truth: {len(gt):,} scored buildings -> {len(gt_blocks):,} blocks; "
          f"{len(ignored):,} ignored ({ignore_u.area / 1e6:.3f} km2)")

    summary, quads, by_class, by_size = [], [], [], []
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        pr = load(path, aoi)
        pr_blocks = blocks(shapely.intersection(pr.geometry.values, aoi))
        pr_u = shapely.union_all(pr_blocks)

        row = dict(dataset=name, features=len(pr), **area_metrics(gt_u, pr_u, region),
                   **object_metrics(gt_blocks, pr_blocks, ignore_parts))
        cov = coverage(gt, pr_blocks)
        gt[f"cov_{name}"] = cov.round(3)
        row["bld_detected_pct"] = 100 * np.mean(cov >= 0.5)
        summary.append(row)
        print(f"{name}: area F1 {row['f1']:.3f}, IoU {row['iou']:.3f}, "
              f"object F1 {row['obj_f1']:.3f}, buildings detected {row['bld_detected_pct']:.1f}%")

        for q, qregion in quadrants.items():
            quads.append(dict(dataset=name, quadrant=q, **area_metrics(gt_u, pr_u, qregion)))

        det = gt.assign(detected=cov >= 0.5, covered_m2=cov * gt.area, gt_m2=gt.area)
        for key, out in (("class", by_class), ("size_band", by_size)):
            grp = det.groupby(key, observed=True)
            out.append(pd.DataFrame({
                "dataset": name,
                "buildings": grp.size(),
                "detected_pct": 100 * grp["detected"].mean(),
                "area_recall_pct": 100 * grp["covered_m2"].sum() / grp["gt_m2"].sum(),
            }).reset_index())

    pd.DataFrame(summary).round(4).to_csv(EVAL_DIR / "summary.csv", index=False)
    pd.DataFrame(quads).round(4).to_csv(EVAL_DIR / "by_quadrant.csv", index=False)
    pd.concat(by_class).round(1).to_csv(EVAL_DIR / "by_class.csv", index=False)
    pd.concat(by_size).round(1).to_csv(EVAL_DIR / "by_size.csv", index=False)
    gt.drop(columns="size_band").assign(size_band=gt["size_band"].astype(str)).to_file(
        EVAL_DIR / "gt_coverage.gpkg", driver="GPKG", layer="gt_coverage")

    pd.set_option("display.width", 200)
    cols = ["dataset", "features", "precision", "recall", "f1", "iou", "pred_km2",
            "pred_blocks", "obj_precision", "obj_recall", "obj_f1", "bld_detected_pct"]
    print("\n" + pd.DataFrame(summary)[cols].round(3).to_string(index=False))
    for title, df, idx in (("Area F1 by quadrant", pd.DataFrame(quads), "quadrant"),):
        print(f"\n{title}\n" + df.pivot(index=idx, columns="dataset", values="f1").round(3).to_string())
    for title, frames, idx in (("Buildings detected % by class", by_class, "class"),
                               ("Buildings detected % by size (m2)", by_size, "size_band")):
        df = pd.concat(frames)
        print(f"\n{title}\n" + df.pivot(index=idx, columns="dataset", values="detected_pct")
              .round(1).to_string())


if __name__ == "__main__":
    main()
