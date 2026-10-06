"""Describe each LiDAR footprint and label it against the ground truth.

Labels (by share of the footprint's area):
  match    >= 50% on scored ground truth buildings
  ignore   >= 50% on ignore-zone buildings
  partial  some building overlap, but neither of the above
  false    < 10% on any ground truth building

Per-object features: area, height stats from the nDSM, roughness stats, share
of smooth pixels, compactness and rectangularity. Writes
outputs/diagnostics/lidar_objects.gpkg and prints how well each feature
separates matches from false detections (AUC), for the south (urban) and north
(rural) halves of the AOI.
"""

import importlib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu

import config
from groundtruth import load_ground_truth

OUT_DIR = config.OUTPUTS / "diagnostics"
LIDAR_PATH = config.OUTPUTS / "footprints" / "lidar_baseline.gpkg"
NDVI_PATH = config.PROCESSED / "ndvi_s2_2022.tif"  # optional, from 07_sentinel2_ndvi.py
SMOOTH_THRESHOLD = 1.25  # roughness below this counts as smooth (matches extraction)


def overlap_share(geoms, others):
    i, j = shapely.STRtree(others).query(geoms, predicate="intersects")
    covered = np.bincount(i, shapely.area(shapely.intersection(geoms[i], others[j])), len(geoms))
    return covered / shapely.area(geoms)


def zonal(values, labels, n, func):
    return func(values, labels, np.arange(1, n + 1))


def auc(a, b):
    """Probability a random value from `a` exceeds one from `b` (0.5 = no separation)."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return mannwhitneyu(a, b).statistic / (len(a) * len(b))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = importlib.import_module("03_extract_footprints")

    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()
    scored, ignored = load_ground_truth(aoi)
    lid = gpd.read_file(LIDAR_PATH).reset_index(drop=True)
    geoms = lid.geometry.values

    s_share = overlap_share(geoms, shapely.make_valid(scored.geometry.values))
    i_share = overlap_share(geoms, shapely.make_valid(ignored.geometry.values))
    lid["scored_share"] = s_share.round(3)
    lid["ignore_share"] = i_share.round(3)
    lid["label"] = np.select(
        [s_share >= 0.5, i_share >= 0.5, s_share + i_share < 0.1],
        ["match", "ignore", "false"], "partial")

    with rasterio.open(config.NDSM_PATH) as src:
        ndsm = src.read(1); transform = src.transform
    with rasterio.open(config.DSM_PATH) as src:
        rough = ext.roughness(src.read(1))

    n = len(lid)
    labels = rasterize(((g, i + 1) for i, g in enumerate(geoms)), out_shape=ndsm.shape,
                       transform=transform, dtype="int32")
    lid["h_max"] = zonal(ndsm, labels, n, ndi.maximum)
    lid["h_mean"] = zonal(ndsm, labels, n, ndi.mean)
    lid["h_std"] = zonal(ndsm, labels, n, ndi.standard_deviation)
    lid["h_p10"] = lid["h_mean"] - 1.28 * lid["h_std"]  # rough lower tail, cheap
    lid["rough_mean"] = zonal(rough, labels, n, ndi.mean)
    lid["rough_median"] = zonal(rough, labels, n, ndi.median)
    lid["smooth_frac"] = zonal((rough < SMOOTH_THRESHOLD).astype("float32"), labels, n, ndi.mean)
    feats = []
    if NDVI_PATH.exists():
        # 10 m NDVI resampled bilinearly onto the 1 m grid
        with rasterio.open(NDVI_PATH) as src:
            ndvi = src.read(1, out_shape=ndsm.shape, resampling=Resampling.bilinear)
        ndvi = np.where(ndvi < -1, 0.0, ndvi)  # nodata -> neutral
        lid["ndvi_mean"] = zonal(ndvi, labels, n, ndi.mean)
        lid["ndvi_max"] = zonal(ndvi, labels, n, ndi.maximum)
        feats = ["ndvi_mean", "ndvi_max"]

    area = shapely.area(geoms)
    lid["area_m2"] = area.round(1)
    lid["compactness"] = 4 * np.pi * area / shapely.length(geoms) ** 2
    lid["rectangularity"] = area / shapely.area(shapely.oriented_envelope(geoms))
    c = shapely.centroid(geoms)
    lid["half"] = np.where(shapely.get_y(c) >= 95000, "north", "south")
    feats += ["area_m2", "h_max", "h_mean", "h_std", "rough_mean", "rough_median",
             "smooth_frac", "compactness", "rectangularity"]
    lid[feats] = lid[feats].astype("float64").round(3)
    lid.to_file(OUT_DIR / "lidar_objects.gpkg", driver="GPKG", layer="lidar_objects")

    pd.set_option("display.width", 200)
    print("Objects and area (km2) by label and half:")
    print(lid.pivot_table(index="label", columns="half", values="area_m2",
                          aggfunc=["count", lambda a: round(a.sum() / 1e6, 3)]))

    rows = []
    for half, d in lid.groupby("half"):
        m, f = d[d.label == "match"], d[d.label == "false"]
        for ft in feats:
            rows.append(dict(half=half, feature=ft, match_median=m[ft].median(),
                             false_median=f[ft].median(), auc_match_gt_false=auc(m[ft], f[ft])))
    sep = pd.DataFrame(rows)
    sep["separation"] = (sep["auc_match_gt_false"] - 0.5).abs()
    print("\nFeature separation, match vs false (AUC 0.5 = useless, 0/1 = perfect):")
    print(sep.sort_values(["half", "separation"], ascending=[True, False]).round(3).to_string(index=False))
    sep.round(4).to_csv(OUT_DIR / "feature_separation.csv", index=False)


if __name__ == "__main__":
    main()
