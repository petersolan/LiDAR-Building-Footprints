"""Extract building footprints from the nDSM and DSM (height + roughness baseline).

1. Tall:   nDSM above a minimum building height.
2. Seeds:  tall pixels with a smooth surface. Roughness is the mean absolute
           Laplacian of the DSM over a 5x5 window: planar roofs score low, tree
           canopy scores high.
3. Clean:  morphological opening removes thin slivers (e.g. hedges, tree edges
           that happen to be smooth); tiny seed blobs are dropped.
4. Grow:   seeds are grown back into the tall mask by a few pixels to recover
           roof edges and walls, where the height step makes roughness high.
5. Tidy:   closing, small-hole filling and a minimum-area filter.
6. Objects: each connected footprint must reach a minimum height (matching the
           ground truth's height filter) and be mostly smooth; blobs dominated
           by rough pixels are tree crowns that slipped through.
7. Vectorise and simplify, writing outputs/footprints/lidar_baseline.gpkg.
"""

import argparse

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import shape
from skimage import morphology

import config

PARAMS = dict(
    min_height=2.5,     # m above ground
    max_roughness=1.0,  # mean |Laplacian| of DSM over 5x5 (tuned on SW + NE)
    open_radius=0,      # px, disk radius for opening the seeds (0 = off)
    min_seed_px=10,     # px, smallest seed blob kept
    grow_px=4,          # px, geodesic growth of seeds into the tall mask
    max_hole_px=10,     # px, holes up to this size are filled
    min_area_m2=20.0,   # smallest footprint kept
    min_object_height=3.5,     # m, max nDSM within a footprint
    min_smooth_frac=0.0,       # share of a footprint's pixels below max_roughness
    max_object_roughness=99.0,  # mean roughness within a footprint
    simplify_m=0.5,     # Douglas-Peucker tolerance for the polygons
)

OUT_DIR = config.OUTPUTS / "footprints"
MASK_PATH = OUT_DIR / "lidar_baseline_mask.tif"
VECTOR_PATH = OUT_DIR / "lidar_baseline.gpkg"


def roughness(dsm):
    return ndi.uniform_filter(np.abs(ndi.laplace(dsm.astype("float32"))), 5)


def extract_mask(ndsm, dsm, p=PARAMS, rough=None):
    """Boolean building mask. `rough` can be passed in to reuse it while tuning."""
    tall = ndsm > p["min_height"]
    if rough is None:
        rough = roughness(dsm)

    seeds = tall & (rough < p["max_roughness"])
    if p["open_radius"]:
        seeds = ndi.binary_opening(seeds, structure=morphology.disk(p["open_radius"]).astype(bool))
    # skimage >= 0.26: max_size removes objects of size <= max_size
    seeds = morphology.remove_small_objects(seeds, max_size=p["min_seed_px"] - 1)

    # Geodesic dilation: grow seeds only into pixels that are tall
    mask = ndi.binary_dilation(seeds, iterations=p["grow_px"], mask=tall) if p["grow_px"] else seeds

    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), bool))
    mask = morphology.remove_small_holes(mask, max_size=p["max_hole_px"])
    min_px = int(p["min_area_m2"] / config.RESOLUTION ** 2)
    mask = morphology.remove_small_objects(mask, max_size=min_px - 1)
    return filter_objects(mask, ndsm, rough, p)


def filter_objects(mask, ndsm, rough, p=PARAMS):
    """Drop connected footprints that are too low or mostly rough (vegetation)."""
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    idx = np.arange(1, n + 1)
    max_h = ndi.maximum(ndsm, labels, idx)
    mean_rough = ndi.mean(rough, labels, idx)
    smooth_frac = ndi.mean(rough < p["max_roughness"], labels, idx)
    keep = ((max_h >= p["min_object_height"])
            & (smooth_frac >= p["min_smooth_frac"])
            & (mean_rough <= p["max_object_roughness"]))
    return np.concatenate([[False], keep])[labels]


def vectorise(mask, transform, crs, p=PARAMS):
    geoms = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=transform)
             if v == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["geometry"] = gdf.simplify(p["simplify_m"], preserve_topology=True).make_valid()
    gdf = gdf[gdf.area >= p["min_area_m2"]].reset_index(drop=True)
    gdf["area_m2"] = gdf.area.round(1)
    return gdf


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k, v in PARAMS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    p = vars(parser.parse_args())

    with rasterio.open(config.NDSM_PATH) as src:
        ndsm = src.read(1)
        profile = src.profile
        transform, crs = src.transform, src.crs
    with rasterio.open(config.DSM_PATH) as src:
        dsm = src.read(1)

    mask = extract_mask(ndsm, dsm, p)
    print(f"Building pixels: {mask.sum():,} ({100 * mask.mean():.2f}% of AOI)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="uint8", nodata=0, predictor=1)
    with rasterio.open(MASK_PATH, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)

    gdf = vectorise(mask, transform, crs, p)
    gdf.to_file(VECTOR_PATH, driver="GPKG", layer="lidar_baseline")
    print(f"Footprints: {len(gdf):,}, {gdf.area.sum() / 1e6:.3f} km2 -> {VECTOR_PATH}")
    print(f"Params: {p}")


if __name__ == "__main__":
    main()
