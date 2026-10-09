"""Explain why ground truth buildings were or weren't extracted as footprints.

Usage: python explain_building.py FID [FID ...]   (fid from gtd_buildings.parquet)

Re-runs each step of 03_extract_footprints.py on a window around the building
and reports, as a share of the building's pixels, what survives each step:
tall (nDSM), smooth (roughness), seeds, grown mask, size clean-up and the
per-footprint filters (height, NDVI). The first step that loses most of the
building is the likely cause.
"""

import argparse
import importlib

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage as ndi
from skimage import morphology

import config

ext = importlib.import_module("03_extract_footprints")
PAD_M = 60  # context around the building, so neighbouring components are whole


def read_window(path, bounds, shape=None, band=1, resampling=Resampling.nearest):
    with rasterio.open(path) as src:
        win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        data = src.read(band, window=win, out_shape=shape, resampling=resampling,
                        boundless=True, fill_value=0)
        return data, src.window_transform(win)


def gt_status(row):
    limit = config.GT_EXCLUDE_TYPES.get(row.description, 0.0)
    h = row.height_relativemax_m
    if row.geometry.area < limit:
        return f"ignored (excluded type under {limit:g} m2)"
    if h != h:  # NaN
        return "ignored (no OS height)"
    if h < config.GT_MIN_HEIGHT:
        return f"ignored (OS height {h} m < {config.GT_MIN_HEIGHT} m)"
    return "scored"


def explain(row, p, regularised):
    g = row.geometry
    minx, miny, maxx, maxy = g.bounds
    bounds = (np.floor(minx) - PAD_M, np.floor(miny) - PAD_M,
              np.ceil(maxx) + PAD_M, np.ceil(maxy) + PAD_M)
    ndsm, tr = read_window(config.NDSM_PATH, bounds)
    dsm, _ = read_window(config.DSM_PATH, bounds)
    final, _ = read_window(ext.MASK_PATH, bounds)
    ndvi = None
    if ext.NDVI_PATH.exists():
        ndvi, _ = read_window(ext.NDVI_PATH, bounds, shape=ndsm.shape,
                              resampling=Resampling.bilinear)
        ndvi = np.where(ndvi < -1, 0.0, ndvi)

    b = rasterize([(g, 1)], out_shape=ndsm.shape, transform=tr, dtype="uint8").astype(bool)
    npx = b.sum()
    share = lambda m: 100 * (m & b).sum() / npx

    rough = ext.roughness(dsm)
    tall = ndsm > p["min_height"]
    smooth = tall & (rough < p["max_roughness"])
    seeds = smooth
    if p["open_radius"]:
        seeds = ndi.binary_opening(seeds, structure=morphology.disk(p["open_radius"]).astype(bool))
    seeds = morphology.remove_small_objects(seeds, max_size=p["min_seed_px"] - 1)
    grown = ndi.binary_dilation(seeds, iterations=p["grow_px"], mask=tall) if p["grow_px"] else seeds
    tidy = ndi.binary_closing(grown, structure=np.ones((3, 3), bool))
    tidy = morphology.remove_small_holes(tidy, max_size=p["max_hole_px"])
    min_px = int(p["min_area_m2"] / config.RESOLUTION ** 2)
    sized = morphology.remove_small_objects(tidy, max_size=min_px - 1)

    h = ndsm[b]
    print(f"\n=== fid {row.fid}: {row.description} ({row.buildinguse}) ===")
    print(f"  OS: area {g.area:.0f} m2, height {row.height_relativemax_m} m, "
          f"{row.connectivity}; {gt_status(row)}")
    c = g.centroid
    print(f"  location: E {c.x:.0f}, N {c.y:.0f} "
          f"(1 km tile SX{int(c.x // 1000) % 100:02d}{int(c.y // 1000) % 100:02d})")
    print(f"  LiDAR nDSM in footprint: max {h.max():.1f} m, median {np.median(h):.1f} m, "
          f"p10 {np.percentile(h, 10):.1f} m")
    print(f"  roughness in footprint: median {np.median(rough[b]):.2f} "
          f"(threshold {p['max_roughness']})")
    if ndvi is not None:
        print(f"  Sentinel-2 NDVI in footprint: mean {ndvi[b].mean():.2f} "
              f"(filter > {p['max_object_ndvi']} for footprints < {p['ndvi_max_area_m2']:g} m2)")

    stages = [("tall (> %.1f m)" % p["min_height"], tall), ("tall and smooth", smooth),
              ("seeds after clean-up", seeds), ("grown into tall area", grown),
              ("after closing / size filter", sized)]
    print("  share of building pixels surviving each step:")
    for name, m in stages:
        print(f"    {name:30s} {share(m):5.1f}%")

    # Per-footprint filters, for components touching the building
    labels, n = ndi.label(sized)
    touching = np.unique(labels[b & sized])
    touching = touching[touching > 0]
    for lab in touching:
        comp = labels == lab
        area = comp.sum() * config.RESOLUTION ** 2
        reasons = []
        if ndsm[comp].max() < p["min_object_height"]:
            reasons.append(f"max height {ndsm[comp].max():.1f} m < {p['min_object_height']} m")
        if ndvi is not None and p["max_object_ndvi"] < 1.0:
            mean_ndvi = ndvi[comp].mean()
            if mean_ndvi > p["max_object_ndvi"] and area < p["ndvi_max_area_m2"]:
                reasons.append(f"green: mean NDVI {mean_ndvi:.2f} > {p['max_object_ndvi']}"
                               f" and area {area:.0f} m2 < {p['ndvi_max_area_m2']:g} m2")
        verdict = "DROPPED - " + "; ".join(reasons) if reasons else "kept"
        print(f"    footprint candidate {area:.0f} m2 covering {share(comp):.1f}% "
              f"of the building: {verdict}")

    print(f"    {'final mask':30s} {share(final.astype(bool)):5.1f}%")
    reg = regularised[regularised.intersects(g)]
    cov = reg.intersection(g).area.sum() / g.area * 100 if len(reg) else 0.0
    print(f"    {'squared footprints':30s} {cov:5.1f}%  ({len(reg)} footprint(s) touching)")

    # Verdict: smooth/seed steps only keep roof cores that growing expands again,
    # so judge losses at the tall, grown and size steps, then the object filters
    if share(final.astype(bool)) >= 50:
        print("  Detected (>= 50% covered)")
    elif share(seeds) == 0:
        print("  LIKELY CAUSE: no smooth roof pixels survived, so no seed to grow from"
              if share(tall) >= 50 else "  LIKELY CAUSE: not tall enough in the LiDAR nDSM")
    elif share(sized) < 50:
        print("  LIKELY CAUSE: seeds too small to grow over the building, or removed by the "
              "size filter")
    else:
        print("  LIKELY CAUSE: removed by a per-footprint filter (see candidates above)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fids", type=int, nargs="+")
    args = parser.parse_args()
    gt = gpd.read_parquet(config.GROUND_TRUTH_PATH)
    rows = gt[gt["fid"].isin(args.fids)]
    missing = set(args.fids) - set(rows["fid"])
    if missing:
        print(f"Not found in {config.GROUND_TRUTH_PATH.name}: {sorted(missing)}")
    regularised = gpd.read_file(config.OUTPUTS / "footprints" / "lidar_classified.gpkg",
                                bbox=tuple(rows.total_bounds))
    for row in rows.itertuples():
        explain(row, ext.PARAMS, regularised)


if __name__ == "__main__":
    main()
