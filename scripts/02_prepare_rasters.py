"""Mosaic EA LiDAR DSM/DTM tiles onto the analysis grid and derive the nDSM.

Selects the 1 m tiles overlapping config.ANALYSIS_BOUNDS, mosaics them onto a
common 1 m grid covering exactly that extent, and writes dsm.tif, dtm.tif and
ndsm.tif (DSM - DTM) to data/processed. Reports the share of the AOI with no
data so missing tiles are obvious.
"""

import sys

import numpy as np
import rasterio
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.merge import merge

import config

PROFILE = dict(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=config.NODATA,
    compress="deflate",
    predictor=3,
    tiled=True,
    blockxsize=512,
    blockysize=512,
    BIGTIFF="IF_SAFER",
)


def overlaps(bounds, target):
    # Touching edges share no pixels, so require a positive-area intersection
    return (not disjoint_bounds(bounds, target)
            and min(bounds.right, target.right) > max(bounds.left, target.left)
            and min(bounds.top, target.top) > max(bounds.bottom, target.bottom))


def select_tiles(folder, target):
    tiles = []
    for path in sorted(folder.glob("*.tif")):
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 27700:
                sys.exit(f"{path.name}: unexpected CRS {src.crs}")
            if src.res != (config.RESOLUTION, config.RESOLUTION):
                sys.exit(f"{path.name}: unexpected resolution {src.res}")
            if overlaps(src.bounds, target):
                tiles.append(path)
    return tiles


def mosaic(name, folder, out_path, target):
    tiles = select_tiles(folder, target)
    print(f"{name}: {len(tiles)} tiles -> {[p.stem for p in tiles]}")
    if not tiles:
        sys.exit(f"{name}: no tiles overlap the analysis extent")

    sources = [rasterio.open(p) for p in tiles]
    try:
        data, transform = merge(
            sources, bounds=tuple(target), res=config.RESOLUTION,
            nodata=config.NODATA, dtype="float32",
        )
        crs = sources[0].crs
    finally:
        for src in sources:
            src.close()

    band = data[0]
    missing = 100 * np.mean(band == config.NODATA)
    valid = band[band != config.NODATA]
    print(f"{name}: {band.shape[1]} x {band.shape[0]} px, {missing:.2f}% no data, "
          f"elevation {valid.min():.1f} to {valid.max():.1f} m")
    if missing > 0.5:
        print(f"  WARNING: {name} has gaps - check for missing tiles")

    profile = dict(PROFILE, width=band.shape[1], height=band.shape[0],
                   crs=crs, transform=transform)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band, 1)
    return band, profile


def main():
    target = BoundingBox(*config.ANALYSIS_BOUNDS)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    dsm, profile = mosaic("DSM", config.DSM_DIR, config.DSM_PATH, target)
    dtm, _ = mosaic("DTM", config.DTM_DIR, config.DTM_PATH, target)

    valid = (dsm != config.NODATA) & (dtm != config.NODATA)
    ndsm = np.full(dsm.shape, config.NODATA, dtype="float32")
    ndsm[valid] = dsm[valid] - dtm[valid]
    with rasterio.open(config.NDSM_PATH, "w", **profile) as dst:
        dst.write(ndsm, 1)

    h = ndsm[valid]
    print(f"nDSM: {100 * valid.mean():.2f}% valid, "
          f"{100 * np.mean(h < -0.5):.2f}% below -0.5 m, "
          f"{100 * np.mean(h > 2.5):.2f}% above 2.5 m, "
          f"p99 {np.percentile(h, 99):.1f} m")
    print(f"Wrote {config.DSM_PATH.name}, {config.DTM_PATH.name}, "
          f"{config.NDSM_PATH.name} to {config.PROCESSED}")


if __name__ == "__main__":
    main()
