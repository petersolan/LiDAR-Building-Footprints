"""Summer 2022 Sentinel-2 NDVI composite for the analysis AOI.

Searches Earth Search (Element 84, AWS open data) for Sentinel-2 L2A scenes,
masks clouds, cloud shadow and snow with each scene's scene classification
layer (SCL), and takes the per-pixel median NDVI over clear observations.
Output is on a 10 m grid aligned with the analysis extent:
data/processed/ndvi_s2_2022.tif (band 1 median NDVI, band 2 clear count).
"""

import numpy as np
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

import config

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
DATES = "2022-05-15/2022-09-15"
MAX_SCENE_CLOUD = 40   # %, scene-level pre-filter
MIN_CLEAR_SHARE = 0.6  # share of the AOI that must be clear to use a scene
RES = 10.0
OUT_PATH = config.PROCESSED / "ndvi_s2_2022.tif"

# SCL classes kept as clear: vegetation, bare soil, water, unclassified, dark area
CLEAR_SCL = [2, 4, 5, 6, 7]
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3",
                GDAL_HTTP_RETRY_DELAY="2", AWS_NO_SIGN_REQUEST="YES")


def grid():
    minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
    width, height = int((maxx - minx) / RES), int((maxy - miny) / RES)
    return from_origin(minx, maxy, RES, RES), width, height


def read(href, transform, width, height, resampling):
    with rasterio.open(href) as src, WarpedVRT(
            src, crs=config.CRS, transform=transform, width=width, height=height,
            resampling=resampling) as vrt:
        return vrt.read(1)


def reflectance(dn, item):
    # Processing baseline 04.00+ (Jan 2022 onwards) adds a +1000 DN offset, unless
    # Earth Search has already removed it (earthsearch:boa_offset_applied)
    baseline = float(item.properties.get("s2:processing_baseline", "0"))
    already = item.properties.get("earthsearch:boa_offset_applied", False)
    offset = 1000 if baseline >= 4.0 and not already else 0
    out = dn.astype("float32") - offset
    out[dn == 0] = np.nan  # 0 is nodata
    return out / 10000.0


def main():
    bbox = transform_bounds(config.CRS, "EPSG:4326", *config.ANALYSIS_BOUNDS)
    items = pystac_client.Client.open(STAC_URL).search(
        collections=[COLLECTION], bbox=bbox, datetime=DATES,
        query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}}).item_collection()
    print(f"{len(items)} candidate scenes", flush=True)

    transform, width, height = grid()
    stack = []
    with rasterio.Env(**GDAL_ENV):
        for item in sorted(items, key=lambda i: i.datetime):
            scl = read(item.assets["scl"].href, transform, width, height, Resampling.nearest)
            clear = np.isin(scl, CLEAR_SCL)
            share = clear.mean()
            label = f"{item.datetime:%Y-%m-%d} {item.properties.get('s2:mgrs_tile', '')}"
            if share < MIN_CLEAR_SHARE:
                print(f"  skip {label}: {share:.0%} clear", flush=True)
                continue
            red = reflectance(read(item.assets["red"].href, transform, width, height,
                                   Resampling.bilinear), item)
            nir = reflectance(read(item.assets["nir"].href, transform, width, height,
                                   Resampling.bilinear), item)
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = (nir - red) / (nir + red)
            ndvi[~clear | ~np.isfinite(ndvi)] = np.nan
            stack.append(ndvi)
            print(f"  use  {label}: {share:.0%} clear", flush=True)

    if not stack:
        raise SystemExit("No usable scenes")
    cube = np.stack(stack)
    count = np.isfinite(cube).sum(axis=0).astype("float32")
    with np.errstate(all="ignore"):
        median = np.nanmedian(cube, axis=0).astype("float32")
    median[count == 0] = config.NODATA

    profile = dict(driver="GTiff", dtype="float32", count=2, width=width, height=height,
                   crs=config.CRS, transform=transform, nodata=config.NODATA,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(median, 1)
        dst.write(count, 2)
        dst.set_band_description(1, "median NDVI (clear obs)")
        dst.set_band_description(2, "clear observation count")

    valid = median[count > 0]
    print(f"\n{len(stack)} scenes used; clear obs per pixel: min {int(count.min())}, "
          f"median {int(np.median(count))}; NDVI p5/p50/p95 "
          f"{np.percentile(valid, [5, 50, 95]).round(2)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
