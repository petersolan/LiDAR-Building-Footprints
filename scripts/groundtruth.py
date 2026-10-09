"""Load the OS ground truth and split it into scored buildings and ignore zones."""

import geopandas as gpd

import config


def load_ground_truth(aoi):
    """Return (scored, ignored) GeoDataFrames clipped to the AOI's extent.

    Ignored: excluded building types (below each type's area limit), buildings
    under MIN_BUILDING_M2 or below GT_MIN_HEIGHT, and buildings with no
    recorded height.
    """
    gt = gpd.read_parquet(config.GROUND_TRUTH_PATH)
    if gt.crs.to_epsg() != 27700:
        gt = gt.to_crs(config.CRS)
    gt = gt[gt.intersects(aoi)].rename(columns={"fid": "os_fid"}).reset_index(drop=True)

    height = gt["height_relativemax_m"]
    exclude_below = gt["description"].map(config.GT_EXCLUDE_TYPES).fillna(0.0)
    excluded_type = gt.area < exclude_below
    keep = (~excluded_type & (gt.area >= config.MIN_BUILDING_M2)
            & height.notna() & (height >= config.GT_MIN_HEIGHT))
    return gt[keep].reset_index(drop=True), gt[~keep].reset_index(drop=True)
