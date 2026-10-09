"""Shared paths and settings for the Exeter building footprint project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"

CRS = "EPSG:27700"  # British National Grid

# AOI as supplied, and the analysis extent it is clipped to: the 10 km OS grid
# square SX99. The supplied AOI overhangs SX99 by ~20-50 m on the W and S edges.
AOI_PATH = DATA / "aoi" / "aoi.shp"
ANALYSIS_BOUNDS = (290000.0, 90000.0, 300000.0, 100000.0)  # minx, miny, maxx, maxy
AOI_ANALYSIS_PATH = PROCESSED / "aoi_analysis.gpkg"

DSM_DIR = DATA / "height" / "dsm"
DTM_DIR = DATA / "height" / "dtm"
RESOLUTION = 1.0  # metres
NODATA = -9999.0

DSM_PATH = PROCESSED / "dsm.tif"
DTM_PATH = PROCESSED / "dtm.tif"
NDSM_PATH = PROCESSED / "ndsm.tif"

GROUND_TRUTH_PATH = DATA / "ground_truth_data" / "gtd_buildings.parquet"

# OS Open Roads RoadLink (SX square): footprint orientation (optional) and the bridge rule
ROADS_PATH = DATA / "roads" / "SX_RoadLink.shp"
# Railways from OS Open Zoomstack (prepare_rail.py), for the bridge rule
RAIL_PATH = DATA / "roads" / "zoomstack_rail.gpkg"
GROUND_TRUTH_GPKG = DATA / "ground_truth_data" / "bld_fts_building.gpkg"

BENCHMARKS = {
    "microsoft": DATA / "footprints" / "microsoft_buildings.parquet",
    "osm": DATA / "footprints" / "osm_buildings.parquet",
    "os_open": DATA / "footprints" / "os_buildings.parquet",
}

# Ground truth buildings left out of scoring. They become "ignore" zones, so a
# dataset is neither rewarded nor penalised for footprints there.
# Excluded building types, each with the footprint area (m2) below which it is
# excluded; inf = always. Large "Unknown Building" features are mostly real
# buildings that OS has not classified, so only small ones are excluded.
GT_EXCLUDE_TYPES = {
    "Domestic Outbuilding": float("inf"),
    "Unknown Building": 50.0,
    "Electricity Sub Station": float("inf"),
}
GT_MIN_HEIGHT = 3.5  # m, OS height_relativemax_m; buildings with no height are ignored too
# Smallest footprint that counts as a building, for every dataset: smaller ground
# truth buildings become ignore zones and smaller predicted footprints are left
# out. 30 m2 drops kiosks and fragments; 40 m2 also lost ~4,100 real garages.
MIN_BUILDING_M2 = 30.0
