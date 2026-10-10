"""Export the footprints with heights for the web demo (site/).

Reads outputs/footprints/lidar_classified.gpkg and writes
site/data/footprints.geojson: simplified by 0.5 m in British National Grid
(so the tolerance is in metres), then reprojected to WGS84 with coordinates
rounded to 6 decimals (~0.1 m). Only LiDAR-derived columns are kept; the OS
ground truth is never exported.
"""

import geopandas as gpd

import config

SRC = config.OUTPUTS / "footprints" / "lidar_classified.gpkg"
OUT = config.ROOT / "site" / "data" / "footprints.geojson"
COLUMNS = ["area_m2", "height_median_m", "height_p95_m", "height_max_m"]


def main() -> None:
    gdf = gpd.read_file(SRC)[COLUMNS + ["geometry"]]
    gdf = gdf.dropna(subset=["height_max_m"])
    gdf["geometry"] = gdf.geometry.simplify(0.5, preserve_topology=True)
    for col in COLUMNS:
        gdf[col] = gdf[col].round(1)
    gdf = gdf.to_crs("EPSG:4326")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)
    gdf.to_file(OUT, driver="GeoJSON", engine="pyogrio",
                layer_options={"COORDINATE_PRECISION": 6, "RFC7946": "YES"})
    print(f"{len(gdf):,} footprints -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
