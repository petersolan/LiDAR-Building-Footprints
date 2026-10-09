"""Cut railway lines for the AOI from OS Open Zoomstack (used by the bridge rule).

OS Open Roads has no railways, so footprints over a railway (footbridges, road
bridges, station canopies) can't be recognised from roads alone. Tunnels are
left out: buildings can stand above them.

Usage: python prepare_rail.py <path to OS_Open_Zoomstack.gpkg>
"""

import argparse
from pathlib import Path

import geopandas as gpd

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zoomstack", type=Path, help="OS Open Zoomstack GeoPackage")
    args = parser.parse_args()

    minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
    rail = gpd.read_file(args.zoomstack, layer="rail", bbox=(minx - 200, miny - 200, maxx + 200, maxy + 200))
    rail = rail[rail["type"] != "Tunnel"].to_crs(config.CRS)
    config.RAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    rail[["type", "geometry"]].to_file(config.RAIL_PATH, layer="rail")
    print(f"{len(rail)} rail lines, {rail.length.sum() / 1000:.1f} km -> {config.RAIL_PATH}")


if __name__ == "__main__":
    main()
