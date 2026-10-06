"""Profile the AOI, ground truth and benchmark footprint datasets.

Writes the analysis AOI (supplied AOI clipped to SX99) and prints, per dataset:
CRS, feature count, geometry types, invalid geometries, attribute columns, and
coverage inside the analysis AOI. A summary table goes to outputs/vector_profile.csv.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import config


def load(path):
    try:
        return gpd.read_parquet(path)
    except Exception:
        # Not GeoParquet (e.g. a WKB column without geo metadata): let GDAL read it
        return gpd.read_file(path, engine="pyogrio")


def analysis_aoi():
    aoi = gpd.read_file(config.AOI_PATH).to_crs(config.CRS)
    clipped = aoi.union_all().intersection(box(*config.ANALYSIS_BOUNDS))
    out = gpd.GeoDataFrame({"name": ["exeter_sx99"]}, geometry=[clipped], crs=config.CRS)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_file(config.AOI_ANALYSIS_PATH, driver="GPKG")
    print(f"AOI: supplied area {aoi.area.sum() / 1e6:.3f} km2, "
          f"analysis area {clipped.area / 1e6:.3f} km2 -> {config.AOI_ANALYSIS_PATH}")
    return clipped


def profile(name, path, aoi):
    gdf = load(path)
    print(f"\n=== {name}: {path.name} ===")
    print(f"CRS: {gdf.crs.to_string() if gdf.crs else None}")
    if gdf.crs is None or gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(config.CRS)
    print(f"Features: {len(gdf):,}")
    print(f"Geometry types: {gdf.geom_type.value_counts().to_dict()}")
    invalid = (~gdf.is_valid).sum()
    empty = gdf.is_empty.sum()
    print(f"Invalid: {invalid:,}  Empty: {empty:,}")
    print(f"Columns: {[c for c in gdf.columns if c != gdf.geometry.name]}")

    inside = gdf[gdf.intersects(aoi)]
    geoms = inside.geometry.make_valid()
    areas = geoms.area
    clipped_area = geoms.intersection(aoi).area.sum()
    print(f"Inside analysis AOI: {len(inside):,} features, "
          f"{clipped_area / 1e6:.3f} km2 footprint area "
          f"({100 * clipped_area / aoi.area:.1f}% of AOI)")
    print(f"Footprint area m2: min {areas.min():.1f}, median {areas.median():.1f}, "
          f"mean {areas.mean():.1f}, max {areas.max():.1f}")
    print(f"Under 10 m2: {(areas < 10).sum():,}  Under 20 m2: {(areas < 20).sum():,}")

    return {
        "dataset": name,
        "file": path.name,
        "source_crs": gdf.crs.to_string(),
        "features_total": len(gdf),
        "invalid": int(invalid),
        "features_in_aoi": len(inside),
        "footprint_km2_in_aoi": round(clipped_area / 1e6, 4),
        "coverage_pct": round(100 * clipped_area / aoi.area, 2),
        "median_area_m2": round(areas.median(), 1),
        "under_20m2": int((areas < 20).sum()),
    }


def main():
    aoi = analysis_aoi()
    datasets = {"ground_truth": config.GROUND_TRUTH_PATH, **config.BENCHMARKS}
    rows = [profile(name, path, aoi) for name, path in datasets.items()]

    config.OUTPUTS.mkdir(exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(config.OUTPUTS / "vector_profile.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
