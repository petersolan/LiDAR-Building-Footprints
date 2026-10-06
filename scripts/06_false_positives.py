"""How much of each dataset's footprint area lies outside the OS buildings?

Compared against *all* OS buildings (scored and ignored alike), so a shed that
OS maps is not counted as false here. Predicted area outside every OS building
is split into:
  edge      within 2 m of an OS building (outline misalignment, eaves, walls)
  isolated  more than 2 m from any OS building (nothing mapped there)

Also counts whole false footprints: features with < 10% of their area on OS
buildings. Results by dataset and quadrant go to outputs/eval/false_positives.csv
and the false footprints to outputs/eval/false_footprints.gpkg.
"""

import importlib

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import config
from groundtruth import load_ground_truth

ev = importlib.import_module("04_evaluate")

EDGE_M = 2.0
FALSE_MAX_SHARE = 0.1


def main():
    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()
    scored, ignored = load_ground_truth(aoi)
    os_parts = ev.blocks(np.concatenate([scored.geometry.values, ignored.geometry.values]))
    os_u = shapely.union_all(os_parts)
    near_u = shapely.buffer(os_u, EDGE_M)
    print(f"OS buildings (all): {os_u.area / 1e6:.3f} km2", flush=True)

    rows, false_feats = [], []
    for name, path in ev.DATASETS.items():
        pr = ev.load(path, aoi)
        geoms = ev.polygons(shapely.intersection(pr.geometry.values, aoi))
        pr_u = shapely.union_all(geoms)

        outside = shapely.difference(pr_u, os_u)
        isolated = shapely.difference(outside, near_u)

        share = ev.overlap_fraction(geoms, os_parts)
        false_geoms = geoms[share < FALSE_MAX_SHARE]
        false_feats.append(gpd.GeoDataFrame(
            {"dataset": name, "area_m2": shapely.area(false_geoms).round(1)},
            geometry=false_geoms, crs=config.CRS))

        for q, qbox in [("all", aoi)] + list(ev.QUADRANTS.items()):
            pa = shapely.intersection(pr_u, qbox).area
            out = shapely.intersection(outside, qbox).area
            iso = shapely.intersection(isolated, qbox).area
            in_q = shapely.intersects(false_geoms, qbox) if q != "all" else np.ones(len(false_geoms), bool)
            rows.append(dict(
                dataset=name, quadrant=q,
                pred_km2=pa / 1e6,
                outside_km2=out / 1e6, outside_pct=100 * out / pa,
                edge_km2=(out - iso) / 1e6, isolated_km2=iso / 1e6,
                isolated_pct=100 * iso / pa,
                false_features=int(in_q.sum()),
                false_features_pct=100 * in_q.sum() / max(shapely.intersects(geoms, qbox).sum(), 1),
            ))
        print(f"{name} done", flush=True)

    df = pd.DataFrame(rows)
    ev.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(ev.EVAL_DIR / "false_positives.csv", index=False)
    pd.concat(false_feats).to_file(ev.EVAL_DIR / "false_footprints.gpkg", driver="GPKG",
                                   layer="false_footprints")

    pd.set_option("display.width", 200)
    print("\nWhole AOI:")
    print(df[df.quadrant == "all"].drop(columns="quadrant").round(3).to_string(index=False))
    print("\nIsolated false area (% of predicted area) by quadrant:")
    print(df[df.quadrant != "all"].pivot(index="quadrant", columns="dataset",
                                         values="isolated_pct").round(1).to_string())
    print("\nWhole false features by quadrant:")
    print(df[df.quadrant != "all"].pivot(index="quadrant", columns="dataset",
                                         values="false_features").to_string())


if __name__ == "__main__":
    main()
