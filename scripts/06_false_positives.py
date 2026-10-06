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
    scored_u = shapely.union_all(ev.polygons(scored.geometry.values))
    near_u = shapely.buffer(os_u, EDGE_M)
    regions = [("all", aoi)] + list(ev.QUADRANTS.items())

    # Ground truth building area per quadrant, before and after filtering
    gt_rows = [dict(quadrant=q,
                    os_all_m2=shapely.intersection(os_u, r).area,
                    os_scored_m2=shapely.intersection(scored_u, r).area) for q, r in regions]
    print(f"OS buildings (all): {os_u.area:,.0f} m2", flush=True)

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

        for q, region in regions:
            pa = shapely.intersection(pr_u, region).area
            out = shapely.intersection(outside, region).area
            iso = shapely.intersection(isolated, region).area
            in_q = shapely.intersects(false_geoms, region)
            rows.append(dict(
                dataset=name, quadrant=q,
                pred_m2=pa,
                outside_m2=out, outside_pct=100 * out / pa,
                edge_m2=out - iso, isolated_m2=iso, isolated_pct=100 * iso / pa,
                false_features=int(in_q.sum()),
                false_features_m2=shapely.area(shapely.intersection(false_geoms[in_q], region)).sum(),
            ))
        print(f"{name} done", flush=True)

    gt = pd.DataFrame(gt_rows)
    df = pd.DataFrame(rows).merge(gt, on="quadrant")
    m2 = ["pred_m2", "outside_m2", "edge_m2", "isolated_m2", "false_features_m2",
          "os_all_m2", "os_scored_m2"]
    df[m2] = df[m2].round(0).astype("int64")
    ev.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(ev.EVAL_DIR / "false_positives.csv", index=False)
    pd.concat(false_feats).to_file(ev.EVAL_DIR / "false_footprints.gpkg", driver="GPKG",
                                   layer="false_footprints")

    pd.set_option("display.width", 220)
    print("\nGround truth building area (m2):")
    print(gt.set_index("quadrant").round(0).astype("int64").to_string())
    print("\nWhole AOI (m2):")
    cols = ["dataset", "pred_m2", "outside_m2", "edge_m2", "isolated_m2", "isolated_pct",
            "false_features", "false_features_m2"]
    print(df[df.quadrant == "all"][cols].round(1).to_string(index=False))
    for col, title in (("isolated_m2", "Isolated false area (m2)"),
                       ("false_features_m2", "Area of whole false features (m2)"),
                       ("false_features", "Whole false features (count)")):
        t = df.pivot(index="quadrant", columns="dataset", values=col)
        t.insert(0, "OS all m2", gt.set_index("quadrant")["os_all_m2"].round(0).astype("int64"))
        print(f"\n{title} by quadrant:")
        print(t.to_string())


if __name__ == "__main__":
    main()
