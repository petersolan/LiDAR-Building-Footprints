"""Grid-search extraction parameters in parallel (pixel-level, with ignore zones).

Calibration quadrants: SW (urban) + NE (rural). Hold-out: SE (urban) + NW (rural).
The objective is the mean of the calibration quadrants' F1, so the rural
quadrant, where vegetation errors matter, weighs as much as the city centre.
Results go to outputs/tuning/grid_results.csv.
"""

import importlib
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

import config
from groundtruth import load_ground_truth

ext = importlib.import_module("03_extract_footprints")

QUADRANTS = {  # row/col slices of the 10000 x 10000 grid (row 0 = north)
    "sw": (slice(5000, None), slice(0, 5000)),
    "ne": (slice(0, 5000), slice(5000, None)),
    "se": (slice(5000, None), slice(5000, None)),
    "nw": (slice(0, 5000), slice(0, 5000)),
}
CALIBRATION, HOLDOUT = ["sw", "ne"], ["se", "nw"]

# History: a coarse grid (max_roughness 1.25-2.0, min_height 2.5/3.0, open_radius
# 1-3, grow_px 1-3, min_smooth_frac 0-0.6) and a refinement (max_roughness
# 0.75-1.25, open_radius 0/1, grow_px 3-5) chose max_roughness=1.0,
# open_radius=0, grow_px=4. This grid adds the Sentinel-2 NDVI object filter;
# with NDVI removing trees, a looser roughness threshold may recover roofs.
# Result: NDVI on all footprints cost urban recall (houses joined to garden trees),
# so the filter was limited to footprints < ndvi_max_area_m2; a follow-up comparison
# chose max_roughness=1.25, grow_px=4, max_object_ndvi=0.7, ndvi_max_area_m2=400.
GRID = dict(
    max_roughness=[1.0, 1.25, 1.5, 1.75],
    grow_px=[3, 4],
    max_object_ndvi=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0],
)

CACHE = config.PROCESSED / "tuning_cache"
ARRAYS = ("ndsm", "rough", "gt", "ign", "ndvi")
_DATA = {}


def load_data():
    with rasterio.open(config.NDSM_PATH) as s:
        ndsm, tr = s.read(1), s.transform
    with rasterio.open(config.DSM_PATH) as s:
        rough = ext.roughness(s.read(1))
    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()
    scored, ignored = load_ground_truth(aoi)
    raster = lambda g: rasterize(((x, 1) for x in g), out_shape=ndsm.shape, transform=tr,
                                 dtype="uint8").astype(bool)
    gt = raster(scored.geometry)
    ign = raster(ignored.geometry) & ~gt
    ndvi = ext.load_ndvi(ndsm.shape)
    if ndvi is None:
        ndvi = np.zeros_like(ndsm)
    CACHE.mkdir(parents=True, exist_ok=True)
    for q, r in QUADRANTS.items():
        for name, a in zip(ARRAYS, (ndsm, rough, gt, ign, ndvi)):
            np.save(CACHE / f"{q}_{name}.npy", a[r])


def init_worker():
    for q in QUADRANTS:
        _DATA[q] = tuple(np.load(CACHE / f"{q}_{n}.npy", mmap_mode="r") for n in ARRAYS)


def score(pred, gt, ign):
    v = ~ign
    tp = (pred & gt & v).sum(); fp = (pred & ~gt & v).sum(); fn = (~pred & gt & v).sum()
    return 2 * tp / (2 * tp + fp + fn), tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def evaluate(params, quads=CALIBRATION):
    p = dict(ext.PARAMS, **params)
    out = dict(params)
    for q in quads:
        ndsm, rough, gt, ign, ndvi = _DATA[q]
        f1, pr, rc = score(ext.extract_mask(ndsm, None, p, rough=rough, ndvi=ndvi), gt, ign)
        out.update({f"f1_{q}": f1, f"p_{q}": pr, f"r_{q}": rc})
    out["objective"] = np.mean([out[f"f1_{q}"] for q in CALIBRATION])
    return out


def main():
    combos = [dict(zip(GRID, c)) for c in itertools.product(*GRID.values())]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print("Preparing quadrant cache...", flush=True)
    load_data()
    print(f"{len(combos)} combinations on {workers} workers", flush=True)

    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        results = []
        for i, r in enumerate(pool.map(evaluate, combos, chunksize=2), 1):
            results.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(combos)} done, best so far "
                      f"{max(x['objective'] for x in results):.3f}", flush=True)

    df = pd.DataFrame(results).sort_values("objective", ascending=False)
    out_dir = config.OUTPUTS / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out_dir / "grid_results.csv", index=False)

    pd.set_option("display.width", 200)
    cols = list(GRID) + ["objective", "f1_sw", "f1_ne", "p_ne", "r_ne"]
    print("\nTop 10 on calibration (SW + NE):")
    print(df[cols].head(10).round(3).to_string(index=False))

    init_worker()
    best = df.iloc[0][list(GRID)].to_dict()
    best = {k: type(GRID[k][0])(v) for k, v in best.items()}
    for name, params in (("current defaults", {k: ext.PARAMS[k] for k in GRID}), ("best", best)):
        r = evaluate(params, quads=CALIBRATION + HOLDOUT)
        print(f"\n{name}: {params}")
        print("  " + "  ".join(f"{q.upper()} F1 {r[f'f1_{q}']:.3f} (P {r[f'p_{q}']:.3f}, "
                               f"R {r[f'r_{q}']:.3f})" for q in CALIBRATION + HOLDOUT))


if __name__ == "__main__":
    main()
