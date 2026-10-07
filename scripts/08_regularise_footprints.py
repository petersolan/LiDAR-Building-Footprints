"""Turn the LiDAR building mask into clean, footprint-like polygons.

1. Trace:      each connected building (4-connectivity, as in the vector
               output) is traced with marching squares at the 0.5 level, so
               outlines run between pixel centres instead of along pixel edges.
2. Simplify:   Douglas-Peucker removes the remaining pixel stair-steps.
3. Square up:  each footprint's dominant orientation comes from the
               length-weighted mean of its edge directions (modulo 90 degrees).
               Edges within a tolerance of that orientation or its
               perpendicular snap to it; other edges keep their direction.
               Consecutive parallel edges merge when the step between them is
               small, otherwise a perpendicular connector is inserted. Corners
               are rebuilt by intersecting consecutive edges.
   Road alignment: if a road centreline (OS Open Roads) lies within
               road_max_dist_m and its local direction is within road_snap_deg
               of the footprint's own orientation (modulo 90 degrees), the
               road's direction is used instead, so rows of houses line up with
               the street and each other.
4. Fall back:  if squaring changes the shape too much (IoU with the traced
               outline below min_iou), the simplified outline is kept, so
               curved and irregular buildings are left alone.
5. No overlaps: where two squared footprints overlap, the overlap goes to the
               footprint whose traced outline covers more of it; the other is
               cut back. Neighbours may share edges but never overlap.

Uses only the LiDAR mask (no OS geometry). Run with --sample to process four
500 m test tiles (written with the matching OS, Microsoft, OSM and OS Open
footprints for comparison), or --full for the whole AOI.
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.windows import from_bounds
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage import measure

import config
from groundtruth import load_ground_truth

MASK_PATH = config.OUTPUTS / "footprints" / "lidar_baseline_mask.tif"
OUT_FULL = config.OUTPUTS / "footprints" / "lidar_regularised.gpkg"
OUT_SAMPLE = config.OUTPUTS / "footprints" / "regularise_samples.gpkg"

PARAMS = dict(
    simplify_m=0.75,    # Douglas-Peucker tolerance after tracing
    snap_deg=44.0,      # edges within this angle of the main axes are snapped (45 = all)
    step_m=0.75,        # parallel edges closer than this merge into one
    min_iou=0.85,       # below this, keep the simplified outline instead
    min_area_m2=20.0,   # drop pieces smaller than this after overlap removal
    use_roads=False,    # align orientation to nearby roads (tested: no measurable gain)
    road_max_dist_m=30.0,   # roads further than this are ignored
    road_snap_deg=15.0,     # adopt the road direction if within this angle
)
# Roads used for alignment: motorways, roundabouts and slip roads rarely set
# the orientation of the buildings beside them
ROAD_EXCLUDE_FUNCTION = ["Motorway"]
ROAD_EXCLUDE_FORM = ["Roundabout", "Slip Road"]
SAMPLE_SIZE = 500.0
NEAR_PARALLEL = np.sin(np.radians(5))


# --- tracing -----------------------------------------------------------------

def trace(component, row0, col0, transform):
    """Polygon (with holes) from a boolean component, contours between pixel centres."""
    padded = np.pad(component, 1)
    rings = []
    for c in measure.find_contours(padded.astype("float32"), 0.5):
        rows, cols = c[:, 0] - 1 + row0, c[:, 1] - 1 + col0
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        ring = Polygon(np.column_stack([xs, ys]))
        if ring.is_valid and ring.area > 0:
            rings.append(ring)
    if not rings:
        return None
    rings.sort(key=lambda r: r.area, reverse=True)
    outer = rings[0]
    holes = [r.exterior.coords for r in rings[1:] if outer.contains(r)]
    return Polygon(outer.exterior.coords, holes)


# --- squaring ----------------------------------------------------------------

def dominant_angle(ring):
    xy = np.asarray(ring.coords)
    v = np.diff(xy, axis=0)
    length = np.hypot(v[:, 0], v[:, 1])
    ang = np.arctan2(v[:, 1], v[:, 0])
    return np.arctan2((length * np.sin(4 * ang)).sum(), (length * np.cos(4 * ang)).sum()) / 4


def intersect(p1, d1, p2, d2):
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < NEAR_PARALLEL:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def square_ring(xy, theta, p):
    """Square up one closed ring; returns vertex array or None."""
    pts = np.asarray(xy)[:-1]
    n = len(pts)
    if n < 3:
        return None
    tol = np.radians(p["snap_deg"])
    lines = []  # [point, direction, axis (0/1/None), length, original end vertex]
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        v = b - a
        length = np.hypot(*v)
        if length == 0:
            continue
        ang = np.arctan2(v[1], v[0])
        rel = (ang - theta) % (np.pi / 2)
        axis = None
        if min(rel, np.pi / 2 - rel) <= tol:
            k90 = int(np.round((ang - theta) / (np.pi / 2)))
            ang = theta + k90 * np.pi / 2
            axis = k90 % 2
        lines.append([(a + b) / 2, np.array([np.cos(ang), np.sin(ang)]), axis, length, b])
    if len(lines) < 3:
        return None

    # Rotate so the list doesn't start in the middle of a run of parallel edges
    start = next((i for i in range(len(lines))
                  if lines[i][2] is None or lines[i][2] != lines[i - 1][2]), 0)
    lines = lines[start:] + lines[:start]

    merged = []
    for ln in lines:
        prev = merged[-1] if merged else None
        if prev is not None and ln[2] is not None and ln[2] == prev[2]:
            normal = np.array([-prev[1][1], prev[1][0]])
            step = np.dot(ln[0] - prev[0], normal)
            if abs(step) < p["step_m"]:
                w = prev[3] + ln[3]
                offset = (np.dot(prev[0], normal) * prev[3] + np.dot(ln[0], normal) * ln[3]) / w
                along = np.dot((prev[0] * prev[3] + ln[0] * ln[3]) / w, prev[1])
                merged[-1] = [along * prev[1] + offset * normal, prev[1], prev[2], w, ln[4]]
                continue
            # Real step: perpendicular connector through the shared original vertex
            merged.append([prev[4], normal, 1 - prev[2], 0.0, prev[4]])
        merged.append(ln)
    if len(merged) > 1 and merged[0][2] is not None and merged[0][2] == merged[-1][2]:
        last, first = merged[-1], merged[0]
        normal = np.array([-last[1][1], last[1][0]])
        if abs(np.dot(first[0] - last[0], normal)) >= p["step_m"]:
            merged.append([last[4], normal, 1 - last[2], 0.0, last[4]])
    if len(merged) < 3:
        return None

    verts = []
    for i, ln in enumerate(merged):
        nxt = merged[(i + 1) % len(merged)]
        v = intersect(ln[0], ln[1], nxt[0], nxt[1])
        # Near-parallel or runaway intersection: keep the original vertex
        if v is None or np.hypot(*(v - ln[4])) > 3 * p["simplify_m"]:
            v = ln[4]
        verts.append(v)
    return np.array(verts)


def angle_diff_90(a, b):
    """Smallest difference between two orientations, modulo 90 degrees (radians)."""
    d = (a - b) % (np.pi / 2)
    return min(d, np.pi / 2 - d)


def regularise(traced, p, roads=None):
    """Return (final, simplified, orientation source)."""
    simple = shapely.make_valid(traced.simplify(p["simplify_m"], preserve_topology=True))
    simple = largest_polygon(simple)
    if simple is None:
        return None, None, None
    theta = dominant_angle(simple.exterior)
    source = "own"
    if roads is not None:
        tree, seg_angle = roads
        idx = tree.query_nearest(simple, max_distance=p["road_max_dist_m"], all_matches=False)
        if len(idx):
            road_theta = seg_angle[idx[0]]
            if angle_diff_90(theta, road_theta) <= np.radians(p["road_snap_deg"]):
                theta, source = road_theta, "road"
    ext = square_ring(simple.exterior.coords, theta, p)
    holes = [h for h in (square_ring(r.coords, theta, p) for r in simple.interiors)
             if h is not None and len(h) >= 3]
    if ext is None or len(ext) < 3:
        return simple, simple, "fallback"
    squared = largest_polygon(shapely.make_valid(Polygon(ext, holes)))
    if squared is None:
        return simple, simple, "fallback"
    iou = squared.intersection(traced).area / squared.union(traced).area
    if iou < p["min_iou"]:
        return simple, simple, "fallback"
    return squared, simple, source


def largest_polygon(geom):
    parts = [g for g in shapely.get_parts(geom) if g.geom_type == "Polygon" and g.area > 0]
    return max(parts, key=lambda g: g.area) if parts else None


# --- overlaps ----------------------------------------------------------------

def remove_overlaps(squared, traced, p):
    """Give each overlap to the footprint whose traced outline covers more of it."""
    out = list(squared)
    tree = shapely.STRtree(out)
    a_idx, b_idx = tree.query(out, predicate="overlaps")
    for i, j in zip(a_idx, b_idx):
        if i >= j or out[i] is None or out[j] is None:
            continue
        overlap = out[i].intersection(out[j])
        if overlap.area < 1e-6:
            continue
        keep_i = overlap.intersection(traced[i]).area >= overlap.intersection(traced[j]).area
        loser = j if keep_i else i
        out[loser] = largest_polygon(shapely.make_valid(out[loser].difference(overlap)))
    return [g if g is not None and g.area >= p["min_area_m2"] else None for g in out]


# --- roads -------------------------------------------------------------------

def load_roads(bounds):
    """STRtree of single road segments within bounds, and each segment's angle."""
    roads = gpd.read_file(config.ROADS_PATH, bbox=bounds)
    roads = roads[~roads["function"].isin(ROAD_EXCLUDE_FUNCTION)
                  & ~roads["formOfWay"].isin(ROAD_EXCLUDE_FORM)]
    segs, angles = [], []
    for line in shapely.force_2d(roads.geometry.values):
        xy = shapely.get_coordinates(line)
        if len(xy) < 2:
            continue
        a, b = xy[:-1], xy[1:]
        ok = np.hypot(*(b - a).T) > 0
        segs.append(shapely.linestrings(np.stack([a[ok], b[ok]], axis=1)))
        angles.append(np.arctan2(*(b[ok] - a[ok]).T[::-1]))
    segs = np.concatenate(segs)
    return shapely.STRtree(segs), np.concatenate(angles)


# --- driver ------------------------------------------------------------------

def process(mask, transform, crs, p=PARAMS, roads=None):
    labels, n = ndi.label(mask)  # default structure = 4-connectivity
    traced, squared, simple, sources = [], [], [], []
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = labels[sl] == lab
        t = trace(comp, sl[0].start, sl[1].start, transform)
        if t is None:
            continue
        sq, si, src = regularise(t, p, roads if p["use_roads"] else None)
        if sq is None:
            continue
        traced.append(t); squared.append(sq); simple.append(si); sources.append(src)
    final = remove_overlaps(squared, traced, p)
    keep = [i for i, g in enumerate(final) if g is not None]
    gdf = gpd.GeoDataFrame({"area_m2": [round(final[i].area, 1) for i in keep],
                            "vertices": [len(final[i].exterior.coords) - 1 for i in keep],
                            "orientation": [sources[i] for i in keep]},
                           geometry=[final[i] for i in keep], crs=crs)
    traced_gdf = gpd.GeoDataFrame(geometry=traced, crs=crs)
    return gdf, traced_gdf


def sample_windows(scored):
    """Four 500 m tiles: densest urban, mid-density urban, suburban, rural."""
    minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
    xs = np.arange(minx, maxx, SAMPLE_SIZE); ys = np.arange(miny, maxy, SAMPLE_SIZE)
    cells = [shapely.box(x, y, x + SAMPLE_SIZE, y + SAMPLE_SIZE) for x in xs for y in ys]
    pts = shapely.centroid(scored.geometry.values)
    tree = shapely.STRtree(cells)
    cell_idx, pt_idx = tree.query(pts, predicate="within")[[1, 0]]
    area = np.bincount(cell_idx, shapely.area(scored.geometry.values)[pt_idx], len(cells))
    detached = np.bincount(cell_idx, (scored["description"].values[pt_idx] == "Detached House"),
                           len(cells))
    order = np.argsort(area)
    picks = {"dense urban": order[-1],
             "mid-density": order[int(len(order) * 0.80)],
             "suburban detached": int(np.argmax(detached)),
             "rural": next(i for i in order if area[i] > 3000 and shapely.get_y(
                 shapely.centroid(cells[i])) > 95000)}
    return {name: cells[i] for name, i in picks.items()}


def shape_metrics(gdf, reference_boundary_tree, reference_area_u, window):
    polys = [g for g in shapely.get_parts(shapely.intersection(gdf.geometry.values, window))
             if g.geom_type == "Polygon" and g.area > 1]
    if not polys:
        return {}
    verts, right, corners = [], 0, 0
    for g in polys:
        xy = np.asarray(g.exterior.coords)[:-1]
        verts.append(len(xy))
        v1 = np.roll(xy, -1, axis=0) - xy
        v0 = xy - np.roll(xy, 1, axis=0)
        cosang = (v0 * v1).sum(1) / (np.hypot(*v0.T) * np.hypot(*v1.T) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))  # turning angle
        corners += len(ang)
        right += int((np.abs(ang - 90) <= 10).sum())
    union = shapely.union_all(polys)
    pts = shapely.points(shapely.get_coordinates(shapely.segmentize(union.boundary, 0.5)))
    _, d = reference_boundary_tree.query_nearest(pts, return_distance=True, all_matches=False)
    ref = shapely.intersection(reference_area_u, window)
    return dict(polygons=len(polys), median_vertices=float(np.median(verts)),
                right_angle_pct=100 * right / max(corners, 1),
                edge_within_0_5m_pct=100 * np.mean(d <= 0.5),
                edge_within_1m_pct=100 * np.mean(d <= 1.0),
                area_iou=union.intersection(ref).area / union.union(ref).area)


def orientation_error(gdf, os_geoms, window):
    """Median angle (degrees, modulo 90) between each footprint's dominant
    orientation and that of the OS building it overlaps most."""
    polys = [largest_polygon(g) for g in gdf.geometry.values if g.intersects(window)]
    polys = [g for g in polys if g is not None and g.area > 20]
    tree = shapely.STRtree(os_geoms)
    errs = []
    for g in polys:
        cand = tree.query(g, predicate="intersects")
        if not len(cand):
            continue
        best = max(cand, key=lambda k: g.intersection(os_geoms[k]).area)
        ref = largest_polygon(os_geoms[best])
        if ref is None:
            continue
        errs.append(np.degrees(angle_diff_90(dominant_angle(g.exterior), dominant_angle(ref.exterior))))
    return float(np.median(errs)) if errs else np.nan


def run_samples(p):
    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()
    scored, ignored = load_ground_truth(aoi)
    os_all = pd.concat([scored, ignored])
    windows = sample_windows(scored)

    layers = {"lidar_regularised": [], "lidar_regularised_noroads": [], "lidar_traced": [],
              "lidar_previous": [], "roads": []}
    previous = gpd.read_file(config.OUTPUTS / "footprints" / "lidar_baseline.gpkg")
    with rasterio.open(MASK_PATH) as src:
        for name, win in windows.items():
            w = from_bounds(*win.bounds, transform=src.transform)
            # Pad by 30 m so buildings crossing the tile edge are traced whole
            wp = from_bounds(*win.buffer(30, join_style="mitre").bounds, transform=src.transform)
            mask = src.read(1, window=wp).astype(bool)
            pad_bounds = win.buffer(30 + p["road_max_dist_m"], join_style="mitre").bounds
            roads = load_roads(pad_bounds)
            reg, traced = process(mask, src.window_transform(wp), src.crs, p, roads)
            reg_nr, _ = process(mask, src.window_transform(wp), src.crs, dict(p, use_roads=False))
            road_lines = gpd.GeoDataFrame(geometry=roads[0].geometries, crs=config.CRS)
            for key, g in (("lidar_regularised", reg), ("lidar_regularised_noroads", reg_nr),
                           ("lidar_traced", traced), ("roads", road_lines)):
                g = g[g.intersects(win)].copy(); g["tile"] = name
                layers[key].append(g)
            prev = previous[previous.intersects(win)].copy(); prev["tile"] = name
            layers["lidar_previous"].append(prev)
            r = layers["lidar_regularised"][-1]
            print(f"{name}: {len(r)} footprints, orientation from road "
                  f"{100 * (r['orientation'] == 'road').mean():.0f}%, own "
                  f"{100 * (r['orientation'] == 'own').mean():.0f}%, "
                  f"fallback {100 * (r['orientation'] == 'fallback').mean():.0f}%", flush=True)

    OUT_SAMPLE.unlink(missing_ok=True)
    tiles = gpd.GeoDataFrame({"tile": list(windows)}, geometry=list(windows.values()), crs=config.CRS)
    tiles.to_file(OUT_SAMPLE, layer="sample_tiles")
    for key, frames in layers.items():
        pd.concat(frames).to_file(OUT_SAMPLE, layer=key)

    datasets = {k: pd.concat(v) for k, v in layers.items()}
    for name, path in config.BENCHMARKS.items():
        datasets[name] = gpd.read_parquet(path)
    for name, frame in datasets.items():
        if name in config.BENCHMARKS:
            clip = frame[frame.intersects(shapely.union_all(list(windows.values())))]
            clip.to_file(OUT_SAMPLE, layer=name)
    os_clip = os_all[os_all.intersects(shapely.union_all(list(windows.values())))]
    os_clip[["description", "height_relativemax_m", "geometry"]].to_file(OUT_SAMPLE, layer="os_ground_truth")

    os_u = shapely.union_all(os_clip.geometry.values)
    segs = shapely.get_parts(shapely.segmentize(os_u.boundary, 1.0))
    os_tree = shapely.STRtree(segs)
    rows = []
    for tile, win in windows.items():
        for name, frame in datasets.items():
            if name in ("lidar_traced", "roads"):
                continue
            sub = frame[frame.intersects(win)]
            rows.append(dict(tile=tile, dataset=name, **shape_metrics(sub, os_tree, os_u, win),
                             orient_err_deg=orientation_error(sub, os_clip.geometry.values, win)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    summary = df.groupby("dataset")[["polygons", "median_vertices", "right_angle_pct",
                                     "edge_within_0_5m_pct", "edge_within_1m_pct", "area_iou",
                                     "orient_err_deg"]].mean()
    print("\nMean over the four tiles (edges/IoU against all OS buildings):")
    print(summary.round(2).to_string())
    print("\nPer tile:")
    print(df.round(2).to_string(index=False))
    print(f"\nWrote {OUT_SAMPLE}")


def run_full(p):
    roads = load_roads(config.ANALYSIS_BOUNDS) if p["use_roads"] else None
    with rasterio.open(MASK_PATH) as src:
        mask = src.read(1).astype(bool)
        gdf, _ = process(mask, src.transform, src.crs, p, roads)
    gdf.to_file(OUT_FULL, layer="lidar_regularised")
    print(f"{len(gdf):,} footprints -> {OUT_FULL}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="process the whole AOI")
    args = parser.parse_args()
    (run_full if args.full else run_samples)(PARAMS)


if __name__ == "__main__":
    main()
