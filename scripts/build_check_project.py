r"""Build outputs/check_results.qgz: a QGIS project for checking the results by eye.

Run with QGIS's Python (PyQGIS), e.g. on Windows:
    "C:\Program Files\QGIS 3.44.13\bin\python-qgis-ltr.bat" scripts\build_check_project.py

Layers, top to bottom (paths are relative, so the project moves with the repo):
  Check        LiDAR false shapes (red), LiDAR footprints (blue outline), OS
               buildings LiDAR misses (orange: under 50% covered), quadrant
               boxes (each scored by a model trained without it)
  Inspect      (off) OS buildings shaded by LiDAR coverage, LiDAR footprints by
               their label against OS with the per-footprint statistics from
               05_diagnose_objects.py, and by height (10_building_heights.py)
  Benchmarks   (off) Microsoft, OSM and OS Open footprints and their false shapes
  Rasters      (off) nDSM height, Sentinel-2 NDVI, DSM hillshade
  Imagery      Google and Bing satellite tiles, for viewing only
"""

import sys
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsGradientColorRamp,
    QgsGraduatedSymbolRenderer,
    QgsHillshadeRenderer,
    QgsPalLayerSettings,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsRectangle,
    QgsReferencedRectangle,
    QgsRendererCategory,
    QgsRendererRange,
    QgsSingleBandPseudoColorRenderer,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "check_results.qgz"
CRS = QgsCoordinateReferenceSystem("EPSG:27700")
BOUNDS = (290000.0, 90000.0, 300000.0, 100000.0)  # as config.ANALYSIS_BOUNDS
QUADRANTS = {"SX99SW urban": (0, 0), "SX99SE urban": (1, 0),
             "SX99NW rural": (0, 1), "SX99NE rural": (1, 1)}


def fill(color, outline, width=0.4, alpha=255):
    """Fill symbol; alpha 0 draws the outline only."""
    c = QColor(color)
    return QgsFillSymbol.createSimple({"color": f"{c.red()},{c.green()},{c.blue()},{alpha}",
                                       "outline_color": outline, "outline_width": str(width),
                                       "style": "solid" if alpha else "no"})


def vector(path, name, layer=None, subset=None):
    uri = str(path) + (f"|layername={layer}" if layer else "")
    lyr = QgsVectorLayer(uri, name, "ogr")
    if not lyr.isValid():
        print(f"Skipped {name}: could not open {uri}", file=sys.stderr)
        return None
    if subset:
        lyr.setSubsetString(subset)
    return lyr


def quadrant_layer():
    """Quadrant boxes, written to outputs/eval/quadrants.gpkg so the project can load them."""
    path = ROOT / "outputs" / "eval" / "quadrants.gpkg"
    fields = QgsFields()
    fields.append(QgsField("quadrant", QMetaType.Type.QString))
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = "GPKG"
    writer = QgsVectorFileWriter.create(str(path), fields, Qgis.WkbType.Polygon, CRS,
                                        QgsProject.instance().transformContext(), opts)
    half = (BOUNDS[2] - BOUNDS[0]) / 2
    for name, (i, j) in QUADRANTS.items():
        x, y = BOUNDS[0] + i * half, BOUNDS[1] + j * half
        f = QgsFeature(fields)
        f.setGeometry(QgsGeometry.fromRect(QgsRectangle(x, y, x + half, y + half)))
        f.setAttribute("quadrant", name)
        writer.addFeature(f)
    del writer  # flush to disk
    lyr = vector(path, "Quadrants")
    lyr.renderer().setSymbol(fill("#000000", "#ffff00", 0.8, 0))
    label = QgsPalLayerSettings()
    label.fieldName = "quadrant"
    fmt = label.format(); fmt.setColor(QColor("#ffff00")); fmt.setSize(14); label.setFormat(fmt)
    lyr.setLabeling(QgsVectorLayerSimpleLabeling(label))
    lyr.setLabelsEnabled(True)
    return lyr


def coverage_layer():
    """OS buildings shaded by how much of each LiDAR covers (red 0 -> green 1)."""
    lyr = vector(ROOT / "outputs" / "eval" / "gt_coverage.gpkg", "OS buildings by LiDAR coverage")
    ranges = [(0.0, 0.25, "#d7191c", "< 25%"), (0.25, 0.5, "#fdae61", "25-50%"),
              (0.5, 0.75, "#a6d96a", "50-75%"), (0.75, 1.01, "#1a9641", "75-100%")]
    lyr.setRenderer(QgsGraduatedSymbolRenderer("cov_lidar", [
        QgsRendererRange(lo, hi, fill(col, "#333333", 0.1), label) for lo, hi, col, label in ranges]))
    return lyr


def label_layer():
    """LiDAR footprints coloured by their label against OS (05_diagnose_objects.py)."""
    lyr = vector(ROOT / "outputs" / "diagnostics" / "lidar_objects.gpkg", "LiDAR footprints by label")
    if lyr is None:
        return None
    cats = [("match", "#1a9641"), ("partial", "#fdae61"), ("ignore", "#999999"), ("false", "#d7191c")]
    lyr.setRenderer(QgsCategorizedSymbolRenderer("label", [
        QgsRendererCategory(v, fill(c, c, 0.3, 120), v) for v, c in cats]))
    return lyr


def height_layer():
    """LiDAR footprints shaded by maximum height above ground (10_building_heights.py)."""
    lyr = vector(ROOT / "outputs" / "footprints" / "lidar_classified.gpkg", "LiDAR footprints by height (max)")
    if lyr is None or lyr.fields().indexOf("height_max_m") < 0:
        return None
    ranges = [(0, 5, "#ffffb2", "under 5 m"), (5, 8, "#fecc5c", "5-8 m"), (8, 12, "#fd8d3c", "8-12 m"),
              (12, 20, "#f03b20", "12-20 m"), (20, 200, "#bd0026", "20 m and over")]
    lyr.setRenderer(QgsGraduatedSymbolRenderer("height_max_m", [
        QgsRendererRange(lo, hi, fill(col, "#333333", 0.1, 220), label) for lo, hi, col, label in ranges]))
    return lyr


def raster(path, name):
    lyr = QgsRasterLayer(str(path), name)
    return lyr if lyr.isValid() else None


def pseudocolor(lyr, lo, hi, colors):
    ramp = QgsGradientColorRamp(QColor(colors[0]), QColor(colors[-1]))
    shader_fn = QgsColorRampShader(lo, hi, ramp)
    shader_fn.classifyColorRamp(5)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(shader_fn)
    lyr.setRenderer(QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), 1, shader))


def main() -> int:
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.setTitle("Building footprints: check results")
    project.setCrs(CRS)
    project.writeEntryBool("Paths", "/Absolute", False)  # relative paths
    root = project.layerTreeRoot()
    out = ROOT / "outputs"

    def add(group, lyr, visible=True):
        if lyr is None:
            return
        project.addMapLayer(lyr, False)
        group.addLayer(lyr).setItemVisibilityChecked(visible)

    check = root.addGroup("Check")
    false_lidar = vector(out / "eval" / "false_footprints.gpkg", "LiDAR false shapes (<10% on OS)", "lidar")
    false_lidar.renderer().setSymbol(fill("#d7191c", "#d7191c", 0.8, 90))
    add(check, false_lidar)
    lidar = vector(out / "footprints" / "lidar_classified.gpkg", "LiDAR footprints")
    lidar.renderer().setSymbol(fill("#000000", "#00b4ff", 0.5, 0))
    add(check, lidar)
    missed = vector(out / "eval" / "gt_coverage.gpkg", "OS buildings LiDAR misses (<50% covered)",
                    subset='"cov_lidar" < 0.5')
    missed.renderer().setSymbol(fill("#ff8c00", "#ff8c00", 0.3, 110))
    add(check, missed)
    add(check, quadrant_layer())

    inspect = root.addGroup("Inspect")
    inspect.setItemVisibilityChecked(False)
    add(inspect, coverage_layer(), False)
    add(inspect, label_layer(), False)
    add(inspect, height_layer(), False)

    bench = root.addGroup("Benchmarks")
    bench.setItemVisibilityChecked(False)
    data = ROOT / "data" / "footprints"
    for name, file, colour in (("Microsoft", "microsoft_buildings.parquet", "#e7298a"),
                               ("OpenStreetMap", "osm_buildings.parquet", "#7570b3"),
                               ("OS Open", "os_buildings.parquet", "#66a61e")):
        lyr = vector(data / file, f"{name} footprints")
        if lyr:
            lyr.renderer().setSymbol(fill("#000000", colour, 0.4, 0))
        add(bench, lyr, False)
    for layer, name in (("microsoft", "Microsoft"), ("osm", "OpenStreetMap"), ("os_open", "OS Open")):
        lyr = vector(out / "eval" / "false_footprints.gpkg", f"{name} false shapes", layer)
        if lyr:
            lyr.renderer().setSymbol(fill("#d7191c", "#d7191c", 0.5, 60))
        add(bench, lyr, False)

    rasters = root.addGroup("Rasters")
    rasters.setItemVisibilityChecked(False)
    processed = ROOT / "data" / "processed"
    ndsm = raster(processed / "ndsm.tif", "nDSM: height above ground (m)")
    if ndsm:
        pseudocolor(ndsm, 0, 20, ["#ffffff", "#08306b"])
    add(rasters, ndsm, False)
    ndvi = raster(processed / "ndvi_s2_2022.tif", "Sentinel-2 NDVI, summer 2022")
    if ndvi:
        pseudocolor(ndvi, 0, 0.9, ["#a6611a", "#018571"])
    add(rasters, ndvi, False)
    dsm = raster(processed / "dsm.tif", "DSM hillshade")
    if dsm:
        dsm.setRenderer(QgsHillshadeRenderer(dsm.dataProvider(), 1, 315, 45))
    add(rasters, dsm, False)

    imagery = root.addGroup("Imagery (viewing only)")
    for name, url in (
        ("Google satellite", "type=xyz&url=https://mt1.google.com/vt/lyrs%3Ds%26x%3D%7Bx%7D%26y%3D%7By%7D%26z%3D%7Bz%7D&zmax=20"),
        ("Bing aerial", "type=xyz&url=http://ecn.t3.tiles.virtualearth.net/tiles/a%7Bq%7D.jpeg?g%3D1&zmax=19"),
    ):
        add(imagery, QgsRasterLayer(url, name, "wms"), name == "Google satellite")

    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(QgsRectangle(*BOUNDS), CRS))
    ok = project.write(str(OUT))
    print(f"{'Wrote' if ok else 'FAILED to write'} {OUT}")
    app.exitQgis()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
