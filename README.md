# Building footprints from LiDAR: Exeter case study

Can open LiDAR elevation data produce building footprints that compete with the
freely available footprint datasets? This project extracts footprints for a
100 km² test area over Exeter, UK (OS grid square **SX99**) from the Environment
Agency's 1 m DSM and DTM, with Sentinel-2 imagery as a vegetation check. It
scores them against Ordnance Survey's detailed building data alongside three
free alternatives: **Microsoft Global ML Building Footprints**,
**OpenStreetMap** and **OS Open** buildings.

## Idea

A building is something that stands above the ground and has a smooth, planar
surface. Trees also stand above the ground, but their canopy is rough and green.
The method uses those properties:

1. **Height above ground (nDSM):** DSM minus DTM. Keep pixels more than 2.5 m tall.
2. **Surface roughness:** the mean absolute Laplacian of the DSM over a 5 × 5 m
   window. Planar roofs score low and tree canopy scores high. Smooth tall
   pixels become building *seeds*.
3. **Grow and tidy:** seeds grow back a few metres into the tall area, which
   recovers roof edges and walls (the height step makes those rough). Holes up
   to 10 m² are filled and tiny blobs dropped.
4. **Object filters:** each footprint must reach 3.5 m at its highest point,
   matching the ground truth filter below. Footprints under 400 m² that are
   green on average (Sentinel-2 NDVI > 0.7) are dropped as vegetation.
5. **Vectorise** and simplify the footprints.

The NDVI comes from a median of clear summer 2022 Sentinel-2 L2A scenes, the
same year as the LiDAR. At 10 m it is too coarse to judge individual roofs, but
it reliably flags tree crowns and hedgerows. It only applies to small
footprints: a house that merges with its garden trees reads as green at 10 m,
and filtering those cost urban recall.

Parameters are tuned by grid search on two quadrants of the AOI: SW, which is
urban, and NE, which is rural. The other two quadrants (SE and NW) are held
out to check the settings generalise.

## Evaluation

Ground truth is the OS National Geographic Database building layer. Outbuildings,
"Unknown Building" and electricity substations are left out, along with buildings
under 3.5 m or without a recorded height. These become **ignore zones**: no
dataset is rewarded or penalised for footprints there.

Each dataset is scored at three levels:

- **Area:** precision, recall, F1 and IoU of the dissolved footprint area. This is
  the headline comparison, since it doesn't depend on how a dataset splits
  terraces into individual houses.
- **Object:** touching buildings are merged into blocks and matched at IoU ≥ 0.5.
- **Per building:** the share of each ground truth building covered, summarised
  by building class and size.

## Results so far

| Dataset | Area precision | Area recall | **Area F1** | IoU | Object F1 | Buildings detected |
|---|---|---|---|---|---|---|
| **LiDAR + Sentinel-2 (this project)** | 0.800 | 0.838 | **0.819** | 0.693 | 0.440 | 85.7% |
| OpenStreetMap | 0.819 | 0.800 | 0.809 | 0.680 | 0.663 | 90.9% |
| Microsoft | 0.758 | 0.800 | 0.779 | 0.638 | 0.659 | 92.3% |
| OS Open | 0.887 | 0.924 | 0.905 | 0.827 | 0.756 | 98.1% |

Area F1 by quadrant:

| Quadrant | LiDAR + S2 | Microsoft | OSM | OS Open |
|---|---|---|---|---|
| SW (urban, calibration) | 0.833 | 0.796 | 0.813 | 0.904 |
| NE (rural, calibration) | 0.753 | 0.716 | 0.773 | 0.879 |
| SE (urban, hold-out) | 0.800 | 0.755 | 0.827 | 0.916 |
| NW (rural, hold-out) | 0.762 | 0.723 | 0.574 | 0.861 |

On area, the footprints score highest of the free datasets and beat Microsoft in
every quadrant, held-out ones included. OS Open probably derives from the same
OS master data as the ground truth, so its lead is expected.

Known weaknesses: neighbouring buildings merge into single shapes (low object
F1), detached houses surrounded by trees are missed (74% detected against about
90% for Microsoft and OSM), and so are buildings under 20 m².

### False detections

This compares each dataset's footprint area outside *all* OS buildings,
including the ignored ones. That area splits into edge effects (within 2 m of
an OS building) and isolated detections (more than 2 m from any OS building).
Whole false shapes are features with under 10% of their area on OS buildings.

| Dataset | Predicted (m²) | Outside OS (m²) | Edge (m²) | Isolated (m²) | Whole false shapes |
|---|---|---|---|---|---|
| **LiDAR + S2** | 5,704,028 | 1,045,213 | 765,518 | 279,695 | 827 (180,576 m²) |
| Microsoft | 5,832,843 | 1,275,323 | 988,319 | 287,004 | 801 (89,573 m²) |
| OSM | 5,302,737 | 883,818 | 731,440 | 152,379 | 408 (37,523 m²) |
| OS Open | 5,849,765 | 587,784 | 512,283 | 75,501 | 19 (4,835 m²) |

Isolated false area by quadrant, next to the OS building area there:

| Quadrant | OS buildings (m²) | LiDAR + S2 | Microsoft | OSM | OS Open |
|---|---|---|---|---|---|
| SW | 3,692,057 | 157,088 | 167,239 | 86,171 | 41,312 |
| SE | 1,676,458 | 82,717 | 86,611 | 50,992 | 25,022 |
| NE | 321,791 | 19,841 | 14,767 | 6,296 | 4,659 |
| NW | 274,237 | 20,049 | 18,388 | 8,920 | 4,508 |

Most area outside OS outlines is edge misalignment for every dataset. LiDAR's
isolated false area is on a par with Microsoft's, but its whole false shapes
cover twice the area: they are larger, typically tree crowns and hedgerow
fragments in rural areas. The Sentinel-2 filter halved the number of false
shapes (from 1,595).

## Repository layout

```
scripts/
  config.py                 paths, CRS, analysis extent, ground truth filters
  groundtruth.py            ground truth loading and ignore zones
  01_profile_vectors.py     profile the AOI, ground truth and benchmark datasets
  02_prepare_rasters.py     mosaic DSM/DTM tiles onto the AOI grid, derive the nDSM
  03_extract_footprints.py  LiDAR footprint extraction
  04_evaluate.py            area / object / per-building evaluation of all datasets
  05_diagnose_objects.py    per-footprint features and false-detection analysis
  06_false_positives.py     footprint area outside OS buildings, edge vs isolated
  07_sentinel2_ndvi.py      summer 2022 Sentinel-2 NDVI composite
  tune_extraction.py        parallel parameter grid search
environment.yml             conda environment (conda-forge)
```

## Running it

```bash
conda env create -f environment.yml
conda activate geo
cd scripts
python 01_profile_vectors.py
python 02_prepare_rasters.py
python 07_sentinel2_ndvi.py      # downloads Sentinel-2 over the AOI; optional
python 03_extract_footprints.py  # skips the NDVI filter if 07 hasn't run
python 04_evaluate.py
python 06_false_positives.py
```

Outputs are written to `outputs/`: footprints, evaluation CSVs, a per-building
coverage GeoPackage and the false footprints, for mapping in QGIS.

> **Windows note:** if PostgreSQL/PostGIS sets `PROJ_LIB` and `GDAL_DATA`
> system-wide, point them at the environment's own copies when it activates:
> `conda env config vars set -n geo PROJ_LIB=<env>\Library\share\proj PROJ_DATA=<env>\Library\share\proj GDAL_DATA=<env>\Library\share\gdal`

## Data

The input data is not included in this repository. Expected layout under `data/`:

| Path | Source | Licence |
|---|---|---|
| `aoi/aoi.shp` | Test area polygon | — |
| `height/dsm/*.tif`, `height/dtm/*.tif` | [EA National LiDAR Programme](https://environment.data.gov.uk/survey) 1 m composite DSM (first return) and DTM, 2022 | OGL v3 |
| `ground_truth_data/gtd_buildings.parquet` | OS National Geographic Database, building features | OS licence (not redistributable) |
| `footprints/microsoft_buildings.parquet` | [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | ODbL |
| `footprints/osm_buildings.parquet` | OpenStreetMap | ODbL |
| `footprints/os_buildings.parquet` | OS Open building data | OGL v3 |

Sentinel-2 L2A is downloaded by `07_sentinel2_ndvi.py` from
[Earth Search](https://earth-search.aws.element84.com/v1) (Copernicus data,
free and open). All data is in British National Grid (EPSG:27700).

The EA's **Vegetation Object Model** is deliberately *not* used as an input. Its
vegetation classification relies on proximity to OS MasterMap features, which
would leak the ground truth into the method.
