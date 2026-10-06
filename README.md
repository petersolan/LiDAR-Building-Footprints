# Building footprints from LiDAR: Exeter case study

Can open LiDAR elevation data alone produce building footprints that compete with
the freely available footprint datasets? This project extracts footprints for a
100 km² test area over Exeter, UK (OS grid square **SX99**) from the Environment
Agency's 1 m DSM and DTM. It scores them against Ordnance Survey's detailed
building data alongside three free alternatives: **Microsoft Global ML Building
Footprints**, **OpenStreetMap** and **OS Open** buildings.

## Idea

A building is something that stands above the ground and has a smooth, planar
surface. Trees also stand above the ground, but their canopy is rough. The
method uses only those two properties:

1. **Height above ground (nDSM):** DSM minus DTM. Keep pixels more than 2.5 m tall.
2. **Surface roughness:** the mean absolute Laplacian of the DSM over a 5 × 5 m
   window. Planar roofs score low and tree canopy scores high. Smooth tall
   pixels become building *seeds*.
3. **Grow and tidy:** seeds grow back a few metres into the tall area, which
   recovers roof edges and walls (the height step makes those rough). Small
   holes are filled and tiny blobs dropped.
4. **Object filter:** each footprint must reach 3.5 m at its highest point,
   matching the ground truth filter below.
5. **Vectorise** and simplify the footprints.

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

| Dataset | Area precision | Area recall | **Area F1** | IoU | Object F1 |
|---|---|---|---|---|---|
| **LiDAR (this project)** | 0.808 | 0.807 | **0.808** | 0.678 | 0.480 |
| Microsoft | 0.758 | 0.800 | 0.779 | 0.638 | 0.659 |
| OpenStreetMap | 0.819 | 0.800 | 0.809 | 0.680 | 0.663 |
| OS Open | 0.887 | 0.924 | 0.905 | 0.827 | 0.756 |

On area, the LiDAR-only footprints match OSM and beat Microsoft. They hold up on
the held-out quadrants: F1 0.806 on SE (urban) and 0.711 on NW (rural). OS Open
probably derives from the same OS master data as the ground truth, so its lead
is expected.

Known weaknesses: neighbouring buildings merge into single shapes (low object
F1), detached houses surrounded by trees are missed, and so are buildings under
20 m².

### False detections

This compares each dataset's footprint area outside *all* OS buildings,
including the ignored ones. That area splits into edge effects (within 2 m of
an OS building) and isolated detections (more than 2 m from any OS building):

| Dataset | Outside OS buildings | Isolated | Isolated, rural quadrants | Whole false shapes |
|---|---|---|---|---|
| **LiDAR** | 17.6% | 5.6% | 11.5–13.6% | 1,595 (10.3%) |
| Microsoft | 21.9% | 4.9% | 4.8–6.6% | 801 (3.1%) |
| OSM | 16.7% | 2.9% | 2.4–5.8% | 408 (1.0%) |
| OS Open | 10.0% | 1.3% | 1.5–1.6% | 19 (0.1%) |

Most area outside OS outlines is edge misalignment for every dataset. In urban
areas LiDAR's isolated false area matches Microsoft's. In rural areas it is 2–3
times higher, mostly small tree crowns and hedgerow fragments.

## Repository layout

```
scripts/
  config.py               paths, CRS, analysis extent, ground truth filters
  groundtruth.py          ground truth loading and ignore zones
  01_profile_vectors.py   profile the AOI, ground truth and benchmark datasets
  02_prepare_rasters.py   mosaic DSM/DTM tiles onto the AOI grid, derive the nDSM
  03_extract_footprints.py  LiDAR footprint extraction
  04_evaluate.py          area / object / per-building evaluation of all datasets
  05_diagnose_objects.py  per-footprint features and false-detection analysis
  06_false_positives.py   footprint area outside OS buildings, edge vs isolated
  tune_extraction.py      parallel parameter grid search
environment.yml           conda environment (conda-forge)
```

## Running it

```bash
conda env create -f environment.yml
conda activate geo
cd scripts
python 01_profile_vectors.py
python 02_prepare_rasters.py
python 03_extract_footprints.py
python 04_evaluate.py
```

Outputs are written to `outputs/` (footprints, evaluation CSVs and a per-building
coverage GeoPackage for mapping misses in QGIS).

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

All data is in British National Grid (EPSG:27700).

The EA's **Vegetation Object Model** is deliberately *not* used as an input. Its
vegetation classification relies on proximity to OS MasterMap features, which
would leak the ground truth into the method.
