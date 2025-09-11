# Offshore Methane Pilot

This repository contains experimental tooling for detecting offshore methane emissions using Sentinel-2 imagery. The codebase grew out of SkyTruth's research efforts and includes utilities for pixel masking, MBSP raster generation and plume polygon extraction.

## Repository layout

| Path | Purpose |
| --- | --- |
| `offshore_methane/` | Core Python package. Modules are described below. |
| `notebooks/` | Example Jupyter notebooks for interactive exploration. |
| `data/` | Small example data such as `structures.csv`+`windows.csv` (inputs), plus `granules.csv` and `process_runs.csv` (outputs). |
| `docs/` | Additional documentation. |
| `tests/` | Unit tests run by `pytest`. |

### Package modules

- **`algos.py`** - local helpers for turning MBSP rasters into plume polygons (`plume_polygons_three_p`) and the `logistic_speckle` filter.
- **`cdse.py`** - convenience wrappers around the Copernicus Data Space API used to fetch Sentinel-2 metadata and products.
- **`config.py`** - runtime configuration including scene dates, masking parameters and export settings.
- **`ee_utils.py`** - thin wrappers around the Earth Engine Python API.  Notable functions include `quick_view` for visual inspection, `export_image`/`export_polygons` for batch exports and `sentinel2_system_indexes` for product searches.
- **`gcp_utils.py`** - utilities for interacting with Google Cloud (locating `gsutil`).
- **`masking.py`** - pixel-mask builders used to compute the C-factor and MBSP masks.  Exposes `build_mask_for_C`, `build_mask_for_MBSP` and an interactive `view_mask` utility.
- **`mbsp.py`** - implementations of the complex and simple MBSP algorithms.
- **`orchestrator.py`** - high-level pipeline that ties everything together: downloading SGA grids, building masks, running MBSP and exporting artefacts in parallel.
- **`sga.py`** - creation and staging of coarse sun-glint angle grids (SGA) either locally, in Cloud Storage or as EE assets.

`__init__.py` re-exports the most frequently used modules so they can be imported directly via `from offshore_methane import mbsp, orchestrator, …`.

## Quick start

### 1. Create the environment

```bash
mamba env create -f environment.yml
conda activate methane
pip install -e .
pre-commit install
```

### 2. Run the unit tests

```bash
pytest
```

### 3. Explore imagery

Use `quick_view` to display a Sentinel-2 scene by system index:

```python
from offshore_methane.ee_utils import quick_view
m = quick_view("20170705T164319_20170705T165225_T15RXL")
# In notebooks: display(m)
```

To inspect the masking logic interactively:

```python
from offshore_methane.masking import view_mask
m = view_mask(
    "20170705T164319_20170705T165225_T15RXL",
    -90.9680,
    27.2922,
    compute_stats=True,
)
```

### 4. Run the orchestrator

There are two phases you can run independently:

1) Discover granules (populate `granules.csv` and `process_runs.csv`):
```bash
python -m offshore_methane.orchestrator discover
```

If a window has no matching Sentinel‑2 granules, a marker row is added to
`process_runs.csv` for that window (with empty `system_index`), so it won’t be
re-discovered on subsequent runs.

2) Process granules (SGA grid, masks, MBSP, exports):
```bash
python -m offshore_methane.orchestrator process
```

You can also run both sequentially with:
```bash
python -m offshore_methane.orchestrator both
```

Exports can target local files, Google Cloud Storage or EE assets depending on `EXPORT_PARAMS`. Discovered granules are appended to `data/granules.csv` and linked to windows in `data/process_runs.csv`.
When `EXPORT_PARAMS.overwrite` is `True`, discovery re-evaluates windows even if mappings already exist.

Filters (structure ids, window ids, granule ids) can be passed programmatically:

```python
from offshore_methane.orchestrator import main
# Discover only for given structures
main("discover", structure_ids=["x1", "x7"]) 
# Process for specific windows or granules
main("process", window_ids=[101, 102])
main("process", system_indexes=["20170705T164319_20170705T165225_T15RXL"]) 
```

When running as a module, you can also set lists in `config.py`:
`STRUCTURES_TO_PROCESS`, `WINDOWS_TO_PROCESS`, `GRANULES_TO_PROCESS`.
The orchestrator auto‑reloads `config.py` at runtime, so edits take effect
without restarting your session.

## Configuration

`config.py` centralises all tunable parameters - date ranges, mask thresholds, export locations and algorithm switches. The table below summarises how each variable is used in the codebase and the impact of tweaking it.

### Scene and AOI

| Name | Used in | Effect |
| --- | --- | --- |
| `STRUCTURES_CSV`, `WINDOWS_CSV` | `csv_utils.load_events` | Primary inputs (normalized split). `events.csv` is legacy fallback. |
| `CENTRE_LON`, `CENTRE_LAT` | `orchestrator.iter_sites` | Fallback coordinates when no windows exist. |
| `START`, `END` | `orchestrator.iter_sites` | Default date window for Sentinel-2 search. |

### Algorithm options

| Name | Used in | Effect when changed |
| --- | --- | --- |
| `SPECKLE_FILTER_MODE` (`"none"`, `"median"`, `"adaptive"`) | `orchestrator.process_product` | Chooses the speckle-reduction strategy. |
| `SPECKLE_RADIUS_PX` | `orchestrator.process_product` | Kernel size for median or adaptive speckle filtering. |
| `LOGISTIC_SIGMA0`, `LOGISTIC_K` | `algos.logistic_speckle` | Shape the logistic weighting for adaptive filtering. Higher `LOGISTIC_K` sharpens the transition; `LOGISTIC_SIGMA0` shifts it. |
| `USE_SIMPLE_MBSP` | `orchestrator.process_product` | Toggle between the complex and simple MBSP implementations. |
| `PLUME_P1`, `PLUME_P2`, `PLUME_P3` | `algos.plume_polygons_three_p` | Monotonic confidence thresholds for plume polygon detection. |
| `SHOW_THUMB` | `orchestrator.process_product` | If true, displays a diagnostic MBSP thumbnail URL. |
| `MAX_WORKERS` | `orchestrator.main` | Number of parallel threads used for EE exports. |

### Export parameters

The `EXPORT_PARAMS` dictionary routes output either to local disk, a Cloud Storage bucket or an EE asset collection.

| Key | Used in | Meaning |
| --- | --- | --- |
| `bucket` | `ee_utils.export_image`/`export_polygons` | Destination GCS bucket. |
| `ee_asset_folder` | same | Base EE folder for exported assets. |
| `preferred_location` | `orchestrator._cleanup_sid_assets`, `ee_utils.*` | Selects `"local"`, `"bucket"` or `"ee_asset_folder"` as the export backend. |
| `overwrite` | same | If `False`, skip exports when a file/asset already exists. |

### Masking parameters

The nested `MASK_PARAMS` dictionary drives pixel masking in `masking.py` and is also consulted by `ee_utils.sentinel2_system_indexes` when searching for scenes.

| Key | Sub-keys | Purpose |
| --- | --- | --- |
| `dist` | `export_radius_m`, `local_radius_m`, `plume_radius_m` | Radii for the export ROI, local mask stats and plume polygon search. |
| `cloud` | `scene_cloud_pct`, `cs_thresh`, `prob_thresh` | Scene-level filter on `CLOUDY_PIXEL_PERCENTAGE` and per-pixel cloud/ shadow thresholds. |
| `wind` | `max_wind_10m`, `time_window` | Limits on wind speed and temporal window for re-analysis data. |
| `outlier` | `bands`, `p_low`, `p_high`, `saturation` | Controls percentile-based outlier masking and saturation cutoff. |
| `ndwi` | `threshold` | Water mask; higher thresholds retain only open water. |
| `sunglint` | `scene_sga_range`, `local_sga_range`, `local_sgi_range` | Sun-glint angle gates used when filtering scenes and building the MBSP mask. |
| `min_valid_pct` | — | Minimum fraction of clear pixels needed before export. |

Changing these values alters the pixel selection process; for instance increasing `cloud.cs_thresh` makes the cloud mask stricter, while enlarging `dist.export_radius_m` expands the export extent.

## Additional resources

- [docs/references.md](docs/references.md) - relevant papers and background material.
- `notebooks/` - exploratory notebooks demonstrating cosine lookups, sunglint correction and a full MBSP demo.

## Data Model (CSV)

- granules.csv (key: system_index)
  - Columns: system_index, sga_scene, cloudiness, timestamp, git_hash.
- process_runs.csv (many-to-many: window_id ↔ system_index)
  - Columns: window_id, system_index, git_hash, last_timestamp (UTC ISO), sga_local_median, sgi_median, valid_pixel_c, valid_pixel_mbsp, hitl_value.
- windows.csv (input)
  - Columns: id (window_id), structure_id, start, end, flare_lat, flare_lon, optional metadata (e.g., citation, EEZ).
- structures.csv (input)
  - Columns: structure_id, lon, lat, optional name, country.

Notes
- Local medians (sga_local_median, sgi_median) are per-run metrics and are stored in process_runs.csv, not granules.csv.
- For legacy projects that used events.csv and event_granule.csv, use the migration: `python -m offshore_methane.csv_migrate`.

CSV conventions
- Missing values are written as blank cells (not the literal strings "nan" or "None").
- Text fields (e.g., `system_index`, `git_hash`, `timestamp`, `structure_id`) use blanks for missing.
- Numeric fields (e.g., medians, valid_pixel_*) use blanks for missing.
- `process_runs.system_index` is blank to mark a window with “no granules found”.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. All contributions must pass linting with `ruff` and the test suite before submission.

## License

This project is released under the [MIT License](LICENSE).
