# Offshore Methane Pilot

This repository contains experimental tooling for detecting offshore methane emissions using Sentinel‑2 imagery. The codebase grew out of SkyTruth's research efforts and includes utilities for pixel masking, MBSP raster generation and plume polygon extraction.

## Repository layout

| Path | Purpose |
| --- | --- |
| `offshore_methane/` | Core Python package. Modules are described below. |
| `notebooks/` | Example Jupyter notebooks for interactive exploration. |
| `data/` | Small example data such as `sites.csv` listing known events. |
| `docs/` | Additional documentation. |
| `tests/` | Unit tests run by `pytest`. |

### Package modules

- **`algos.py`** – local helpers for turning MBSP rasters into plume polygons (`plume_polygons_three_p`) and the `logistic_speckle` filter.
- **`cdse.py`** – convenience wrappers around the Copernicus Data Space API used to fetch Sentinel‑2 metadata and products.
- **`config.py`** – runtime configuration including scene dates, masking parameters and export settings.
- **`ee_utils.py`** – thin wrappers around the Earth Engine Python API.  Notable functions include `quick_view` for visual inspection, `export_image`/`export_polygons` for batch exports and `sentinel2_system_indexes` for product searches.
- **`gcp_utils.py`** – utilities for interacting with Google Cloud (locating `gsutil`).
- **`masking.py`** – pixel‑mask builders used to compute the C‑factor and MBSP masks.  Exposes `build_mask_for_C`, `build_mask_for_MBSP` and an interactive `view_mask` utility.
- **`mbsp.py`** – implementations of the complex and simple MBSP algorithms.
- **`orchestrator.py`** – high‑level pipeline that ties everything together: downloading SGA grids, building masks, running MBSP and exporting artefacts in parallel.
- **`sga.py`** – creation and staging of coarse sun‑glint angle grids (SGA) either locally, in Cloud Storage or as EE assets.

`__init__.py` re‑exports the most frequently used modules so they can be imported directly via `from offshore_methane import mbsp, orchestrator, …`.

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

Use `quick_view` to display a Sentinel‑2 scene by system index:

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

The main processing pipeline reads sites from `data/sites.csv` and exports MBSP rasters and plume polygons. Adjust parameters in `offshore_methane/config.py` as needed and launch:

```bash
python -m offshore_methane.orchestrator
```

Exports can target local files, Google Cloud Storage or EE assets depending on `EXPORT_PARAMS` in the configuration.

## Configuration

`config.py` centralises all tunable parameters – date ranges, mask thresholds, export locations, MBSP constants and more. Editing this file is the typical way to customise runs. Example fields include `MASK_PARAMS`, `EXPORT_PARAMS`, and `SITES_CSV` which lists event locations.

## Additional resources

- [docs/references.md](docs/references.md) – relevant papers and background material.
- `notebooks/` – exploratory notebooks demonstrating cosine lookups, sunglint correction and a full MBSP demo.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. All contributions must pass linting with `ruff` and the test suite before submission.

## License

This project is released under the [MIT License](LICENSE).
