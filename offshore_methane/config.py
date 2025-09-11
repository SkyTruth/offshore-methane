# config.py
from pathlib import Path

# ------------------------------------------------------------------
#  Scene / AOI parameters
# ------------------------------------------------------------------
# Resolve repository root (offshore_methane/..)
_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"

# Standard CSV locations (derived from DATA_DIR). Callers should prefer
# csv_utils helpers instead of reading these directly.
GRANULES_CSV = DATA_DIR / "granules.csv"
PROCESS_RUNS_CSV = DATA_DIR / "process_runs.csv"
STRUCTURES_CSV = DATA_DIR / "structures.csv"
WINDOWS_CSV = DATA_DIR / "windows.csv"

# Optional subsets to process
# Prefer passing lists to orchestrator.main(...). These config lists are used
# when running as a module without arguments.
# - WINDOWS_TO_PROCESS: window ids
# - STRUCTURES_TO_PROCESS: structure ids (strings or ints)
# - GRANULES_TO_PROCESS: Sentinel-2 system:index values
#
# Leave as None to process all.
# WINDOWS_TO_PROCESS = None
# STRUCTURES_TO_PROCESS = None
# GRANULES_TO_PROCESS = None


# ------------------------------------------------------------------
#  Algorithm switches / constants
# ------------------------------------------------------------------
# 0 ⇒ no speckle filtering, 1  ⇒  3 × 3 median window (≈ 20 m), 2 ⇒ 5 × 5
SPECKLE_RADIUS_PX = 1  # size of the square window
SPECKLE_FILTER_MODE = "none"  # "none" | "median" | "adaptive"
# Logistic curve controls for adaptive speckle filtering
LOGISTIC_SIGMA0 = 0.02  # σ where w = 0.5   (units match image data)
LOGISTIC_K = 300  # slope at σ₀ (bigger ⇒ steeper transition)

USE_SIMPLE_MBSP = True
PLUME_P1, PLUME_P2, PLUME_P3 = -0.02, -0.04, -0.08

SHOW_THUMB = False  # QA only - keep False in bulk
VERBOSE = True
MAX_WORKERS = 32  # parallel threads
XML_SOURCE = "gcp"  # "cdse" | "gcp"
EXPORT_PARAMS = {
    "bucket": "offshore_methane",
    "ee_asset_folder": "projects/cerulean-338116/assets/offshore_methane",
    "preferred_location": "bucket",  # "local", "bucket", "ee_asset_folder"
    "overwrite": True,  # overwrite existing files
}

# ------------------------------------------------------------------
#  Masking parameters
# ------------------------------------------------------------------
MASK_PARAMS = {
    "dist": {
        "export_radius_m": 15_000,
        "local_radius_m": 5_000,
        "plume_radius_m": 1_000,
    },
    "cloud": {
        "scene_cloud_pct": 50,  # metadata
        "cs_thresh": 0.7,  # cloudy above
        "prob_thresh": 65,  # BACKUP: cloudy below
    },
    "wind": {
        "max_wind_10m": 9,  # m s-1, ERA5 / CFSv2 upper limit
        "time_window": 3,
    },
    "outlier": {
        "bands": ["B11", "B12"],
        "p_low": 2,
        "p_high": 100,
        "saturation": 10_000,
    },
    "ndwi": {"threshold": -10.0},
    "sunglint": {
        "scene_sga_range": (0.0, 40.0),  # deg
        "local_sga_range": (0.0, 30.0),  # deg
        "local_sgi_range": (
            -0.60,
            1.0,
        ),  # NDI # -.6 needs to be higher to actually do anything
        "outlier_std_range": (-10, 10),  # SGX Outlier
    },
    "min_valid_pct": 0.01,
}
