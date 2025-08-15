# config.py
from pathlib import Path

# ------------------------------------------------------------------
#  Scene / AOI parameters
# ------------------------------------------------------------------
SITES_CSV = Path("../data/sites.csv")
CENTRE_LON, CENTRE_LAT = -90.96802087968751, 27.29220815000002  # PROTOTYPICAL
# SITES_TO_PROCESS = range(0, 47)  # 1 header and 46 S2 images
# SITES_TO_PROCESS = [42]  # Canonical site
# SITES_TO_PROCESS = [29]  # ethan's example

START, END = (
    "2017-07-05",  # Inclusive
    "2017-07-05",  # Inclusive
)  # Known pollution event  # PROTOTYPICAL


# ------------------------------------------------------------------
#  Algorithm switches / constants
# ------------------------------------------------------------------
# 0 ⇒ no speckle filtering, 1  ⇒  3 × 3 median window (≈ 20 m), 2 ⇒ 5 × 5
SPECKLE_RADIUS_PX = 2  # size of the square window
SPECKLE_FILTER_MODE = "median"  # "none" | "median" | "adaptive"
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
    "preferred_location": "local",  # "local", "bucket", "ee_asset_folder"
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
    "ndwi": {"threshold": 0.1},
    "sunglint": {
        "scene_sga_range": (0.0, 40.0),  # deg
        "local_sga_range": (0.0, 30.0),  # deg
        "local_sgi_range": (
            -0.60,
            1.0,
        ),  # NDI # -.6 needs to be higher to actually do anything
        "outlier_std_range": (0, 3),  # SGX Outlier
    },
    "min_valid_pct": 0.01,
}
