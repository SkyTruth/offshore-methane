# config.py
from pathlib import Path

# ------------------------------------------------------------------
#  Scene / AOI parameters
# ------------------------------------------------------------------
SITES_CSV = Path("../data/sites.csv")
SITES_TO_PROCESS = range(221, 225)

CENTRE_LON, CENTRE_LAT = -90.96802087968751, 27.29220815000002  # PROTOTYPICAL
START, END = (
    "2017-07-05",  # Inclusive
    "2017-07-05",  # Inclusive
)  # Known pollution event  # PROTOTYPICAL
# START, END = "2016-01-01", "2016-12-01"  # Known pollution event # PROTOTYPICAL
# CENTRE_LON, CENTRE_LAT = 101.30972804846851, 9.12181368179713  # VENTING?
# START, END = "2024-11-26", "2024-12-18"  # Known pollution event # VENTING?

# ------------------------------------------------------------------
#  Algorithm switches / constants
# ------------------------------------------------------------------
# 0 ⇒ no speckle filtering, 1  ⇒  3 × 3 median window (≈ 20 m), 2 ⇒ 5 × 5
SPECKLE_RADIUS_PX = 10  # size of the square window
SPECKLE_FILTER_MODE = "none"  # "none" | "median" | "adaptive"
# Logistic curve controls for adaptive speckle filtering
LOGISTIC_SIGMA0 = 0.02  # σ where w = 0.5   (units match image data)
LOGISTIC_K = 300  # slope at σ₀ (bigger ⇒ steeper transition)

USE_SIMPLE_MBSP = True
PLUME_P1, PLUME_P2, PLUME_P3 = -0.02, -0.04, -0.08

SHOW_THUMB = False  # QA only - keep False in bulk
MAX_WORKERS = 32  # parallel threads
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
        "cs_thresh": 0.65,  # cloudy above
        "prob_thresh": 65,  # BACKUP: cloudy below
    },
    "wind": {
        "max_wind_10m": 9,  # m s-1, ERA5 / CFSv2 upper limit
        "time_window": 3,
    },
    "outlier": {
        "bands": ["B11", "B12"],
        "p_low": 2,
        "p_high": 100,
        "saturation": 10_000,
    },
    "ndwi": {"threshold": 0.0},
    "sunglint": {
        "scene_sga_range": (0.0, 40.0),  # deg
        "local_sga_range": (0.0, 30.0),  # deg
        "local_sgi_range": (-0.30, 1.0),  # NDI
    },
    "min_valid_pct": 0.4,
}
