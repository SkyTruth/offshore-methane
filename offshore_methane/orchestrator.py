# %%
# orchestrator.py
#!/usr/bin/env python3
"""
Centralised run-time configuration & immutable constants.
Refactored for high-throughput, thread-parallel processing.
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee

from offshore_methane.algos import logistic_speckle, plume_polygons_three_p
from offshore_methane.ee_utils import (
    export_image,
    export_polygons,
    product_ok,
    sentinel2_system_indexes,
)
from offshore_methane.mbsp import mbsp_complex_ee, mbsp_simple_ee
from offshore_methane.sga import ensure_sga_asset

# ------------------------------------------------------------------
#  Scene / AOI parameters
# ------------------------------------------------------------------
CENTRE_LON, CENTRE_LAT = -90.96802087968751, 27.29220815000002
START, END = "2017-07-04", "2017-07-06"  # Known pollution event
AOI_RADIUS_M = 5_000
LOCAL_PLUME_DIST_M = 500

SCENE_MAX_CLOUD = 20  # % metadata CLOUDY_PIXEL_PERCENTAGE
LOCAL_MAX_CLOUD = 5  # % QA60 inside AOI
MAX_WIND_10M = 9  # m s‑1 upper limit for ERA5 10 m wind

SCENE_SGA_RANGE = (0.0, 25)  # deg
LOCAL_SGA_RANGE = (0.0, 20)  # deg

# ------------------------------------------------------------------
#  Algorithm switches / constants
# ------------------------------------------------------------------
# 0 ⇒ no speckle filtering, 1  ⇒  3 × 3 median window (≈ 20 m), 2 ⇒ 5 × 5
SPECKLE_RADIUS_PX = 10  # size of the square window
SPECKLE_FILTER_MODE = "adaptive"  # "none" | "median" | "adaptive"
# Logistic curve controls for adaptive speckle filtering
LOGISTIC_SIGMA0 = 0.02  # σ where w = 0.5   (units match image data)
LOGISTIC_K = 300  # slope at σ₀ (bigger ⇒ steeper transition)

USE_SIMPLE_MBSP = True
PLUME_P1, PLUME_P2, PLUME_P3 = -0.02, -0.04, -0.08

SHOW_THUMB = True  # QA only - keep False in bulk
MAX_WORKERS = 32  # parallel threads
EXPORT_PARAMS = {
    "bucket": "offshore_methane",
    "ee_asset_folder": "projects/cerulean-338116/assets/offshore_methane",
    # "preferred_location": None, # Uncomment only one
    "preferred_location": "bucket",  # Uncomment only one
    # "preferred_location": "ee_asset_folder", # Uncomment only one
}

# ------------------------------------------------------------------
#  Other files
# ------------------------------------------------------------------
SITES_CSV = Path("../data/sites.csv")  # @Ben to help us generate and manage this file


# ------------------------------------------------------------------
#  Site iterator
# ------------------------------------------------------------------
def iter_sites():
    if SITES_CSV.is_file():
        with open(SITES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                yield {
                    "lon": float(row["lon"]),
                    "lat": float(row["lat"]),
                    "start": row.get("start", START),
                    "end": row.get("end", END),
                }
    else:
        print(f"⚠  {SITES_CSV} not found - processing single hard-coded site")
        yield dict(lon=CENTRE_LON, lat=CENTRE_LAT, start=START, end=END)


# ------------------------------------------------------------------
#  Per-product work unit
# ------------------------------------------------------------------
def process_product(site: dict, sid: str) -> list[ee.batch.Task]:
    """
    Entire original inner-loop wrapped in a function that can run in
    parallel threads.  Returns the list of EE tasks it started.

    NOTE: No need to call `ee.Initialize()` here - the global session
    established in `ee_utils` is thread-safe and already active.
    """
    tasks: list[ee.batch.Task] = []
    centre_pt = ee.Geometry.Point([site["lon"], site["lat"]])
    export_roi = centre_pt.buffer(AOI_RADIUS_M)

    print(f"▶ {sid}")

    # ----------- load S2 image -----------------
    s2 = ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")
    if s2 is None:
        print("  ⚠ EE image not found")
        return tasks

    # ----------- SGA asset ---------------------
    sga_src = ensure_sga_asset(sid, **EXPORT_PARAMS)

    sga_img = (
        ee.Image.loadGeoTIFF(sga_src)
        if isinstance(sga_src, str) and sga_src.startswith("gs://")
        else ee.Image(sga_src)
    )

    # ----------- local cloud/glint -------------------
    if not product_ok(
        s2,
        sga_img,
        centre_pt,
        AOI_RADIUS_M,
        LOCAL_MAX_CLOUD,
        LOCAL_SGA_RANGE,
        MAX_WIND_10M,
    ).getInfo():
        print("  ✗ local cloud/glint rejected")
        return tasks

    # ---------------- MBSP computation ---------
    if USE_SIMPLE_MBSP:
        R_img = mbsp_simple_ee(s2, centre_pt)
        mode_tag = "MBSPs"
    else:
        R_img = mbsp_complex_ee(s2, sga_img, centre_pt, LOCAL_SGA_RANGE)
        mode_tag = "MBSPc"

    # ---------------- Speckle filter -----------
    if SPECKLE_FILTER_MODE == "adaptive" and SPECKLE_RADIUS_PX > 0:
        R_img = logistic_speckle(R_img, SPECKLE_RADIUS_PX, LOGISTIC_SIGMA0, LOGISTIC_K)
    elif SPECKLE_FILTER_MODE == "median" and SPECKLE_RADIUS_PX > 0:
        R_img = (
            ee.Image(R_img)
            .clip(export_roi)
            .toFloat()
            .focal_median(radius=SPECKLE_RADIUS_PX, units="pixels", kernelType="square")
        )
    else:
        R_img = ee.Image(R_img).clip(export_roi).toFloat()

    # ---------------- Thumbnail ----------------
    if SHOW_THUMB:
        url = R_img.visualize(
            min=-0.1, max=0.1, palette=["red", "white", "blue"]
        ).getThumbURL({"region": export_roi, "dimensions": 512})
        print(f"  🖼  thumb → {url}")

    # ---------------- Export raster ------------
    desc = f"{mode_tag}_{sid}"
    rast_task = export_image(R_img, desc, export_roi, **EXPORT_PARAMS)
    if rast_task:
        tasks.append(rast_task)
        print(f"  ⧗ raster task {rast_task.id} started")

    # ---------------- Plume polygons -----------
    vect_fc = plume_polygons_three_p(
        R_img, export_roi, PLUME_P1, PLUME_P2, PLUME_P3
    ).filterBounds(centre_pt.buffer(LOCAL_PLUME_DIST_M))

    # ---------------- Export vectors -----------
    vect_task = export_polygons(vect_fc, desc, **EXPORT_PARAMS)
    if vect_task:
        tasks.append(vect_task)
        print(f"  ⧗ vector task {vect_task.id} started")

    return tasks


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    active: list[ee.batch.Task] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []
        for site in iter_sites():
            centre_pt = ee.Geometry.Point([site["lon"], site["lat"]])

            # -------- product search ----------
            products = sentinel2_system_indexes(
                centre_pt, site["start"], site["end"], cloud_pct=SCENE_MAX_CLOUD
            )

            if not products:
                print(f"No products for {site}")
                continue
            print(f"{len(products)} product(s) for {site}")

            for sid in products:
                futures.append(pool.submit(process_product, site, sid))

        for fut in as_completed(futures):
            active.extend(fut.result())

    # -------- monitor outstanding tasks --------
    print("Waiting for EE exports …")
    while any(t.status()["state"] in ("READY", "RUNNING") for t in active):
        print({t.id: t.status()["state"] for t in active})
        time.sleep(10)
    print("Done.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
# %%
