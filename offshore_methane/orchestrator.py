# %%
#!/usr/bin/env python3
"""
Centralised run-time configuration & immutable constants.
Edit here rather than sprinkling literals through the codebase.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import ee

from offshore_methane.algos import plume_polygons_three_p
from offshore_methane.ee_utils import (
    export_image,
    export_polygons,
    local_cloud_check,
    local_glint_check,
    scene_glint_check,
    sentinel2_product_ids,
)
from offshore_methane.mbsp import mbsp_complex_ee, mbsp_simple_ee
from offshore_methane.sga import ensure_sga_asset

# ------------------------------------------------------------------
#  Scene / AOI parameters
# ------------------------------------------------------------------
CENTRE_LON, CENTRE_LAT = -90.96802087968751, 27.29220815000002
START, END = "2017-07-04", "2017-07-06"
AOI_RADIUS_M = 5_000
LOCAL_PLUME_DIST_M = 500

SCENE_MAX_CLOUD = 20  # % metadata CLOUDY_PIXEL_PERCENTAGE
LOCAL_MAX_CLOUD = 5  # % QA60 inside AOI

SCENE_SGA_RANGE = (0.0, 25)  # deg
LOCAL_SGA_RANGE = (0.0, 20)  # deg

# ------------------------------------------------------------------
#  Algorithm switches / constants
# ------------------------------------------------------------------
# 0 ⇒ no speckle filtering, 1  ⇒  3 × 3 median window (≈ 20 m), 2 ⇒ 5 × 5
SPECKLE_RADIUS_PX = 3
USE_SIMPLE_MBSP = True
PLUME_P1, PLUME_P2, PLUME_P3 = -0.02, -0.04, -0.08

SHOW_THUMB = True
EXPORT_PARAMS = {  # uncomment ZERO or ONE key
    "bucket": "offshore_methane",
    "ee_asset_folder": "projects/cerulean-338116/assets/offshore_methane",
    "max_concurrent_tasks": 10,
    "preferred_location": None,
    # "preferred_location": "bucket",
    # "preferred_location": "ee_asset_folder",
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
        print(f"⚠  {SITES_CSV} not found – processing single hard-coded site")
        yield dict(lon=CENTRE_LON, lat=CENTRE_LAT, start=START, end=END)


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    active = []

    for site in iter_sites():
        centre_pt = ee.Geometry.Point([site["lon"], site["lat"]])
        products = sentinel2_product_ids(
            centre_pt, site["start"], site["end"], cloud_pct=20
        )
        if not products:
            print(f"No products for {site}")
            continue
        print(f"{len(products)} product(s) for {site}")

        for pid in products:
            print(f"▶ {pid}")
            s2 = ee.Image(
                ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filter(ee.Filter.eq("PRODUCT_ID", pid))
                .first()
            )
            if s2 is None:
                print("  ⚠ EE image not found")
                continue

            if not scene_glint_check(s2, SCENE_SGA_RANGE):
                print("  ✗ scene SGA out of range")
                continue
            if not local_cloud_check(s2, centre_pt, AOI_RADIUS_M, LOCAL_MAX_CLOUD):
                print("  ✗ too cloudy inside AOI")
                continue

            # Obtain the source (asset ID *or* gs:// URL) …
            sga_src = ensure_sga_asset(pid, **EXPORT_PARAMS)
            sga_img = (
                ee.Image.loadGeoTIFF(sga_src)
                if isinstance(sga_src, str) and sga_src.startswith("gs://")
                else ee.Image(sga_src)
            )

            if not local_glint_check(sga_img, centre_pt, AOI_RADIUS_M, LOCAL_SGA_RANGE):
                print("  ✗ local SGA out of range")
                continue

            # ---------------- MBSP computation -----------------
            if USE_SIMPLE_MBSP:
                R_img = mbsp_simple_ee(s2, centre_pt)
                mode_tag = "MBSPs"
            else:
                R_img = mbsp_complex_ee(s2, sga_img, centre_pt, LOCAL_SGA_RANGE)
                mode_tag = "MBSPc"

            export_roi = centre_pt.buffer(AOI_RADIUS_M)
            if SPECKLE_RADIUS_PX > 0:
                R_img = (
                    ee.Image(R_img)
                    .clip(export_roi)
                    .toFloat()
                    .focal_median(
                        radius=SPECKLE_RADIUS_PX, units="pixels", kernelType="square"
                    )
                )
            else:
                R_img = ee.Image(R_img).clip(export_roi).toFloat()

            if SHOW_THUMB:
                url = R_img.visualize(
                    min=-0.1, max=0.1, palette=["red", "white", "blue"]
                ).getThumbURL({"region": export_roi, "dimensions": 512})
                print(f"  🖼  thumb → {url}")

            desc = f"{mode_tag}_{pid}"
            rast_task = export_image(R_img, desc, export_roi, **EXPORT_PARAMS)
            if rast_task:
                active.append(rast_task)
                print(f"  ⧗ raster task {rast_task.id} started")

            vect_fc = plume_polygons_three_p(
                R_img, export_roi, PLUME_P1, PLUME_P2, PLUME_P3
            ).filterBounds(centre_pt.buffer(LOCAL_PLUME_DIST_M))
            vect_task = export_polygons(vect_fc, desc, **EXPORT_PARAMS)
            if vect_task:
                active.append(vect_task)
                print(f"  ⧗ vector task {vect_task.id} started")

            # throttle EE queue
            while (
                sum(t.status()["state"] in ("READY", "RUNNING") for t in active)
                >= EXPORT_PARAMS["max_concurrent_tasks"]
            ):
                time.sleep(1)

    # ------------- monitor outstanding tasks -----------------
    print("Waiting for EE exports …")
    while any(t.status()["state"] in ("READY", "RUNNING") for t in active):
        print({t.id: t.status()["state"] for t in active})
        time.sleep(10)
    print("Done.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()

# %%
