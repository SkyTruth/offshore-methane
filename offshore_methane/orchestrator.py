# %%
# orchestrator.py
#!/usr/bin/env python3
"""
Centralised run-time configuration & immutable constants.
Refactored for high-throughput, thread-parallel processing.
Now also records a per-product audit trail in ../data/orch_history.csv
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

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
MAX_WIND_10M = 9  # m s-1 upper limit for ERA5 10 m wind

SCENE_SGA_RANGE = (0.0, 25)  # deg
LOCAL_SGA_RANGE = (0.0, 20)  # deg

# ------------------------------------------------------------------
#  Algorithm switches / constants
# ------------------------------------------------------------------
SPECKLE_RADIUS_PX = 10  # size of the square window
SPECKLE_FILTER_MODE = "adaptive"  # "none" | "median" | "adaptive"
LOGISTIC_SIGMA0 = 0.02
LOGISTIC_K = 300

USE_SIMPLE_MBSP = True
PLUME_P1, PLUME_P2, PLUME_P3 = -0.02, -0.04, -0.08

SHOW_THUMB = True
MAX_WORKERS = 32
EXPORT_PARAMS = {
    "bucket": "offshore_methane",
    "ee_asset_folder": "projects/cerulean-338116/assets/offshore_methane",
    "preferred_location": "bucket",
    "overwrite": True,
}

# ------------------------------------------------------------------
#  History log
# ------------------------------------------------------------------
HIST_CSV = Path("../data/orch_history.csv")
HIST_LOCK = Lock()  # protects concurrent writers
HIST_RUN_TS = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

FIELDNAMES = [
    "run_ts",
    "site_lon",
    "site_lat",
    "site_start",
    "site_end",
    "sentinel_id",
    "product_ok",
    "sga_src",
    "sga_new",
    "speckle_mode",
    "speckle_radius_px",
    "logistic_sigma0",
    "logistic_k",
    "use_simple_mbsp",
    "plume_p1",
    "plume_p2",
    "plume_p3",
    "raster_task_id",
    "raster_task_state",
    "vector_task_id",
    "vector_task_state",
    "rast_new",
    "overwrite",
    "thumb_url",
    "elapsed_ms",
]


def _log_history(row: dict[str, str | int | float | bool]) -> None:
    """Append *one* row to orch_history.csv in a thread-safe way."""
    with HIST_LOCK:
        HIST_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not HIST_CSV.exists()
        with HIST_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, FIELDNAMES, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)


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
    Also logs a history row to orch_history.csv.
    """
    t0 = time.time()
    tasks: list[ee.batch.Task] = []

    centre_pt = ee.Geometry.Point([site["lon"], site["lat"]])
    export_roi = centre_pt.buffer(AOI_RADIUS_M)

    # ---------- load S2 image ---------------
    print(f"▶ {sid}")
    s2 = ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")
    if s2 is None:
        print("  ⚠ EE image not found")
        _log_history(
            {
                **_base_fields(site, sid),
                "product_ok": False,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
        )
        return tasks

    # ---------- SGA asset ------------------
    sga_src, sga_new = ensure_sga_asset(sid, **EXPORT_PARAMS)
    sga_img = (
        ee.Image.loadGeoTIFF(sga_src)
        if isinstance(sga_src, str) and sga_src.startswith("gs://")
        else ee.Image(sga_src)
    )

    # ---------- local cloud/glint ----------
    product_pass = product_ok(
        s2,
        sga_img,
        centre_pt,
        AOI_RADIUS_M,
        LOCAL_MAX_CLOUD,
        LOCAL_SGA_RANGE,
        MAX_WIND_10M,
    ).getInfo()

    if not product_pass:
        print("  ✗ local cloud/glint rejected")
        _log_history(
            {
                **_base_fields(site, sid, sga_src, sga_new),
                "product_ok": False,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
        )
        return tasks

    # ---------- MBSP computation -----------
    if USE_SIMPLE_MBSP:
        R_img = mbsp_simple_ee(s2, centre_pt)
    else:
        R_img = mbsp_complex_ee(s2, sga_img, centre_pt, LOCAL_SGA_RANGE)

    # ---------- Speckle filter -------------
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

    # ---------- Thumbnail ------------------
    thumb_url = ""
    if SHOW_THUMB:
        thumb_url = R_img.visualize(
            min=-0.1, max=0.1, palette=["red", "white", "blue"]
        ).getThumbURL({"region": export_roi, "dimensions": 512})
        print(f"  🖼  thumb → {thumb_url}")

    # ---------- Export raster --------------
    EXPORT_PARAMS["overwrite"] = EXPORT_PARAMS["overwrite"] or sga_new
    rast_task, rast_new = export_image(R_img, sid, export_roi, **EXPORT_PARAMS)
    if rast_task:
        tasks.append(rast_task)
        print(f"  ⧗ raster task {rast_task.id} started")

    # ---------- Plume polygons -------------
    vect_fc = plume_polygons_three_p(
        R_img, export_roi, PLUME_P1, PLUME_P2, PLUME_P3
    ).filterBounds(centre_pt.buffer(LOCAL_PLUME_DIST_M))

    # ---------- Export vectors -------------
    EXPORT_PARAMS["overwrite"] = EXPORT_PARAMS["overwrite"] or sga_new or rast_new
    suffix = f"{site['lon']:.2f}_{site['lat']:.2f}"
    vect_task, _ = export_polygons(vect_fc, sid, suffix, **EXPORT_PARAMS)
    if vect_task:
        tasks.append(vect_task)
        print(f"  ⧗ vector task {vect_task.id} started")

    # ---------- Log history ----------------
    _log_history(
        {
            **_base_fields(site, sid, sga_src, sga_new),
            "product_ok": True,
            "raster_task_id": getattr(rast_task, "id", ""),
            "raster_task_state": getattr(rast_task, "state", "READY")
            if rast_task
            else "",
            "vector_task_id": getattr(vect_task, "id", ""),
            "vector_task_state": getattr(vect_task, "state", "READY")
            if vect_task
            else "",
            "rast_new": rast_new,
            "overwrite": EXPORT_PARAMS["overwrite"],
            "thumb_url": thumb_url,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    )

    return tasks


def _base_fields(site: dict, sid: str, sga_src: str | None = "", sga_new: bool = False):
    """Helper that returns constant fields shared by every history row."""
    return dict(
        run_ts=HIST_RUN_TS,
        site_lon=site["lon"],
        site_lat=site["lat"],
        site_start=site["start"],
        site_end=site["end"],
        sentinel_id=sid,
        sga_src=sga_src,
        sga_new=sga_new,
        speckle_mode=SPECKLE_FILTER_MODE,
        speckle_radius_px=SPECKLE_RADIUS_PX,
        logistic_sigma0=LOGISTIC_SIGMA0 if SPECKLE_FILTER_MODE == "adaptive" else "",
        logistic_k=LOGISTIC_K if SPECKLE_FILTER_MODE == "adaptive" else "",
        use_simple_mbsp=USE_SIMPLE_MBSP,
        plume_p1=PLUME_P1,
        plume_p2=PLUME_P2,
        plume_p3=PLUME_P3,
    )


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
    print({t.id: t.status()["state"] for t in active})
    while any(t.status()["state"] in ("READY", "RUNNING") for t in active):
        time.sleep(1)
    print({t.id: t.status()["state"] for t in active})
    print("Done.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
# %%
