# %%
# orchestrator.py
#!/usr/bin/env python3
"""
Centralised run-time configuration & immutable constants.
Refactored for high-throughput, thread-parallel processing.
"""

from __future__ import annotations

import csv
import shutil  # needed for local clean-ups
import subprocess  # needed for gsutil clean-ups
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import ee

import offshore_methane.config as cfg
from offshore_methane.algos import logistic_speckle, plume_polygons_three_p
from offshore_methane.ee_utils import (
    export_image,
    export_polygons,
    sentinel2_system_indexes,
)
from offshore_methane.masking import build_mask_for_C, build_mask_for_MBSP
from offshore_methane.mbsp import mbsp_complex, mbsp_sgx, mbsp_simple
from offshore_methane.sga import ensure_sga_asset


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
def _cleanup_sid_assets(sid: str) -> None:
    """
    Delete any previously-exported artefacts (raster, vectors, SGA, XML …)
    that share the Sentinel-2 system:index *sid*.

    The concrete deletion strategy follows the user-selected export
    backend in EXPORT_PARAMS["preferred_location"].
    """
    loc = cfg.EXPORT_PARAMS["preferred_location"]

    if loc == "bucket":
        prefix = f"gs://{cfg.EXPORT_PARAMS['bucket']}/{sid}"
        subprocess.run(
            ["gsutil", "-m", "rm", "-r", prefix],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  ↻ removed Cloud-Storage objects under {prefix}")

    elif loc == "ee_asset_folder":
        folder = cfg.EXPORT_PARAMS["ee_asset_folder"]
        for suffix in ("MBSP", "VEC", "SGA", "xml"):
            asset_id = f"{folder}/{sid}_{suffix}"
            try:
                ee.data.deleteAsset(asset_id)
                print(f"  ↻ deleted EE asset {asset_id}")
            except Exception as exc:
                if "Asset not found" not in str(exc):
                    print(f"  ⚠ could not delete {asset_id}: {exc}")

    else:  # loc == "local"
        local_dir = Path("../data") / sid
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)
            print(f"  ↻ removed local directory {local_dir}")


def add_days_to_date(date_str: str, days: int, fmt: str = "%Y-%m-%d") -> str:
    """
    Converts a date string to a datetime object, adds specified days, and returns a string.
    """
    dt = datetime.strptime(date_str, fmt)
    new_dt = dt + timedelta(days=days)
    return new_dt.strftime(fmt)


def _update_sites_csv(updates: dict[int, str]) -> None:
    """
    Persist discovered system_index values back to sites.csv.

    parameters
    ----------
    updates : dict
        Maps row index → semicolon-delimited system_index string.
    """
    if not updates or not cfg.SITES_CSV.is_file():
        return

    with open(cfg.SITES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    # Ensure column exists
    fieldnames = list(rows[0].keys())
    if "system_index" not in fieldnames:
        fieldnames.append("system_index")

    for idx, value in updates.items():
        if 0 <= idx < len(rows):
            rows[idx]["system_index"] = value

    with open(cfg.SITES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"↻ updated {cfg.SITES_CSV} with system_index for {len(updates)} site(s)")


# ------------------------------------------------------------------
#  Site iterator
# ------------------------------------------------------------------
def iter_sites():
    """
    Iterate over rows in sites.csv (or fallback to single site).
    Yields dicts containing location, date bounds, optional system_index,
    and the row index for later CSV updates.
    """
    if cfg.SITES_TO_PROCESS and cfg.SITES_CSV.is_file():
        with open(cfg.SITES_CSV, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for i, row in enumerate(rows):
            if i not in cfg.SITES_TO_PROCESS:
                continue
            yield {
                "row_index": i,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "start": row.get("start", cfg.START),
                "end": add_days_to_date(row.get("end", cfg.END), 1),
                "system_index": row.get("system_index", "").strip(),
            }
    else:
        print(f"⚠  {cfg.SITES_CSV} not found - processing single hard-coded site")
        yield dict(
            row_index=-1,
            lon=cfg.CENTRE_LON,
            lat=cfg.CENTRE_LAT,
            start=cfg.START,
            end=add_days_to_date(cfg.END, 1),
            system_index="",
        )


# ------------------------------------------------------------------
#  Per-product work unit
# ------------------------------------------------------------------
def process_product(site: dict, sid: str) -> list[ee.batch.Task]:
    """
    Run the full MBSP + export pipeline for one Sentinel-2 product.
    Returns the list of Earth-Engine tasks it started.
    """
    tasks: list[ee.batch.Task] = []

    centre_pt = ee.Geometry.Point([site["lon"], site["lat"]])
    export_roi = centre_pt.buffer(cfg.MASK_PARAMS["dist"]["export_radius_m"])

    print(f"▶ {sid}")

    # ----------- load S2 image -----------------
    s2 = ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")
    if s2 is None:
        print("  ⚠ EE image not found")
        return tasks

    # ----------- SGA asset ---------------------
    try:
        sga_src, sga_new = ensure_sga_asset(sid, **cfg.EXPORT_PARAMS)
    except Exception as e:
        print(f"  ✗ error computing SGA grid for {sid}: {e}")
        _cleanup_sid_assets(sid)
        return tasks

    sga_img = (
        ee.Image.loadGeoTIFF(sga_src)
        if isinstance(sga_src, str) and sga_src.startswith("gs://")
        else ee.Image(sga_src)
    )

    # Add SGA, SGI, etc.
    s2 = s2.addBands(sga_img.rename("SGA"))
    b_vis = s2.select("B2").add(s2.select("B3")).add(s2.select("B4")).divide(3)
    s2 = s2.addBands(b_vis.rename("B_vis"))
    sgi = s2.normalizedDifference(["B12", "B_vis"])
    s2 = s2.addBands(sgi.rename("SGI"))

    # ----------- Generate masks ------------------
    mask_c, kept_c = build_mask_for_C(s2, centre_pt, cfg.MASK_PARAMS)
    print(f"  {kept_c.multiply(100).getInfo():.1f}% Clear Pixels for {sid}")
    if kept_c.getInfo() < cfg.MASK_PARAMS["min_valid_pct"]:
        print("  ⚠ C mask mostly empty - skipping export")
        # build_mask_for_C(s2, centre_pt, cfg.MASK_PARAMS, True)
        _cleanup_sid_assets(sid)
        return tasks

    mask_mbsp, kept_mbsp = build_mask_for_MBSP(s2, centre_pt, cfg.MASK_PARAMS)
    if kept_mbsp.getInfo() < cfg.MASK_PARAMS["min_valid_pct"]:
        print("  ⚠ MBSP mask mostly empty - skipping export")
        # build_mask_for_MBSP(s2, centre_pt, cfg.MASK_PARAMS, True) # Uncomment to debug
        _cleanup_sid_assets(sid)
        return tasks

    # ---------------- MBSP computation ---------
    if cfg.MBSP_ALG == "simple":
        R_img = mbsp_simple(s2, mask_c, mask_mbsp)
    elif cfg.MBSP_ALG == "sgx":
        R_img = mbsp_sgx(s2, mask_c, mask_mbsp)
    elif cfg.MBSP_ALG == "complex":
        R_img = mbsp_complex(s2, sga_img, centre_pt, cfg.MASK_PARAMS["local_sga_range"])

    # ---------------- Speckle filter -----------
    if cfg.SPECKLE_FILTER_MODE == "adaptive" and cfg.SPECKLE_RADIUS_PX > 0:
        R_img = logistic_speckle(
            R_img, cfg.SPECKLE_RADIUS_PX, cfg.LOGISTIC_SIGMA0, cfg.LOGISTIC_K
        )
    elif cfg.SPECKLE_FILTER_MODE == "median" and cfg.SPECKLE_RADIUS_PX > 0:
        R_img = (
            ee.Image(R_img)
            .clip(export_roi)
            .toFloat()
            .focal_median(
                radius=cfg.SPECKLE_RADIUS_PX, units="pixels", kernelType="square"
            )
        )
    else:
        R_img = ee.Image(R_img).clip(export_roi).toFloat()

    # ---------------- Thumbnail ----------------
    if cfg.SHOW_THUMB:
        url = R_img.visualize(
            min=-0.1, max=0.1, palette=["red", "white", "blue"]
        ).getThumbURL({"region": export_roi, "dimensions": 512})
        print(f"  🖼  thumb → {url}")

    # ---------------- Export raster ------------
    cfg.EXPORT_PARAMS["overwrite"] = cfg.EXPORT_PARAMS["overwrite"] or sga_new
    try:
        rast_task, rast_new = export_image(R_img, sid, export_roi, **cfg.EXPORT_PARAMS)
    except Exception as exc:
        print(f"  ✗ raster export failed for {sid}: {exc}")
        _cleanup_sid_assets(sid)
        return tasks
    if rast_task:
        tasks.append(rast_task)
        print(f"  ⧗ raster task {rast_task.id} started")

    # ---------------- Plume polygons -----------
    plume_roi = centre_pt.buffer(cfg.MASK_PARAMS["dist"]["plume_radius_m"])
    vect_fc = plume_polygons_three_p(
        R_img, plume_roi, cfg.PLUME_P1, cfg.PLUME_P2, cfg.PLUME_P3
    )

    # ---------------- Export vectors -----------
    cfg.EXPORT_PARAMS["overwrite"] = (
        cfg.EXPORT_PARAMS["overwrite"] or sga_new or rast_new
    )
    suffix = f"{site['lon']:.3f}_{site['lat']:.3f}"
    vect_task, _ = export_polygons(vect_fc, sid, suffix, **cfg.EXPORT_PARAMS)
    if vect_task:
        tasks.append(vect_task)
        print(f"  ⧗ vector task {vect_task.id} started")

    return tasks


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    start_time = time.time()
    active: list[ee.batch.Task] = []
    csv_updates: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        futures = []
        for site in iter_sites():
            # Decide where to get product IDs
            if site["system_index"]:
                products = [
                    s.strip() for s in site["system_index"].split(";") if s.strip()
                ]
            else:
                centre_pt = ee.Geometry.Point([site["lon"], site["lat"]])
                products = sentinel2_system_indexes(
                    centre_pt,
                    str(site["start"]),
                    str(site["end"]),
                )
                if products:
                    csv_updates[int(site["row_index"])] = ";".join(products)

            if not products:
                print(f"No products for {site}")
                continue
            print(f"{len(products)} product(s) for {site}")

            for sid in products:
                futures.append(pool.submit(process_product, site, sid))

        for fut in as_completed(futures):
            active.extend(fut.result())

    # Update CSV with any newly-discovered system_index values
    _update_sites_csv(csv_updates)

    # -------- monitor outstanding tasks --------
    print("Waiting for EE exports …")
    print({t.id: t.status()["state"] for t in active})
    while any(t.status()["state"] in ("READY", "RUNNING") for t in active):
        time.sleep(1)
    print({t.id: t.status()["state"] for t in active})
    print(f"Done in {time.time() - start_time:.2f} seconds.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
# %%
