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
from offshore_methane.csv_utils import (
    load_event_granule as csv_load_event_granule,
)
from offshore_methane.csv_utils import (
    load_events as csv_load_events,
)
from offshore_methane.csv_utils import (
    update_local_metrics,
    upsert_event_granule,
    upsert_granule,
    virtual_db,
)
from offshore_methane.ee_utils import (
    export_image,
    export_polygons,
    sentinel2_system_indexes,
)
from offshore_methane.masking import build_mask_for_C, build_mask_for_MBSP
from offshore_methane.mbsp import mbsp_complex_ee, mbsp_simple_ee
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
        if cfg.VERBOSE:
            print(f"  ↻ removed Cloud-Storage objects under {prefix}")

    elif loc == "ee_asset_folder":
        folder = cfg.EXPORT_PARAMS["ee_asset_folder"]
        for suffix in ("MBSP", "VEC", "SGA", "xml"):
            asset_id = f"{folder}/{sid}_{suffix}"
            try:
                ee.data.deleteAsset(asset_id)
                if cfg.VERBOSE:
                    print(f"  ↻ deleted EE asset {asset_id}")
            except Exception as exc:
                if "Asset not found" not in str(exc):
                    if cfg.VERBOSE:
                        print(f"  ⚠ could not delete {asset_id}: {exc}")

    else:  # loc == "local"
        local_dir = Path("../data") / sid
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)
            if cfg.VERBOSE:
                print(f"  ↻ removed local directory {local_dir}")


def add_days_to_date(date_str: str, days: int, fmt: str = "%Y-%m-%d") -> str:
    """
    Converts a date string to a datetime object, adds specified days, and returns a string.
    """
    dt = datetime.strptime(date_str, fmt)
    new_dt = dt + timedelta(days=days)
    return new_dt.strftime(fmt)


def _record_event_granules(associations: dict[int, list]) -> None:
    """
    Record discovered granules and event↔granule mappings using csv_utils upserts.

    associations maps event_id → list where items are either system_index strings
    or dicts with keys: system_index, sga_scene, cloudiness, timestamp.
    When the list is empty, a single marker row is appended to event_granule.csv
    for that event with an empty system_index (if not already present).
    """
    if not associations:
        return

    def _mark_no_granules(eid: int) -> None:
        df = csv_load_event_granule()
        have = False
        if not df.empty:
            have = bool(df.loc[df["event_id"] == int(eid)].shape[0])
        if not have:
            # Append a marker row (event_id, "")
            if cfg.EVENT_GRANULE_CSV.is_file():
                with open(cfg.EVENT_GRANULE_CSV, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["event_id", "system_index"])
                    w.writerow({"event_id": int(eid), "system_index": ""})
            else:
                with open(cfg.EVENT_GRANULE_CSV, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["event_id", "system_index"])
                    w.writeheader()
                    w.writerow({"event_id": int(eid), "system_index": ""})

    new_g, new_m = 0, 0
    for event_id, items in associations.items():
        if not items:
            _mark_no_granules(event_id)
            continue
        for item in items:
            if isinstance(item, dict):
                sid = (item.get("system_index") or "").strip()
                # Upsert quality metadata if present
                if sid:
                    upsert_granule(item)
                    new_g += 1
            else:
                sid = (item or "").strip()

            if sid:
                upsert_event_granule(int(event_id), sid)
                new_m += 1

    if cfg.VERBOSE:
        print(f"↻ recorded {new_g} new/updated granule(s) and {new_m} mapping(s)")


def _update_granule_local_metrics(
    sid: str, sga_local_median: float | None, sgi_median: float | None
) -> None:
    # Delegate to csv_utils helper
    update_local_metrics(sid, sga_local_median=sga_local_median, sgi_median=sgi_median)


# ------------------------------------------------------------------
#  Site iterator
# ------------------------------------------------------------------
def iter_sites():
    """
    Iterate over rows in sites.csv (or fallback to single site).
    Yields dicts containing location, date bounds, optional system_index,
    and the row index for later CSV updates.
    """
    if cfg.EVENTS_CSV.is_file():
        with open(cfg.EVENTS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))

        # Build optional filter set (supports event id or row index)
        allow: set[int] | None = None
        if cfg.EVENTS_TO_PROCESS is not None:
            allow = {int(x) for x in cfg.EVENTS_TO_PROCESS}

        for i, row in enumerate(rows):
            eid = int(row.get("id", i))
            if allow is not None and (i not in allow and eid not in allow):
                continue
            yield {
                "event_id": eid,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "start": row.get("start", cfg.START),
                "end": add_days_to_date(row.get("end", cfg.END), 1),
            }
    else:
        print(f"⚠  {cfg.EVENTS_CSV} not found - processing single hard-coded event")
        yield dict(
            event_id=-1,
            lon=cfg.CENTRE_LON,
            lat=cfg.CENTRE_LAT,
            start=cfg.START,
            end=add_days_to_date(cfg.END, 1),
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

    if cfg.VERBOSE:
        print(f"▶ {sid}")

    # ----------- load S2 image -----------------
    s2 = ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")
    if s2 is None:
        if cfg.VERBOSE:
            print("  ⚠ EE image not found")
        return tasks

    # ----------- SGA asset ---------------------
    try:
        sga_src, sga_new = ensure_sga_asset(sid, **cfg.EXPORT_PARAMS)
    except Exception as e:
        if cfg.VERBOSE:
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
    sgi = s2.normalizedDifference(["B8A", "B_vis"])
    s2 = s2.addBands(sgi.rename("SGI"))

    # ----------- Generate masks ------------------
    mask_c, kept_c = build_mask_for_C(s2, centre_pt, cfg.MASK_PARAMS)
    if cfg.VERBOSE:
        print(f"  {kept_c.multiply(100).getInfo():.1f}% Clear Pixels for {sid}")
    if kept_c.getInfo() < cfg.MASK_PARAMS["min_valid_pct"]:
        if cfg.VERBOSE:
            print("  ⚠ C mask mostly empty - skipping export")
        # build_mask_for_C(s2, centre_pt, cfg.MASK_PARAMS, True)
        _cleanup_sid_assets(sid)
        return tasks

    mask_mbsp, kept_mbsp = build_mask_for_MBSP(s2, centre_pt, cfg.MASK_PARAMS)
    if kept_mbsp.getInfo() < cfg.MASK_PARAMS["min_valid_pct"]:
        if cfg.VERBOSE:
            print("  ⚠ MBSP mask mostly empty - skipping export")
        # build_mask_for_MBSP(s2, centre_pt, cfg.MASK_PARAMS, True) # Uncomment to debug
        _cleanup_sid_assets(sid)
        return tasks

    # ---------------- MBSP computation ---------
    if cfg.USE_SIMPLE_MBSP:
        R_img = mbsp_simple_ee(s2, mask_c, mask_mbsp)
    else:
        R_img = mbsp_complex_ee(
            s2, sga_img, centre_pt, cfg.MASK_PARAMS["sunglint"]["local_sga_range"]
        )

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
        print(f"  🖼  thumb → {url} {sid}")

    # ---------------- Local metrics (medians) --
    try:
        med_sga = (
            s2.select("SGA")
            .reduceRegion(ee.Reducer.median(), export_roi, scale=20, bestEffort=True)
            .get("SGA")
            .getInfo()
        )
    except Exception:
        med_sga = None
    try:
        med_sgi = (
            s2.select("SGI")
            .reduceRegion(ee.Reducer.median(), export_roi, scale=20, bestEffort=True)
            .get("SGI")
            .getInfo()
        )
    except Exception:
        med_sgi = None

    # Persist local metrics to granules.csv immediately
    try:
        _update_granule_local_metrics(
            sid,
            None if med_sga is None else float(med_sga),
            None if med_sgi is None else float(med_sgi),
        )
    except Exception:
        pass

    # ---------------- Export raster ------------
    cfg.EXPORT_PARAMS["overwrite"] = cfg.EXPORT_PARAMS["overwrite"] or sga_new
    try:
        rast_task, rast_new = export_image(R_img, sid, export_roi, **cfg.EXPORT_PARAMS)
    except Exception as exc:
        if cfg.VERBOSE:
            print(f"  ✗ raster export failed for {sid}: {exc}")
        _cleanup_sid_assets(sid)
        return tasks
    if rast_task:
        tasks.append(rast_task)
        if cfg.VERBOSE:
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
        if cfg.VERBOSE:
            print(f"  ⧗ vector task {vect_task.id} started")

    return tasks


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def _load_event_rows() -> list[dict]:
    df = csv_load_events()
    return df.to_dict(orient="records") if not df.empty else []


def _load_mappings() -> tuple[dict[int, list[str]], set[int]]:
    """Compatibility helper retained for now; prefer csv_utils.virtual_db.

    Returns (event_id → [system_index, …], events_with_any_row).
    """
    df = csv_load_event_granule()
    if df.empty:
        return {}, set()
    have_row = set(df["event_id"].dropna().astype(int).tolist())
    df = df[df["system_index"].astype(str).str.len() > 0]
    out: dict[int, list[str]] = {}
    if not df.empty:
        for eid, grp in df.groupby("event_id"):
            out[int(eid)] = [str(s) for s in grp["system_index"].tolist()]
    return out, have_row


def discover_granules_for_new_events() -> None:
    """
    Phase 1: For events that currently have no row in event_granule.csv,
    discover their Sentinel-2 granules and append to granules.csv + event_granule.csv.
    """
    rows = _load_event_rows()
    if not rows:
        print(f"⚠  {cfg.EVENTS_CSV} not found or empty - nothing to do")
        return

    # Events that already have any mapping row (including marker rows)
    _, have_row = _load_mappings()

    # Allow filter
    allow: set[int] | None = None
    if cfg.EVENTS_TO_PROCESS is not None:
        allow = {int(x) for x in cfg.EVENTS_TO_PROCESS}

    # Discover granules for events missing any mapping
    for i, row in enumerate(rows):
        eid = int(row.get("id", i))
        if allow is not None and (i not in allow and eid not in allow):
            continue
        if eid in have_row:
            continue

        centre_pt = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        products = sentinel2_system_indexes(
            centre_pt,
            str(row.get("start", cfg.START)),
            add_days_to_date(str(row.get("end", cfg.END)), 1),
        )
        if cfg.VERBOSE:
            print(f"event {eid}: {len(products)} product(s)")
        if products:
            _record_event_granules({eid: products})
        else:
            _record_event_granules({eid: []})


def process_event_granules() -> None:
    """
    Phase 2: Using EVENTS_TO_PROCESS, select relevant granules (from event_granule.csv
    joined with granules.csv), lightly filter by cloudiness and sunglint, then fully
    process each remaining granule (SGA grid, masks, MBSP, exports).
    """
    rows = _load_event_rows()
    if not rows:
        print(f"⚠  {cfg.EVENTS_CSV} not found or empty - nothing to do")
        return

    # Build the virtual database with default scene-level filters
    db = virtual_db()

    # Allow filter
    allow: set[int] | None = None
    if cfg.EVENTS_TO_PROCESS is not None:
        allow = {int(x) for x in cfg.EVENTS_TO_PROCESS}
        if not db.empty:
            db = db[db["event_id"].isin(allow)]

    start_time = time.time()
    active: list[ee.batch.Task] = []
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        futures = []
        if not db.empty:
            # Iterate unique (event_id, system_index) pairs
            for (eid, sid), grp in db.groupby(["event_id", "system_index"]):
                # Build site dict from the first row
                r = grp.iloc[0]
                site = {
                    "event_id": int(eid),
                    "lon": float(r["lon"]),
                    "lat": float(r["lat"]),
                    "start": str(r.get("start", cfg.START)),
                    "end": add_days_to_date(str(r.get("end", cfg.END)), 1),
                }
                futures.append(pool.submit(process_product, site, str(sid)))

        for fut in as_completed(futures):
            res = fut.result()
            if res:
                active.extend(res)

    # Monitor outstanding tasks
    if active:
        print("Waiting for EE exports …")
        print({t.id: t.status()["state"] for t in active})
        while any(t.status()["state"] in ("READY", "RUNNING") for t in active):
            time.sleep(1)
        print({t.id: t.status()["state"] for t in active})
    print(f"Done in {time.time() - start_time:.2f} seconds.")


def main(cmd: str = "both"):
    if cmd == "discover":
        discover_granules_for_new_events()
    elif cmd == "process":
        process_event_granules()
    elif cmd == "both":
        discover_granules_for_new_events()
        process_event_granules()
    else:
        print("Usage: python -m offshore_methane.orchestrator [discover|process|both]")


# %%
# ------------------------------------------------------------------
if __name__ == "__main__":
    main(cmd="discover")

# %%
