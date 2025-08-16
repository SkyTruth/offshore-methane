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
    Record discovered system:index values by appending to granules.csv and
    event_granule.csv. Idempotent: only adds missing granules and mappings.

    associations maps event_id → list of system_index strings.
    """
    if not associations:
        return

    # Load or initialize granules.csv
    granules: set[str] = set()
    granules_rows: list[dict] = []
    if cfg.GRANULES_CSV.is_file():
        with open(cfg.GRANULES_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                granules.add(row.get("system_index", ""))
                granules_rows.append(row)

    # Load or initialize event_granule.csv
    mappings: set[tuple[int, str]] = set()
    mapping_rows: list[dict] = []
    eids_with_any: set[int] = set()
    if cfg.EVENT_GRANULE_CSV.is_file():
        with open(cfg.EVENT_GRANULE_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    eid = int(row.get("event_id", -1))
                except (TypeError, ValueError):
                    continue
                eids_with_any.add(eid)
                sid = row.get("system_index", "")
                mappings.add((eid, sid))
                mapping_rows.append(row)

    # Append any missing granules and mappings (including 'no granules' markers)
    new_granules: list[dict] = []
    new_mappings: list[dict] = []
    for event_id, sids in associations.items():
        if not sids and event_id not in eids_with_any:
            # Mark that this event yielded no granules (empty system_index row)
            new_mappings.append({"event_id": str(event_id), "system_index": ""})
            eids_with_any.add(event_id)
        for item in sids:
            if isinstance(item, dict):
                sid = (item.get("system_index") or "").strip()
                # Values may be None; normalize to formatted strings or blanks later
                provided_q = {
                    "sunglint": item.get("sunglint"),
                    "cloudiness": item.get("cloudiness"),
                    "timestamp": item.get("timestamp"),
                }
            else:
                sid = (item or "").strip()
                provided_q = None

            if sid and sid not in granules:
                if provided_q is not None:
                    q = {
                        "sunglint": ""
                        if provided_q["sunglint"] is None
                        else f"{float(provided_q['sunglint']):.1f}",
                        "cloudiness": ""
                        if provided_q["cloudiness"] is None
                        else f"{float(provided_q['cloudiness']):.1f}",
                        "timestamp": provided_q["timestamp"] or "",
                    }
                else:
                    q = {"sunglint": "", "cloudiness": "", "timestamp": ""}
                new_granules.append({"system_index": sid, **q})
                granules.add(sid)

            if sid and (event_id, sid) not in mappings:
                new_mappings.append({"event_id": str(event_id), "system_index": sid})
                mappings.add((event_id, sid))

    # Write granules.csv (preserve existing rows + append new)
    if new_granules or not cfg.GRANULES_CSV.is_file():
        fieldnames = ["system_index", "sunglint", "cloudiness", "timestamp"]
        with open(cfg.GRANULES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # keep any existing rows with the same schema (normalize keys)
            for r in granules_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
            for r in new_granules:
                writer.writerow(r)

    # Write event_granule.csv (preserve existing + append new)
    if new_mappings or not cfg.EVENT_GRANULE_CSV.is_file():
        # Canonical two columns only: event_id, system_index
        fieldnames = ["event_id", "system_index"]
        with open(cfg.EVENT_GRANULE_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in mapping_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
            for r in new_mappings:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    if cfg.VERBOSE:
        print(
            f"↻ recorded {len(new_granules)} new granule(s) and {len(new_mappings)} mapping(s)"
        )


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
            s2, sga_img, centre_pt, cfg.MASK_PARAMS["local_sga_range"]
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
    if not cfg.EVENTS_CSV.is_file():
        return []
    with open(cfg.EVENTS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _load_mappings() -> tuple[dict[int, list[str]], set[int]]:
    """
    Return (event_id → [system_index, …], events_with_any_row) from event_granule.csv.
    events_with_any_row includes events that have a 'no-granule' marker row (empty system_index).
    """
    if not cfg.EVENT_GRANULE_CSV.is_file():
        return {}, set()
    out: dict[int, list[str]] = {}
    have_row: set[int] = set()
    with open(cfg.EVENT_GRANULE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                eid = int(row.get("event_id", -1))
            except (TypeError, ValueError):
                continue
            have_row.add(eid)
            sid = (row.get("system_index") or "").strip()
            # Only non-empty SIDs populate the mapping list
            if sid:
                out.setdefault(eid, []).append(sid)
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

    mappings, have_row = _load_mappings()

    # Build the candidate list of events
    allow: set[int] | None = None
    if cfg.EVENTS_TO_PROCESS is not None:
        allow = {int(x) for x in cfg.EVENTS_TO_PROCESS}

    pending = []
    for i, row in enumerate(rows):
        eid = int(row.get("id", i))
        if allow is not None and (i not in allow and eid not in allow):
            continue
        if eid in have_row:
            continue  # already recorded (has granules or marked as none)
        pending.append((eid, row))

    if not pending:
        if cfg.VERBOSE:
            print("✓ No new events to discover")
        return

    # Discover granules and record
    for eid, row in pending:
        centre_pt = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        products = sentinel2_system_indexes(
            centre_pt,
            str(row.get("start", cfg.START)),
            add_days_to_date(str(row.get("end", cfg.END)), 1),
        )
        # Always record the event: empty list marks 'no granules found'
        if cfg.VERBOSE:
            print(f"event {eid}: {len(products)} product(s)")
        if products:
            # Record each granule immediately to guard against mid-run errors
            for item in products:
                _record_event_granules({eid: [item]})
        else:
            # Record 'no granules' marker for this event now
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

    # Index events for quick lookup when building site dicts
    by_id = {int(r.get("id", i)): r for i, r in enumerate(rows)}

    # Load mappings and quality metadata
    mappings, _ = _load_mappings()  # event_id → [sids], ignore have_row here
    gran_meta: dict[str, dict] = {}
    if cfg.GRANULES_CSV.is_file():
        with open(cfg.GRANULES_CSV, newline="") as f:
            for r in csv.DictReader(f):
                gran_meta[r.get("system_index", "")] = r

    allow: set[int] | None = None
    if cfg.EVENTS_TO_PROCESS is not None:
        allow = {int(x) for x in cfg.EVENTS_TO_PROCESS}

    # Thresholds (scene-level)
    max_cloud = float(cfg.MASK_PARAMS["cloud"]["scene_cloud_pct"])  # %
    sga_min, sga_max = cfg.MASK_PARAMS["sunglint"]["scene_sga_range"]

    start_time = time.time()
    active: list[ee.batch.Task] = []
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        futures = []
        for eid, sids in mappings.items():
            if allow is not None and eid not in allow:
                continue
            ev = by_id.get(eid)
            if not ev:
                continue
            # Build site dict from event row
            site = {
                "event_id": eid,
                "lon": float(ev["lon"]),
                "lat": float(ev["lat"]),
                "start": ev.get("start", cfg.START),
                "end": add_days_to_date(ev.get("end", cfg.END), 1),
            }

            for sid in sids:
                meta = gran_meta.get(sid, {})
                try:
                    cloud = float(meta.get("cloudiness", ""))
                except ValueError:
                    cloud = None
                try:
                    sga = float(meta.get("sunglint", ""))
                except ValueError:
                    sga = None

                # Filter if we have metadata; if missing, allow through
                if cloud is not None and cloud > max_cloud:
                    continue
                if sga is not None and not (sga_min < sga < sga_max):
                    continue

                futures.append(pool.submit(process_product, site, sid))

        for fut in as_completed(futures):
            active.extend(fut.result())

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
