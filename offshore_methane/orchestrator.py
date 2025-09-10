# %%
# orchestrator.py
#!/usr/bin/env python3
"""
Centralised run-time configuration & immutable constants.
Refactored for high-throughput, thread-parallel processing.
"""

from __future__ import annotations

import shutil  # needed for local clean-ups
import subprocess  # needed for gsutil clean-ups
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import ee

import offshore_methane.config as cfg
from offshore_methane.algos import logistic_speckle, plume_polygons_three_p
from offshore_methane.csv_utils import (
    current_git_hash,
    unified_db,
    update_run_metrics,
    upsert_granule,
    virtual_db,
)
from offshore_methane.csv_utils import (
    load_events as csv_load_events,
)
from offshore_methane.csv_utils import load_process_runs as csv_load_process_runs
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
        local_dir = cfg.DATA_DIR / sid
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
    Record discovered granules and window↔granule mappings using csv_utils upserts.

    associations maps window_id → list where items are either system_index strings
    or dicts with keys: system_index, sga_scene, cloudiness, timestamp.
    When the list is empty, a single marker row is appended to process_runs.csv
    for that window with an empty system_index (if not already present).
    """
    if not associations:
        return

    def _mark_no_granules(eid: int) -> None:
        df = csv_load_process_runs()
        have = False
        if not df.empty:
            have = bool(df.loc[df["window_id"] == int(eid)].shape[0])
        if not have:
            # Append a marker row (window_id, "") into process_runs.csv
            from offshore_methane.csv_utils import upsert_process_run

            upsert_process_run(int(eid), "")

    new_g, new_m = 0, 0
    for event_id, items in associations.items():
        if not items:
            _mark_no_granules(event_id)
            continue
        gh = current_git_hash()
        from offshore_methane.csv_utils import upsert_process_run

        for item in items:
            if isinstance(item, dict):
                sid = (item.get("system_index") or "").strip()
                # Upsert quality metadata if present
                if sid:
                    payload = dict(item)
                    payload["git_hash"] = gh
                    upsert_granule(payload)
                    new_g += 1
            else:
                sid = (item or "").strip()

            if sid:
                upsert_process_run(int(event_id), sid, git_hash=gh)
                new_m += 1

    if cfg.VERBOSE:
        print(f"↻ recorded {new_g} new/updated granule(s) and {new_m} mapping(s)")


def _update_granule_local_metrics(
    window_id: int,
    sid: str,
    sga_local_median: float | None,
    sgi_median: float | None,
    *,
    valid_pixel_c: float | None = None,
    valid_pixel_mbsp: float | None = None,
    hitl_value: int | None = None,
) -> None:
    # Delegate to csv_utils helper
    try:
        gh = current_git_hash()
    except Exception:
        gh = None
    # New per-run metrics live in process_runs.csv
    update_run_metrics(
        int(window_id),
        sid,
        sga_local_median=None if sga_local_median is None else float(sga_local_median),
        sgi_median=None if sgi_median is None else float(sgi_median),
        valid_pixel_c=None if valid_pixel_c is None else float(valid_pixel_c),
        valid_pixel_mbsp=None if valid_pixel_mbsp is None else float(valid_pixel_mbsp),
        hitl_value=None if hitl_value is None else int(hitl_value),
        git_hash=gh,
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
    # Pick up any on-disk changes to config values
    try:
        from .utils import refresh_config as _refresh

        _refresh()
    except Exception:
        pass
    # Prefer csv_utils.load_events which supports structures/windows split
    rows_df = csv_load_events()
    if not rows_df.empty:
        rows = rows_df.to_dict(orient="records")

        for i, row in enumerate(rows):
            wid = int(row.get("id", i))
            yield {
                "window_id": wid,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "start": row.get("start"),
                "end": add_days_to_date(row.get("end"), 1),
            }
    else:
        raise Exception("⚠  No windows/structures found")


# ------------------------------------------------------------------
#  Per-product work unit
# ------------------------------------------------------------------
def process_product(site: dict, sid: str) -> list[ee.batch.Task]:
    """
    Run the full MBSP + export pipeline for one Sentinel-2 product.
    Returns the list of Earth-Engine tasks it started.
    """
    tasks: list[ee.batch.Task] = []
    gh = None
    try:
        gh = current_git_hash()
    except Exception:
        pass

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
    kept_c_val = None
    try:
        kept_c_val = float(kept_c.getInfo())
    except Exception:
        kept_c_val = None
    if cfg.VERBOSE and kept_c_val is not None:
        print(f"  {kept_c_val * 100:.1f}% Clear Pixels for {sid}")
    if kept_c_val is None or kept_c_val < cfg.MASK_PARAMS["min_valid_pct"]:
        if cfg.VERBOSE:
            print("  ⚠ C mask mostly empty - skipping export")
        # build_mask_for_C(s2, centre_pt, cfg.MASK_PARAMS, True)
        _cleanup_sid_assets(sid)
        return tasks

    mask_mbsp, kept_mbsp = build_mask_for_MBSP(s2, centre_pt, cfg.MASK_PARAMS)
    kept_mbsp_val = None
    try:
        kept_mbsp_val = float(kept_mbsp.getInfo())
    except Exception:
        kept_mbsp_val = None
    if kept_mbsp_val is None or kept_mbsp_val < cfg.MASK_PARAMS["min_valid_pct"]:
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

    # Persist local metrics to process_runs.csv immediately
    try:
        _update_granule_local_metrics(
            int(site.get("window_id", -1)),
            sid,
            None if med_sga is None else float(med_sga),
            None if med_sgi is None else float(med_sgi),
            valid_pixel_c=kept_c_val,
            valid_pixel_mbsp=kept_mbsp_val,
        )
    except Exception:
        pass

    # ---------------- Export raster ------------
    cfg.EXPORT_PARAMS["overwrite"] = cfg.EXPORT_PARAMS["overwrite"] or sga_new
    # Generate a run timestamp now so it can be embedded in shared artefacts
    # Use timezone-aware UTC timestamp
    from datetime import timezone as _tz
    ts = (
        datetime.now(_tz.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    try:
        rast_task, rast_new = export_image(
            R_img,
            sid,
            export_roi,
            last_timestamp=ts,
            suffix=f"{site['lon']:.3f}_{site['lat']:.3f}",
            **cfg.EXPORT_PARAMS,
        )
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
    # Use structure_id as suffix for saved artefacts
    suffix = str(site.get("structure_id", ""))
    vect_task, _ = export_polygons(
        vect_fc, sid, suffix, last_timestamp=ts, **cfg.EXPORT_PARAMS
    )
    if vect_task:
        tasks.append(vect_task)
        if cfg.VERBOSE:
            print(f"  ⧗ vector task {vect_task.id} started")

    # ---------------- Update process run metadata -----------
    try:
        from offshore_methane.csv_utils import upsert_process_run

        upsert_process_run(
            int(site.get("window_id", -1)), sid, git_hash=gh, last_timestamp=ts
        )
    except Exception:
        pass

    return tasks


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def _load_event_rows(
    *,
    window_ids: set[int] | None = None,
    structure_ids: set[str] | set[int] | None = None,
) -> list[dict]:
    """Load candidate window rows (id, lon, lat, start, end).

    When structure_ids are provided and normalized schema exists, restrict
    to those windows; otherwise, filter by window_ids when provided.
    """
    # Prefer the normalized schema for structure-based filtering
    from offshore_methane.csv_utils import load_structures, load_windows

    have_struct = load_structures()
    have_win = load_windows()

    if structure_ids is not None and not have_struct.empty and not have_win.empty:
        s = have_struct.copy()
        w = have_win.copy()
        s["structure_id"] = s["structure_id"].astype(str)
        # w_cols = {str(c).lower() for c in w.columns}
        if "id" not in w.columns and "window_id" in w.columns:
            w = w.rename(columns={"window_id": "id"})
        w["structure_id"] = w["structure_id"].astype(str)
        # Filter windows by requested structure ids
        sid_set = {str(x) for x in structure_ids}
        w = w[w["structure_id"].astype(str).isin(sid_set)].copy()
        # Join to get lon/lat
        j = w.merge(s[["structure_id", "lon", "lat"]], on="structure_id", how="left")
        # Normalize required columns
        rows: list[dict] = []
        for _, r in j.iterrows():
            try:
                rows.append(
                    {
                        "id": int(r.get("id")),
                        "lon": float(r.get("lon")),
                        "lat": float(r.get("lat")),
                        "start": str(r.get("start")),
                        "end": str(r.get("end")),
                    }
                )
            except Exception:
                continue
        # Optionally filter by window_ids intersection
        if window_ids is not None:
            rows = [rw for rw in rows if int(rw.get("id", -1)) in window_ids]
        return rows

    # Fallback to load_events view
    df = csv_load_events()
    if df.empty:
        return []
    if window_ids is not None:
        wid_set = {int(x) for x in window_ids}
        df = df[df["id"].astype(int).isin(wid_set)]
    return df.to_dict(orient="records")


def _load_mappings() -> tuple[dict[int, list[str]], set[int]]:
    """Compatibility helper retained for now; prefer csv_utils.virtual_db.

    Returns (window_id → [system_index, …], windows_with_any_row).
    """
    df = csv_load_process_runs()
    if df.empty:
        return {}, set()
    have_row = set(df["window_id"].dropna().astype(int).tolist())
    df = df[df["system_index"].astype(str).str.len() > 0]
    out: dict[int, list[str]] = {}
    if not df.empty:
        for eid, grp in df.groupby("window_id"):
            out[int(eid)] = [str(s) for s in grp["system_index"].tolist()]
    return out, have_row


def discover_granules_for_new_events(
    *,
    structure_ids: set[str] | set[int] | None = None,
    window_ids: set[int] | None = None,
    force: bool = False,
) -> None:
    """
    Phase 1: For windows that currently have no row in process_runs.csv,
    discover their Sentinel-2 granules and append to granules.csv + process_runs.csv.
    """
    # Refresh config to honor recent edits
    try:
        from .utils import refresh_config as _refresh

        _refresh()
    except Exception:
        pass
    rows = _load_event_rows(window_ids=window_ids, structure_ids=structure_ids)
    if not rows:
        print("⚠  No windows/structures found or dataset is empty - nothing to do")
        return

    # Windows that already have any mapping row (including marker rows)
    _, have_row = _load_mappings()

    processed = 0
    # Discover granules for events missing any mapping
    for i, row in enumerate(rows):
        eid = int(row.get("id", i))
        if (not force) and (eid in have_row):
            continue
        processed += 1

        centre_pt = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        products = sentinel2_system_indexes(
            centre_pt,
            str(row.get("start")),
            add_days_to_date(str(row.get("end")), 1),
        )
        if cfg.VERBOSE:
            print(f"window {eid}: {len(products)} product(s)")
        if products:
            _record_event_granules({eid: products})
        else:
            _record_event_granules({eid: []})
    if processed == 0:
        print(
            "0 windows matched filters (already discovered). Set overwrite=True to force."
        )


def process_event_granules(
    *,
    structure_ids: set[str] | set[int] | None = None,
    window_ids: set[int] | None = None,
    system_indexes: set[str] | None = None,
) -> None:
    """
    Phase 2: Using the provided filters, select relevant granules (from
    process_runs.csv joined with granules.csv), lightly filter by quality,
    then fully process each remaining granule (SGA grid, masks, MBSP, exports).
    """
    # Refresh config
    try:
        from .utils import refresh_config as _refresh

        _refresh()
    except Exception:
        pass
    rows = _load_event_rows()
    if not rows:
        print("⚠  No windows/structures found or dataset is empty - nothing to do")
        return

    # Build scene-filtered DB, then refine by explicit lists
    db = virtual_db()

    # Intersect with explicit filters if provided
    if system_indexes is not None and not db.empty:
        sys_set = {str(s) for s in system_indexes}
        db = db[db["system_index"].astype(str).isin(sys_set)]
    if window_ids is not None and not db.empty:
        wid_set = {int(w) for w in window_ids}
        db = db[db["window_id"].astype(int).isin(wid_set)]
    if structure_ids is not None:
        # Build allowed (window_id, system_index) pairs from unified_db
        pair_df = unified_db(structure_ids=list(structure_ids))
        if not pair_df.empty:
            pair_df = pair_df[pair_df["system_index"].astype(str).str.len() > 0]
            pairs = set(
                (int(r["window_id"]), str(r["system_index"]))
                for _, r in pair_df.iterrows()
            )
            if not db.empty:
                db = db[
                    db.apply(
                        lambda r: (int(r["window_id"]), str(r["system_index"]))
                        in pairs,
                        axis=1,
                    )
                ]

    start_time = time.time()
    active: list[ee.batch.Task] = []
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        futures = []
        if not db.empty:
            # Iterate unique (window_id, system_index) pairs
            for (eid, sid), grp in db.groupby(["window_id", "system_index"]):
                # Build site dict from the first row
                r = grp.iloc[0]
                site = {
                    "window_id": int(eid),
                    "lon": float(r["lon"]),
                    "lat": float(r["lat"]),
                    "structure_id": str(r.get("structure_id", "")),
                    "start": str(r.get("start")),
                    "end": add_days_to_date(str(r.get("end")), 1),
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


def main(
    cmd: str = "both",
    *,
    structure_ids: list[str | int] | None = None,
    window_ids: list[int] | None = None,
    system_indexes: list[str] | None = None,
):
    """Entry point for notebook-style execution.

    Filters can be provided programmatically; when None, config lists are used.
    """
    # Always refresh config at the top of a run so subsequent reads reflect edits.
    try:
        from .utils import refresh_config as _refresh

        _refresh()
    except Exception:
        pass
    # Resolve filters from config when not provided
    if structure_ids is not None:
        sids = set(str(x) for x in structure_ids)
    else:
        sids = set(str(x) for x in (getattr(cfg, "STRUCTURES_TO_PROCESS", []) or []))
    if window_ids is not None:
        wids = set(int(x) for x in window_ids)
    else:
        cfg_w = getattr(cfg, "WINDOWS_TO_PROCESS", []) or []
        wids = set(int(x) for x in cfg_w)
    if system_indexes is not None:
        gids = set(str(x) for x in system_indexes)
    else:
        gids = set(str(x) for x in (getattr(cfg, "GRANULES_TO_PROCESS", []) or []))

    if cmd == "discover":
        discover_granules_for_new_events(
            structure_ids=sids or None,
            window_ids=wids or None,
            force=bool(cfg.EXPORT_PARAMS.get("overwrite", False)),
        )
    elif cmd == "process":
        process_event_granules(
            structure_ids=sids or None,
            window_ids=wids or None,
            system_indexes=gids or None,
        )
    elif cmd == "both":
        discover_granules_for_new_events(
            structure_ids=sids or None,
            window_ids=wids or None,
            force=bool(cfg.EXPORT_PARAMS.get("overwrite", False)),
        )
        process_event_granules(
            structure_ids=sids or None,
            window_ids=wids or None,
            system_indexes=gids or None,
        )
    else:
        print("Usage: python -m offshore_methane.orchestrator [discover|process|both]")


# %%
# ------------------------------------------------------------------
if __name__ == "__main__":
    main(cmd="both", structure_ids=["x86"])

# %%
