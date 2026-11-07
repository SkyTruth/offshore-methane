# %%
"""Nightly flaring at GFW oil sites from VIIRS SDRs, for a date range.

1. NASA CMR footprints -> which ~86 s NOAA granules cover which sites.
2. Each granule (in parallel processes): M10 detection near sites, M11-M13
   confirmation and Planck fit, cloud mask at the site pixel (``granule.py``).
3. One row per site per local night: mean clear-sky radiant heat (MW, clear
   non-detections count as 0) and the matching BCM/yr rate.

Per-granule results are cached under ``out_dir/overpasses_v3/`` to resume.
Needs ``outputs/gfw_oil_sites.csv`` from ``fetch_gfw_oil.py``.
"""

from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import granule  # noqa: E402
import granule_coverage  # noqa: E402
import nodd  # noqa: E402

OUT_DIR = HERE / "outputs"
SITES_CSV = OUT_DIR / "gfw_oil_sites.csv"
PIPELINE_VERSION = "v3"
SZA_CLEAR = 97.0  # clear-overpass rule (MYC25Q4 README)
BCM_PER_MW_YEAR = 0.0116  # John Zink calibration slope (MYC25Q4 README)
SLICE_MARGIN = timedelta(seconds=20)
REQUIRED = ("geo", "M10", "M11", "M12", "M13")


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def build_jobs(sites: pd.DataFrame, start: date, end: date) -> list[dict]:
    """Granule jobs covering local nights ``start``..``end`` for all satellites.

    The ~01:30 local overpasses of night N fall on UTC dates N and N+1.
    """
    utc_days = _days(start, end + timedelta(days=1))
    session = requests.Session()
    jobs = []
    for sat, short_name in granule_coverage.PRODUCTS.items():
        slices = pd.concat(
            [
                granule_coverage.site_slices(
                    sites, granule_coverage.cmr_granules(short_name, d.isoformat())
                )
                for d in utc_days
            ],
            ignore_index=True,
        )
        for col in ("slice_start", "slice_end"):
            slices[col] = pd.to_datetime(slices[col]).dt.tz_convert(None)
        index: dict[str, dict] = {}
        for d in [utc_days[0] - timedelta(days=1), *utc_days]:
            day = datetime(d.year, d.month, d.day)
            index.update(nodd.granule_index(sat, day, session))
        bucket = nodd.BUCKETS[sat][0]
        site_xy = sites.set_index("site_id")[["lat", "lon"]]
        for gid, g in sorted(index.items()):
            if not all(k in g for k in REQUIRED):
                continue
            hit = slices[
                (slices.slice_start <= g["end"] + SLICE_MARGIN)
                & (slices.slice_end >= g["start"] - SLICE_MARGIN)
            ]
            if hit.empty:
                continue
            ids = sorted({int(s) for s in " ".join(hit.site_ids).split()})
            xy = site_xy.loc[ids]
            jobs.append(
                {
                    "satellite": sat,
                    "bucket": bucket,
                    "gid": gid,
                    "start": g["start"],
                    "keys": {k: g.get(k) for k in nodd.PRODUCTS},
                    "sites": list(zip(ids, xy.lat, xy.lon)),
                }
            )
        print(f"{sat}: {len(jobs)} granule jobs so far")
    return jobs


def _cache_path(job: dict, cache_dir: Path) -> Path:
    return cache_dir / job["satellite"] / f"{job['gid']}.csv"


def run_jobs(
    jobs: list[dict], cache_dir: Path, workers: int = 8, attempts: int = 3
) -> pd.DataFrame:
    """Process granules in parallel, caching one CSV per granule.

    Failed granules (usually transient network errors) are retried up to
    ``attempts`` passes; any still missing are reported and left uncached.
    """
    for attempt in range(1, attempts + 1):
        todo = [j for j in jobs if not _cache_path(j, cache_dir).exists()]
        print(
            f"pass {attempt}: {len(jobs)} granules, "
            f"{len(jobs) - len(todo)} cached, {len(todo)} to run"
        )
        if not todo or not _run_pass(todo, cache_dir, workers):
            break
    frames = [
        pd.read_csv(p)
        for p in (_cache_path(j, cache_dir) for j in jobs)
        if p.exists() and p.stat().st_size > 0
    ]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_pass(todo: list[dict], cache_dir: Path, workers: int) -> int:
    """One parallel pass over ``todo``; returns the number of failed granules."""
    t0, done, nbytes, errors = time.time(), 0, 0, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(granule.process_granule, j): j for j in todo}
        for fut in as_completed(futures):
            job = futures[fut]
            done += 1
            try:
                recs = fut.result()
            except Exception:
                errors.append({"granule": job["gid"], "satellite": job["satellite"]})
                print(
                    f"ERROR {job['satellite']} {job['gid']}\n{traceback.format_exc()}"
                )
                continue
            path = _cache_path(job, cache_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(recs, columns=granule.RECORD_COLUMNS)
            df.to_csv(path, index=False)
            nbytes += int(df["bytes_granule"].max()) if len(df) else 0
            if done % 25 == 0 or done == len(todo):
                mins = (time.time() - t0) / 60
                print(
                    f"{done}/{len(todo)} granules, {nbytes / 1e9:.2f} GB "
                    f"(observed-site granules), {mins:.1f} min"
                )
    if errors:
        print(f"{len(errors)} granules failed this pass")
    return len(errors)


def dedupe_overpasses(over: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-satellite/site runs with consecutive gaps <10 s.

    Keep the smallest site-pixel distance; ties keep the earliest record.
    """
    if over.empty:
        return over.copy()
    ordered = over.copy()
    ordered["_time"] = pd.to_datetime(ordered["time_utc"])
    ordered = ordered.sort_values(["satellite", "site_id", "_time"], kind="stable")
    gap = ordered.groupby(["satellite", "site_id"])["_time"].diff()
    ordered["_observation"] = (gap.isna() | (gap >= pd.Timedelta(seconds=10))).cumsum()
    return (
        ordered.sort_values("dist_m", kind="stable")
        .drop_duplicates("_observation")
        .sort_values("_time", kind="stable")
        .drop(columns=["_time", "_observation"])
        .reset_index(drop=True)
    )


def strict_clear(over: pd.DataFrame) -> pd.Series:
    """Shared clear-overpass rule, including isolated cloudy detections."""
    return (over["sza"] >= SZA_CLEAR) & (
        over["cloud_mask"].isin([0, 1])
        | (over["detected"] & (over["cloud_clear_neighbors"] >= 6))
    )


def site_nights(
    over: pd.DataFrame, sites: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """One row per site per local night using strict cloud clearing.

    Detections under non-isolated cloud are excluded from the clear-sky value
    (``rh_mw``) but recorded in ``n_detected_cloud``; a night whose only
    detections are such is ``flaring_cloud`` (the flare was lit, its clear-sky
    heat is unmeasured). Detclear columns retain the old detected-or-clear rule
    for comparison. Counts follow deduplication.
    """
    over = dedupe_overpasses(over)
    over = over.merge(sites[["site_id", "lon"]], on="site_id")
    local = pd.to_datetime(over["time_utc"]) + pd.to_timedelta(over["lon"] / 15, "h")
    over["night"] = (local - pd.Timedelta(hours=12)).dt.date
    over = over[(over["sza"] >= SZA_CLEAR)].copy()
    over["clear_detclear"] = over["detected"] | over["cloud_mask"].isin([0, 1])
    over["clear"] = strict_clear(over)
    over["detected_strict"] = over["detected"] & over["clear"]
    over["detected_cloud"] = over["detected"] & ~over["clear"]
    over["rh_detclear"] = over["rh_mw"].where(over["clear_detclear"], 0.0)
    over["temp_strict"] = over["temp_k"].where(over["detected_strict"])
    over["rh_clear"] = over["rh_mw"].where(over["clear"], 0.0)
    agg = (
        over.groupby(["night", "site_id"])
        .agg(
            n_overpasses=("clear", "size"),
            n_clear=("clear", "sum"),
            n_detected=("detected_strict", "sum"),
            n_detected_cloud=("detected_cloud", "sum"),
            n_clear_detclear=("clear_detclear", "sum"),
            rh_sum_detclear=("rh_detclear", "sum"),
            rh_sum=("rh_clear", "sum"),
            rh_max_mw=("rh_clear", "max"),
            temp_k=("temp_strict", "max"),
        )
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [_days(start, end), sites["site_id"]], names=["night", "site_id"]
    ).to_frame(index=False)
    out = grid.merge(agg, on=["night", "site_id"], how="left")
    out = out.merge(sites[["site_id", "lat", "lon"]], on="site_id")
    counts = ("n_overpasses", "n_clear", "n_detected", "n_detected_cloud")
    for col in (*counts, "n_clear_detclear"):
        out[col] = out[col].fillna(0).astype(int)
    out["rh_mw"] = (out["rh_sum"] / out["n_clear"]).where(out["n_clear"] > 0)
    out["bcm_per_year"] = BCM_PER_MW_YEAR * out["rh_mw"]
    out["rh_mw_detclear"] = (out["rh_sum_detclear"] / out["n_clear_detclear"]).where(
        out["n_clear_detclear"] > 0
    )
    out["bcm_per_year_detclear"] = BCM_PER_MW_YEAR * out["rh_mw_detclear"]
    out["status"] = "not_observed"
    out.loc[out["n_overpasses"] > 0, "status"] = "cloudy"
    out.loc[out["n_clear"] > 0, "status"] = "clear_no_flare"
    out.loc[out["n_detected_cloud"] > 0, "status"] = "flaring_cloud"
    out.loc[out["n_detected"] > 0, "status"] = "flaring"
    cols = [
        "night",
        "site_id",
        "lat",
        "lon",
        "status",
        "rh_mw",
        "bcm_per_year",
        "rh_max_mw",
        "temp_k",
        "n_overpasses",
        "n_clear",
        "n_detected",
        "n_detected_cloud",
        "n_clear_detclear",
        "rh_mw_detclear",
        "bcm_per_year_detclear",
    ]
    return out[cols].sort_values(["night", "site_id"]).reset_index(drop=True)


def main(
    start: str = "2026-10-29",
    end: str = "2026-11-04",
    workers: int = 8,
    sites_csv: Path = SITES_CSV,
    out_dir: Path = OUT_DIR,
    max_granules: int | None = None,
) -> pd.DataFrame:
    """Run nights ``start``..``end`` (inclusive, local evening dates)."""
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(sites_csv)
    jobs = build_jobs(sites, d0, d1)
    if max_granules:
        jobs = jobs[:max_granules]
    over = run_jobs(jobs, out_dir / f"overpasses_{PIPELINE_VERSION}", workers)
    tag = f"{start}_{end}"
    over.to_csv(out_dir / f"overpasses_{tag}.csv", index=False)
    nights = site_nights(over, sites, d0, d1)
    nights.to_csv(out_dir / f"site_nights_{tag}.csv", index=False)
    print(nights["status"].value_counts().to_string())
    return nights


# %%
if __name__ == "__main__":
    main()
