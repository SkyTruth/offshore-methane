# %%
"""Run the review-3 cases for v4 and v5 through normal coverage jobs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import json
import subprocess

import pandas as pd
import granule
import pipeline

CASES = {
    "NOAA-20": "d20261010_t2330408 d20261011_t2311458 d20261013_t0043475 d20261014_t0704356 d20261014_t2216245 d20261016_t1810529 d20261021_t2144572 d20261022_t2126021 d20261022_t2305365 d20261027_t0758447 d20261030_t2216381 d20261102_t2119529 d20261102_t2258015 d20261103_t2240322 d20261105_t2204060".split(),
    "NOAA-21": "d20261013_t2317594 d20261022_t0837523 d20261030_t0748551 d20261030_t2118195".split(),
    "SNPP": "d20261020_t2323324 d20261029_t0703487".split(),
}
OUT = pipeline.OUT_DIR / "dev_v5"
V4_BASELINE = "d6c50b5"


def run_case(satellite: str, gid: str) -> None:
    """Build normal jobs, run both versions, save site and constituent tables."""
    import importlib.util

    sites = pd.read_csv(pipeline.SITES_CSV)
    day = date(int(gid[1:5]), int(gid[5:7]), int(gid[7:9]))
    jobs = pipeline.build_jobs(sites, day, day)
    (job,) = [j for j in jobs if j["satellite"] == satellite and j["gid"] == gid]
    stem = OUT / "cases" / f"{satellite}_{gid}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for version in ("v4", "v5"):
        if version == "v4":
            spec = importlib.util.spec_from_file_location(
                "baseline_nf", OUT / "baseline_v4/nightfire.py"
            )
            nf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nf)
            spec = importlib.util.spec_from_file_location(
                "baseline_granule", OUT / "baseline_v4/granule.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.nf = nf
            captured = []
            merge = module.merge_scan_copies

            def capture(copies, tree):
                sources = merge(copies, tree)
                captured.append(dict(copies=copies, groups=sources))
                return sources

            module.merge_scan_copies = capture
            records = module.process_granule(job)
            Path(f"{stem}_v4_groups.json").write_text(
                json.dumps(
                    captured,
                    default=lambda x: list(x) if isinstance(x, set) else x.item(),
                )
            )
        else:
            records, copies = granule.process_granule(job)
            pd.DataFrame(copies, columns=granule.COPY_COLUMNS).to_csv(
                f"{stem}_v5_copies.csv", index=False
            )
        pd.DataFrame(records).to_csv(f"{stem}_{version}.csv", index=False)
    print(f"DONE {satellite} {gid}", flush=True)


def main(workers: int = 12) -> None:
    """Run exactly the requested 21 granule cases; preserve historical outputs."""
    baseline = OUT / "baseline_v4"
    baseline.mkdir(parents=True, exist_ok=True)
    for name in ("nightfire.py", "granule.py"):
        content = subprocess.check_output(
            ["git", "show", f"{V4_BASELINE}:nightly-flaring/{name}"]
        )
        (baseline / name).write_bytes(content)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_case, sat, gid)
            for sat, gids in CASES.items()
            for gid in gids
        ]
        for future in as_completed(futures):
            future.result()


def audit_night(workers: int = 12) -> None:
    """Finalize band-rule diagnostics on the same night after the first run."""
    night = date(2026, 10, 29)
    jobs = pipeline.build_jobs(pd.read_csv(pipeline.SITES_CSV), night, night)
    cache = OUT / "overpasses_v5"
    todo = []
    for job in jobs:
        path = pipeline._cache_path(job, cache)
        if not pipeline.cache_complete(job, cache):
            todo.append(job)
            continue
        records = pd.read_csv(path)
        if "n_band_rule_changed" in records:
            continue
        if records.n_fitted.sum() > 0:
            todo.append(job)
        else:
            records["n_band_rule_changed"] = 0
            records.to_csv(path, index=False)
            cp = pipeline._copies_path(job, cache)
            copies = pd.read_csv(cp)
            copies["band_rule_changed"] = False
            copies.to_csv(cp, index=False)
    print(
        f"Band-rule diagnostic finalization: {len(todo)}/{len(jobs)} same-night granules",
        flush=True,
    )
    if todo and pipeline._run_pass(todo, cache, workers):
        raise RuntimeError("Band-rule diagnostic finalization failed")
    over = pipeline.run_jobs(jobs, cache, workers)
    over.to_csv(OUT / "overpasses_2026-10-29_2026-10-29.csv", index=False)
    copies = pd.concat(
        [pd.read_csv(pipeline._copies_path(j, cache)) for j in jobs], ignore_index=True
    )
    copies.to_csv(OUT / "copies_2026-10-29_2026-10-29.csv", index=False)
    nights = pipeline.site_nights(over, pd.read_csv(pipeline.SITES_CSV), night, night)
    nights.to_csv(OUT / "site_nights_2026-10-29_2026-10-29.csv", index=False)


# %%
if __name__ == "__main__":
    main()
