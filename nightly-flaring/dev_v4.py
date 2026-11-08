# %%
"""Reproduce the review-2 regression granules without changing v3 outputs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
import json

import pandas as pd
import granule
import pipeline

CASES = {
    "d20261010_t2330408": [3802, 3801],
    "d20261011_t2311458": [3802],
    "d20261014_t2216245": [4510],
    "d20261016_t1810529": [5869],
    "d20261022_t2126021": [4452],
    "d20261027_t0758447": [876],
    "d20261102_t2119529": [4280],
    "d20261102_t2258015": [3205],
    "d20261103_t2240322": [3804],
}


def run_case(gid: str) -> list[dict]:
    """Use the normal job builder, selecting only this NOAA-20 granule."""
    sites = pd.read_csv(pipeline.SITES_CSV)
    day = date(int(gid[1:5]), int(gid[5:7]), int(gid[7:9]))
    jobs = pipeline.build_jobs(sites, day, day)
    jobs = [j for j in jobs if j["satellite"] == "NOAA-20" and j["gid"] == gid]
    if len(jobs) != 1:
        raise ValueError(f"{gid}: expected one job, got {len(jobs)}")
    return granule.process_granule(jobs[0])[0]


def main() -> None:
    """Save all records and a compact regression comparison."""
    out = pipeline.OUT_DIR / "dev_v4"
    out.mkdir(exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=9) as pool:
        for future in as_completed([pool.submit(run_case, gid) for gid in CASES]):
            records.extend(future.result())
    new = pd.DataFrame(records)
    new.to_csv(out / "regression_overpasses.csv", index=False)
    old = pd.read_csv(pipeline.OUT_DIR / "overpasses_2026-10-06_2026-11-04.csv")
    rows = []
    for gid, ids in CASES.items():
        for sid in ids:
            row = dict(granule=gid, satellite="NOAA-20", site_id=sid)
            for version, frame in [("v3", old), ("v4", new)]:
                hit = frame[
                    (frame.granule == gid)
                    & (frame.site_id == sid)
                    & (frame.satellite == "NOAA-20")
                ]
                if len(hit) != 1:
                    raise ValueError(f"{version} {gid} {sid}: {len(hit)} rows")
                for key in ["rh_mw", "temp_k", "bands", "fit_flag", "n_sources"]:
                    if key in hit:
                        row[f"{version}_{key}"] = hit.iloc[0][key]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "regressions.csv", index=False)
    (out / "regressions.json").write_text(json.dumps(rows, indent=2, default=int))


# %%
if __name__ == "__main__":
    main()
