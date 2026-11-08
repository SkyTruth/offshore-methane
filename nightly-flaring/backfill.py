# %%
"""Backfill runner: warm the CMR cache, run the pipeline with retries, validate.

A backfill outlives one session, so run it detached, e.g.
``nohup .venv/bin/python -u -c "import backfill; backfill.main()" > log &``.
Progress goes to stdout; ``status.json`` in ``out_dir`` records the phase.
The per-granule cache makes every restart resume (D44); jobs run oldest
first (D49) because NODD deletes the oldest nights first.
"""

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import granule_coverage
import pipeline
import validate_myc25


def _status(out_dir: Path, **fields) -> None:
    fields["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = out_dir / "status.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(fields, indent=2))
    tmp.replace(path)


def warm_cmr_cache(start: date, end: date, threads: int = 8) -> list:
    """Fetch CMR footprints for UTC days start..end+1 in parallel (D47)."""
    days = [start + timedelta(d) for d in range((end - start).days + 2)]
    tasks = [
        (sn, d.isoformat()) for sn in granule_coverage.PRODUCTS.values() for d in days
    ]
    failed = []
    with ThreadPoolExecutor(threads) as ex:
        futures = {ex.submit(granule_coverage.cmr_granules, *t): t for t in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:  # retried later by build_jobs
                failed.append(futures[fut])
                print(f"CMR failed {futures[fut]}: {exc!r}"[:200], flush=True)
            if n % 20 == 0:
                print(f"CMR cache {n}/{len(tasks)}", flush=True)
    return failed


def main(
    start: str = "2026-08-09",
    end: str = "2026-10-05",
    out_dir: Path = pipeline.OUT_DIR / "v5_backfill",
    workers: int = 12,
    attempts: int = 5,
) -> None:
    """Run local nights ``start``..``end`` into ``out_dir``, then validate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    _status(out_dir, phase="cmr_cache", start=start, end=end)
    failed = warm_cmr_cache(d0, d1)
    print(f"CMR cache warm, {len(failed)} failed, {(time.time() - t0) / 60:.1f} min")
    for attempt in range(1, attempts + 1):
        _status(out_dir, phase="pipeline", attempt=attempt, start=start, end=end)
        try:
            pipeline.main(start=start, end=end, workers=workers, out_dir=out_dir)
            break
        except Exception:
            print(f"pipeline attempt {attempt} failed\n{traceback.format_exc()}")
            if attempt == attempts:
                _status(out_dir, phase="failed", attempt=attempt)
                raise
            time.sleep(60 * attempt)
    print(f"pipeline done {(time.time() - t0) / 60:.1f} min", flush=True)
    _status(out_dir, phase="validate", start=start, end=end)
    validate_myc25.main(tag=f"{start}_{end}", out_dir=out_dir)
    summary = json.loads((out_dir / f"run_summary_{start}_{end}.json").read_text())
    _status(
        out_dir,
        phase="done",
        start=start,
        end=end,
        minutes=round((time.time() - t0) / 60, 1),
        jobs_expected=summary["expected"],
        jobs_complete=summary["complete"],
    )
    print(f"all done {(time.time() - t0) / 60:.1f} min", flush=True)


# %%
if __name__ == "__main__":
    main()
