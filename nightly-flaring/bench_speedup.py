# %%
"""Bounded single-night T0-T7 benchmarks; default main() writes an offline plan.

Run the generated outputs/speedup/run_protocol.py on a network-enabled session.
The four approved nights are separate experiments, never a backfill invocation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

import granule_coverage
import orbit
import pipeline
import speedup
import store

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs/speedup"
NIGHTS = ("2026-08-14", "2026-09-20", "2026-10-25", "2024-06-20")
REFERENCES = {
    "2026-08-14": HERE / "outputs/reference_v5_backfill",
    "2026-09-20": HERE / "outputs/reference_v5_backfill",
    "2026-10-25": HERE / "outputs/reference_v5_30n",
}
MODES = (
    "control",
    "t1",
    "t7",
    "t1_t2",
    "best",
    "tier_order",
    "reduced",
    "satellite_first",
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    tmp.replace(path)


def _output(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT.resolve()):
        raise ValueError("benchmark outputs must stay under outputs/speedup")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _meter_http():
    """Count CMR/listing traffic, including retry attempts, in the parent only."""
    original = requests.sessions.Session.request
    stats = dict(requests=0, bytes_fetched=0)

    def counted(session, *args, **kwargs):
        stats["requests"] += 1
        response = original(session, *args, **kwargs)
        stats["bytes_fetched"] += len(response.content)
        return response

    requests.sessions.Session.request = counted
    try:
        yield stats
    finally:
        requests.sessions.Session.request = original


def tier1_ids(output: Path) -> tuple[int, ...]:
    """Freeze the lit-site union from the 88 reference nights, read-only."""
    path = output / "tier1_ids.json"
    if path.exists():
        return tuple(json.loads(path.read_text()))
    ids = set()
    nights = set()
    for root in dict.fromkeys(REFERENCES.values()):
        tags = {
            p.stem.removeprefix("site_nights_")
            for suffix in ("csv", "parquet")
            for p in root.glob(f"site_nights_*.{suffix}")
        }
        if len(tags) != 1:
            raise ValueError(f"expected one reference nightly table in {root}")
        frame = store.read_segment(
            root, tags.pop(), "site_nights", columns=["site_id", "status", "night"]
        )
        nights.update(frame.night)
        ids.update(
            frame.loc[
                frame.status.isin(["flaring", "flaring_cloud"]), "site_id"
            ].astype(int)
        )
    _write_json(path, sorted(ids))
    _write_json(
        output / "tier1_evidence.json",
        dict(
            n_nights=len(nights),
            n_sites=len(ids),
            method="lit on >=1 reference night; no MYC additions",
        ),
    )
    return tuple(sorted(ids))


def mode_options(
    name: str, ids: tuple[int, ...], tle_dir: Path, block_size=16384
) -> dict:
    if name not in MODES:
        raise ValueError(f"unknown mode {name!r}")
    if name == "control":
        return {}
    opts = speedup.Options(
        header_prefilter=name != "t7",
        windowed_geo=name not in ("control", "t1", "t7"),
        orbit_whitelist=name not in ("control", "t1", "t1_t2"),
        tier_order=name == "tier_order",
        tier2_detections_only=name == "reduced",
        satellite_first="NOAA-20" if name == "satellite_first" else None,
        tier1_ids=ids,
        tle_dir=str(tle_dir),
        block_size=block_size,
    )
    return asdict(opts)


def _csv_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    return rows[0], rows[1:]


def _without_columns(header, rows, excluded):
    indexes = [i for i, c in enumerate(header) if c not in excluded]
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([header[i] for i in indexes])
    writer.writerows([[row[i] for i in indexes] for row in rows])
    return out.getvalue().encode()


def compare(jobs, cache: Path, reference: Path, opts: dict) -> dict:
    """Compare exact CSV field strings, never pandas-rounded numeric values.

    Only bytes_granule may differ for preserving modes. In T4b, only tier-2
    non-detection rows and their per-site cloud fields are deliberately lost.
    """
    diffs, changed_columns, dropped = [], set(), 0
    raw_equal = checked = skipped_with_rows = 0
    config = speedup.Options(**opts)
    for job in jobs:
        reduced = config.tier2_detections_only and not speedup.tier1(job, config)
        for copies in (False, True):
            path_fn = pipeline._copies_path if copies else pipeline._cache_path
            actual, expected = path_fn(job, cache), path_fn(job, reference)
            label = f"{job['satellite']}/{actual.name}"
            if not actual.exists() or not expected.exists():
                diffs.append(dict(file=label, reason="missing result or reference"))
                continue
            checked += 1
            ah, ar = _csv_rows(actual)
            eh, er = _csv_rows(expected)
            if actual.read_bytes() == expected.read_bytes():
                raw_equal += 1
            if ah != eh:
                diffs.append(dict(file=label, reason="schema differs"))
                continue
            if not copies:
                stats = pipeline._stats_path(job, cache)
                if (
                    stats.exists()
                    and json.loads(stats.read_text())["attempts"][-1].get("skipped")
                    and er
                ):
                    skipped_with_rows += 1
            if reduced and not copies:
                keep = [r for r in er if r[eh.index("detected")] == "True"]
                dropped += len(er) - len(keep)
                er = keep
            excluded = set() if copies else {"bytes_granule"}
            if reduced and not copies:
                excluded |= {"cloud_mask", "cloud_clear_neighbors"}
            if len(ar) == len(er):
                changed_columns.update(
                    c
                    for i, c in enumerate(ah)
                    if any(a[i] != e[i] for a, e in zip(ar, er))
                )
            same = (
                actual.read_bytes() == expected.read_bytes()
                if copies
                else _without_columns(ah, ar, excluded)
                == _without_columns(eh, er, excluded)
            )
            if not same:
                diffs.append(
                    dict(file=label, reason="unexpected CSV content difference")
                )
    return dict(
        identical=not diffs and checked == 2 * len(jobs),
        result_preserving=not config.tier2_detections_only,
        raw_identical_files=raw_equal,
        checked_files=checked,
        expected_files=2 * len(jobs),
        changed_columns=sorted(changed_columns),
        intentional_dropped_rows=dropped,
        skipped_jobs_with_reference_rows=skipped_with_rows,
        differences=diffs,
    )


def reference_scope(jobs, night, sites, cache):
    """Match the reference backfill's CMR scope at single-night UTC boundaries.

    A backfill can include a preceding/following day's CMR slice in a boundary
    granule. Rebuild those site unions from CMR, without listing extra SDR days
    or reading result values. Keep exactly the single-night granule keys.
    """
    if night not in REFERENCES:
        return jobs
    day = date.fromisoformat(night)
    xy = sites.set_index("site_id")[["lat", "lon"]]
    for satellite, short in granule_coverage.PRODUCTS.items():
        frames = [
            granule_coverage.site_slices(
                sites,
                granule_coverage.cmr_granules(
                    short, (day + timedelta(days=offset)).isoformat(), cache_dir=cache
                ),
            )
            for offset in (-1, 2)
        ]
        frame = pd.concat(frames, ignore_index=True)
        if frame.empty:
            continue
        for col in ("slice_start", "slice_end"):
            frame[col] = pd.to_datetime(frame[col]).dt.tz_convert(None)
        for job in jobs:
            if job["satellite"] != satellite:
                continue
            start, end = orbit.interval(job)
            hit = frame[
                (frame.slice_start <= end + pipeline.SLICE_MARGIN)
                & (frame.slice_end >= start - pipeline.SLICE_MARGIN)
            ]
            ids = sorted(
                set(int(site[0]) for site in job["sites"])
                | {int(i) for text in hit.site_ids for i in text.split()}
            )
            job["sites"] = [
                (i, float(xy.loc[i, "lat"]), float(xy.loc[i, "lon"])) for i in ids
            ]
    return jobs


def _prepare(night, output):
    if night not in NIGHTS:
        raise ValueError("only the four SPEC benchmark nights are permitted")
    folder = _output(output / night)
    tag = f"{night}_{night}"
    if pipeline.expected_jobs_path(folder, tag).exists():
        return pipeline.read_expected_jobs(folder, tag)
    # Read the prepared CMR cache, but every new cache write stays in speedup/.
    cache = output / "cmr_cache"
    shutil.copytree(HERE / "outputs/cmr_cache", cache, dirs_exist_ok=True)
    old_cache = granule_coverage.CMR_CACHE_DIR
    granule_coverage.CMR_CACHE_DIR = cache
    t0, cpu = time.perf_counter(), time.process_time()
    try:
        with _meter_http() as counts:
            jobs = pipeline.build_jobs(
                pd.read_csv(pipeline.SITES_CSV),
                date.fromisoformat(night),
                date.fromisoformat(night),
                source="azure",
            )
            jobs = reference_scope(jobs, night, pd.read_csv(pipeline.SITES_CSV), cache)
    finally:
        granule_coverage.CMR_CACHE_DIR = old_cache
    pipeline.persistent_jobs(jobs, folder, tag)
    _write_json(
        folder / "listing_traffic.json",
        dict(
            **counts,
            wall_s=time.perf_counter() - t0,
            cpu_s=time.process_time() - cpu,
            jobs=len(jobs),
        ),
    )
    return jobs


def _milestones(jobs, cache, ids, started):
    end = {}
    stats = []
    for job in jobs:
        path = pipeline._stats_path(job, cache)
        if not path.exists():
            continue
        attempts = json.loads(path.read_text())["attempts"]
        stats.extend((job, attempt) for attempt in attempts)
        if not pipeline.cache_complete(job, cache):
            continue
        record = attempts[-1]
        seconds = max(
            0,
            (
                datetime.fromisoformat(
                    record.get("results_written_utc", record["completed_utc"])
                )
                - started
            ).total_seconds(),
        )
        end[(job["satellite"], job["gid"])] = seconds

    def milestone(selected):
        return (
            max((end[(j["satellite"], j["gid"])] for j in selected), default=0)
            if all((j["satellite"], j["gid"]) in end for j in selected)
            else None
        )

    first = [j for j in jobs if j["satellite"] == "NOAA-20"]
    return dict(
        tier1_complete_s=milestone(
            [j for j in jobs if set(ids).intersection(int(s[0]) for s in j["sites"])]
        ),
        noaa20_complete_s=milestone(first),
        noaa20_jobs=len(first),
        noaa20_bytes=sum(
            s["bytes_fetched"] for j, s in stats if j["satellite"] == "NOAA-20"
        ),
    )


def run_mode(
    night, jobs, name, workers, output, ids, label=None, max_jobs=None, block_size=16384
):
    label = label or name
    run = _output(output / night / f"{label}-w{workers}-b{block_size}")
    result_path = run / "result.json"
    opts = mode_options(name, ids, OUTPUT / "tle", block_size)
    jobs = jobs[:max_jobs] if max_jobs else jobs
    jobs = speedup.configure(jobs, speedup.Options(**opts)) if opts else jobs
    signature = hashlib.sha256(
        json.dumps(
            dict(jobs=jobs, sampled=max_jobs is not None), sort_keys=True, default=str
        ).encode()
    ).hexdigest()
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("signature") != signature:
            raise ValueError("benchmark scope/options changed; use a new output folder")
        if (
            not result["complete"]
            or not result["correctness"]["identical"]
            or not result["traffic"]["accounting_complete"]
        ):
            raise RuntimeError(f"previous benchmark failed its gate: {result_path}")
        return result
    cache = run / "overpasses_v5"
    resumed = any(pipeline.cache_complete(j, cache) for j in jobs)
    if resumed:
        raise RuntimeError(
            "partial processing cache cannot be a cold run; use a fresh output folder"
        )
    started, wall, cpu = (
        datetime.now(timezone.utc),
        time.perf_counter(),
        time.process_time(),
    )
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = child.ru_utime + child.ru_stime
    pipeline.run_jobs(jobs, cache, workers=workers)
    elapsed = time.perf_counter() - wall
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = child.ru_utime + child.ru_stime - child_cpu + time.process_time() - cpu
    traffic = pipeline.traffic_summary(jobs, cache)
    complete = all(pipeline.cache_complete(j, cache) for j in jobs)
    reference = (
        REFERENCES.get(night, output / night / "control-w12-b16384") / "overpasses_v5"
    )
    correctness = compare(jobs, cache, reference, opts)
    if name == "control" and night not in REFERENCES:
        correctness["reference_kind"] = "historic control is the new reference"
    records = [
        json.loads(pipeline._stats_path(j, cache).read_text())["attempts"][-1]
        for j in jobs
        if pipeline._stats_path(j, cache).exists()
    ]
    result = dict(
        night=night,
        signature=signature,
        mode=name,
        label=label,
        workers=workers,
        block_size=block_size,
        jobs=len(jobs),
        complete=complete,
        resumed=resumed,
        sampled=max_jobs is not None,
        wall_s=elapsed,
        cpu_s=cpu_s,
        mean_cpu_cores=cpu_s / elapsed,
        mb_per_s=traffic["bytes_fetched"] / 1e6 / elapsed,
        granules_per_s=len(jobs) / elapsed,
        jobs_skipped=sum(r.get("skipped", False) for r in records),
        empty_jobs=sum(r.get("n_rows", 0) == 0 for r in records),
        traffic=traffic,
        correctness=correctness,
        options=opts,
        **_milestones(jobs, cache, ids, started),
    )
    reasons = {}
    for r in records:
        for key in ("prefilter", "orbit", "geo_mode"):
            if key in r:
                k = key + ":" + r[key]
                reasons[k] = reasons.get(k, 0) + 1
    result["path_counts"] = reasons
    result["noaa20_byte_share"] = (
        result["noaa20_bytes"] / traffic["bytes_fetched"]
        if traffic["bytes_fetched"]
        else None
    )
    result["noaa20_completion_fraction"] = (
        result["noaa20_complete_s"] / elapsed
        if result["noaa20_complete_s"] is not None
        else None
    )
    _write_json(result_path, result)
    if (
        not complete
        or not correctness["identical"]
        or not traffic["accounting_complete"]
    ):
        raise RuntimeError(
            f"benchmark correctness/completeness gate failed: {result_path}"
        )
    return result


def report(output):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("*/*/result.json"))]
    _write_json(output / "results.json", rows)
    flat = []
    for r in rows:
        flat.append(
            {
                k: r[k]
                for k in (
                    "night",
                    "mode",
                    "label",
                    "workers",
                    "block_size",
                    "jobs",
                    "jobs_skipped",
                    "wall_s",
                    "cpu_s",
                    "mean_cpu_cores",
                    "mb_per_s",
                    "granules_per_s",
                    "tier1_complete_s",
                    "noaa20_complete_s",
                    "noaa20_jobs",
                    "noaa20_bytes",
                    "resumed",
                    "sampled",
                )
            }
            | r["traffic"]
            | dict(
                identical=r["correctness"]["identical"],
                result_preserving=r["correctness"]["result_preserving"],
            )
        )
    pd.DataFrame(
        flat,
        columns=list(flat[0])
        if flat
        else [
            "night",
            "mode",
            "workers",
            "jobs",
            "bytes_fetched",
            "requests",
            "wall_s",
            "identical",
        ],
    ).to_csv(output / "results.csv", index=False)
    lines = [
        "# Speed-up benchmark results",
        "",
        "Traffic includes every processing attempt and HEAD. Listing setup is separate in each night's listing_traffic.json. Local TLE reads are charged in worker CPU/wall time; there are no geometry audits or shared HTTP caches. No site-row byte estimates are used.",
        "",
        "| Night | Mode | Workers | GB | Requests | Minutes | Identical* |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['night']} | {r['label']} | {r['workers']} | {r['traffic']['bytes_fetched'] / 1e9:.3f} | {r['traffic']['requests']} | {r['wall_s'] / 60:.2f} | {r['correctness']['identical']} |"
        )
    lines += [
        "",
        "*Exact CSV strings after excluding bytes_granule; T4b additionally drops tier-2 non-detections and sets their site cloud fields unknown. Raw identical-file counts and every changed column are in results.json. Copy CSVs have no exclusions. Resumed runs are unsuitable for timing.",
        "",
        "Inspect full-night equivalence and interleaved control timings before changing defaults. T1/T7/T2 use cold certificates with fallback; every mode has its own result folder and fresh workers/HTTP caches.",
    ]
    if rows:
        controls = [
            r["wall_s"]
            for r in rows
            if r["mode"] == "control"
            and not r["resumed"]
            and r["complete"]
            and not r["sampled"]
        ]
        if controls:
            projected = 220 * pd.Series(controls).median() / 2 / 3600
            lines += [
                "",
                f"Control projection: {projected:.2f} h for 220 consecutive nights, using half the median standalone two-UTC-day processing time. This assumes shared adjacent-day granules; excludes listings and is an estimate, not a measured backfill. No optimized cold projection is claimed.",
            ]
    (output / "LIVE_REPORT.md").write_text("\n".join(lines) + "\n")
    return rows


def run_bounded_protocol(out_dir=OUTPUT / "cold_r2", budget_s=9000):
    """Supervise a separate process group so even active HTTP stops at budget.

    A hard stop cannot account for responses still in flight: mark that traffic
    unknown and leave incomplete output folders for inspection, never timing.
    """
    output = _output(out_dir)
    ledger = output / "protocol_budget.json"
    used = json.loads(ledger.read_text())["elapsed_s"] if ledger.exists() else 0
    remaining = max(0, budget_s - used)
    child = output / "_protocol_child.py"
    child.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import bench_speedup\n"
        'if __name__ == "__main__":\n'
        f"    bench_speedup.main(action='protocol', out_dir={str(output)!r}, budget_s={budget_s!r})\n"
    )
    if remaining <= 0:
        return dict(budget_exhausted=True, output=str(output))
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, str(child)], start_new_session=True)
    try:
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _write_json(
                output / "budget_stop.json",
                dict(
                    hard_deadline=True,
                    elapsed_s=budget_s,
                    accounting_complete=False,
                    reason="Stopped active process group at 2.5-hour budget; in-flight traffic unknown. Incomplete runs are not timing evidence.",
                ),
            )
            _write_json(ledger, dict(elapsed_s=budget_s, budget_s=budget_s))
            return dict(budget_exhausted=True, output=str(output))
        if code:
            raise RuntimeError(
                f"cold protocol exited with status {code}; inspect failure_stop.json"
            )
        return report(output)
    finally:
        # Process-group cleanup covers timeout, Ctrl-C and an abruptly dead parent.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except ProcessLookupError:
            pass
        _write_json(
            ledger,
            dict(
                elapsed_s=min(budget_s, used + time.perf_counter() - started),
                budget_s=budget_s,
            ),
        )


def protocol_schedule():
    schedule = []
    for night in NIGHTS:
        schedule += [
            (night, "control", 12, "control"),
            (night, "best", 12, "best"),
        ]
        if night in REFERENCES:
            schedule.append((night, "control", 12, "control_repeat"))
    # 12 workers is the preceding Sep 20 best run; do not repeat it.
    schedule += [("2026-09-20", "best", w, f"t6_{w}") for w in (24, 32)]
    return schedule


# %%
def main(
    action="plan",
    night="2026-09-20",
    modes=("control", "t1", "t1_t2", "best"),
    workers=12,
    out_dir=OUTPUT,
    budget_s=9000,
    max_jobs=None,
    block_size=16384,
):
    """Plan offline, run approved single-night experiments, or regenerate reports.

    action='protocol' runs control/best/control on three reference nights,
    then historic control/best and Sep 20 best at 24/32 workers. The 12-worker
    run is reused as measured evidence, never as a processing cache. Budget
    2.5 hours including setup; the generated driver enforces the hard deadline.
    """
    output = _output(out_dir)
    if action == "plan":
        _write_json(
            output / "plan.json",
            dict(
                network="not attempted by plan",
                nights=NIGHTS,
                schedule=protocol_schedule(),
                budget_s=budget_s,
                setup="local daily TLEs only; no audits, fresh workers and HTTP caches per mode",
                flags=asdict(speedup.Options()),
            ),
        )
        driver = output / "run_protocol.py"
        driver.write_text(
            '"""Spawn-safe, bounded cold benchmark driver; only four SPEC nights."""\n'
            "import sys\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import bench_speedup\n\n"
            'if __name__ == "__main__":\n'
            f"    bench_speedup.run_bounded_protocol(out_dir={str(output / 'cold_r2')!r}, budget_s={budget_s!r})\n"
        )
        return dict(plan=str(output / "plan.json"), driver=str(driver))
    if action == "report":
        return report(output)
    if action not in ("run", "protocol"):
        raise ValueError("action must be plan, run, protocol or report")
    if budget_s <= 0 or (max_jobs is not None and max_jobs < 1):
        raise ValueError("budget and max_jobs must be positive")
    if action == "protocol" and max_jobs:
        raise ValueError("use a separate action='run' output folder for smoke samples")
    ids = tier1_ids(output)
    schedule = (
        protocol_schedule()
        if action == "protocol"
        else [(night, m, workers, m) for m in modes]
    )
    budget_path = output / "protocol_budget.json"
    prior_elapsed = (
        json.loads(budget_path.read_text())["elapsed_s"] if budget_path.exists() else 0
    )
    t0, last_wall, prepared = time.perf_counter(), 0, {}
    completed = []
    for day, mode, count, label in schedule:
        if prior_elapsed + time.perf_counter() - t0 + max(last_wall, 600) > budget_s:
            _write_json(
                output / "budget_stop.json",
                dict(
                    completed=completed,
                    next=[day, mode, count, label],
                    elapsed_s=prior_elapsed + time.perf_counter() - t0,
                ),
            )
            break
        if day not in prepared:
            jobs = _prepare(day, output)
            if max_jobs:
                jobs = jobs[:max_jobs]
            prepared[day] = jobs
        jobs = prepared[day]
        try:
            read_block = (
                int(label.removeprefix("block_"))
                if label.startswith("block_")
                else block_size
            )
            result = run_mode(
                day, jobs, mode, count, output, ids, label, max_jobs, read_block
            )
            last_wall = result["wall_s"]
            completed.append([day, mode, count, label])
        except Exception as exc:
            _write_json(
                output / "failure_stop.json",
                dict(
                    run=[day, mode, count, label],
                    error=repr(exc),
                    completed=completed,
                    elapsed_s=prior_elapsed + time.perf_counter() - t0,
                ),
            )
            raise
        finally:
            _write_json(
                budget_path,
                dict(
                    elapsed_s=prior_elapsed + time.perf_counter() - t0,
                    budget_s=budget_s,
                ),
            )
            report(output)
    return report(output)


# %%
if __name__ == "__main__":
    main()
