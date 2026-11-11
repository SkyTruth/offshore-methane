"""Offline real-fixture acceptance; run from nightly-flaring with its venv.

    env -u PROJ_DATA -u GDAL_DATA .venv/bin/python -W ignore tests/check_store_real.py

Workers run sequentially so the CSV path, store path and ingest have independent
peak-memory measurements. Parquet exports, outputs/store/ and outputs/store_acceptance/ are written.
The CSV reference needs commit REFERENCE in the local git history.
"""

from __future__ import annotations

import gc
import json
import resource
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_viewer as viewer  # noqa: E402
import store  # noqa: E402

BASE = viewer.OUT_DIR
ROOT = BASE / "store"
ARTIFACTS = BASE / "store_acceptance"
# Last commit whose build_viewer reads the segment CSVs directly. A moving
# HEAD would make the reference share the decoder under test after this change
# is committed.
REFERENCE = "7bc16991feb1c59669746f27e234c1ec6c628fc6"
TABLES = ("site_nights", "overpasses", "copies")


def _report(action, started, **details):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(
        action=action,
        seconds=round(time.perf_counter() - started, 3),
        peak_mib=round(peak / (1024**2 if sys.platform == "darwin" else 1024), 1),
        **details,
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{action}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def _equal(left, right):
    assert_frame_equal(left, right, check_exact=True)
    for col in left.select_dtypes("float64"):
        assert np.array_equal(
            left[col].to_numpy().view("uint64"),
            right[col].to_numpy().view("uint64"),
        ), f"float bits differ: {col}"
    for col in left.select_dtypes("object"):
        # Bound temporary arrays while checking every scalar, including blanks.
        for start in range(0, len(left), 100_000):
            a = left[col].iloc[start : start + 100_000].to_numpy()
            b = right[col].iloc[start : start + 100_000].to_numpy()
            kinds = np.frompyfunc(type, 1, 1)(a)
            assert np.array_equal(kinds, np.frompyfunc(type, 1, 1)(b)), (
                f"object scalar types differ: {col}, rows {start}:{start + len(a)}"
            )
            floats = np.fromiter(
                (issubclass(kind, (float, np.floating)) for kind in kinds),
                dtype=bool,
                count=len(kinds),
            )
            assert np.array_equal(
                a[floats].astype("float64").view("uint64"),
                b[floats].astype("float64").view("uint64"),
            ), f"object float bits differ: {col}"


def _head_loader():
    """Load the committed CSV implementation independently of the working tree."""
    result = subprocess.run(
        ["git", "show", f"{REFERENCE}:nightly-flaring/build_viewer.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SystemExit(
            f"HEAD-era reference {REFERENCE[:7]}:nightly-flaring/build_viewer.py is unavailable; cannot build an independent CSV reference"
        )
    head = types.ModuleType("_head_build_viewer")
    head.__file__ = viewer.__file__
    sys.modules[head.__name__] = head
    exec(
        compile(result.stdout, f"{REFERENCE}:nightly-flaring/build_viewer.py", "exec"),
        head.__dict__,
    )
    return head


def main(action="all"):
    segments = viewer.find_segments(BASE)
    started = time.perf_counter()
    if action == "all":
        _head_loader()  # Fail before starting any heavy work if HEAD is absent.
        for stage in (
            "convert",
            "csv",
            "fingerprint",
            "rebuild",
            "store",
            "incremental",
        ):
            # A fresh process isolates ru_maxrss and releases each heavy load.
            subprocess.run(
                [
                    sys.executable,
                    "-W",
                    "ignore",
                    "-u",
                    "-c",
                    "import sys; sys.path.insert(0, 'tests'); "
                    f"import check_store_real as c; c.main(action={stage!r})",
                ],
                check=True,
            )
        report = {
            stage: json.loads((ARTIFACTS / f"{stage}.json").read_text())
            for stage in (
                "convert",
                "csv",
                "fingerprint",
                "rebuild",
                "store",
                "incremental",
            )
        }
        (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2))
        print("PASS: exact CSV/store frames and rebuild/ingest equivalence", flush=True)
        return report
    if action == "convert":
        # Never forced: an existing Parquet export may be newer than its CSV.
        sizes = [size for segment in segments for size in store.convert(segment)]
        for folder, tag in segments:
            for table in ("overpasses", "copies"):
                path = folder / f"{table}_{tag}.parquet"
                with pq.ParquetFile(path) as file:
                    times = file.read(columns=["time_utc"])["time_utc"]
                    assert times.type == store.STAMP
                    assert all(
                        (chunk.cast(pa.int64()).to_numpy() % 1_000_000 == 0).all()
                        for chunk in times.chunks
                    )
        print(
            "PASS timestamps: all real passes/copies are whole seconds; microsecond storage loses no time",
            flush=True,
        )
        return _report(action, started, sizes=sizes)
    if action == "csv":
        head = _head_loader()
        segments = head.find_segments(BASE)
        nights, over = head.load_segments(segments)
        copies = head.load_copies(segments)
        result = _report(
            action,
            started,
            rows={t: len(f) for t, f in zip(TABLES, (nights, over, copies))},
        )
        for table, frame in zip(TABLES, (nights, over, copies)):
            frame.to_pickle(ARTIFACTS / f"expected-{table}.pkl")
        return result
    if action == "fingerprint":
        # Compare the checksum cost with the former size/mtime probe on the
        # same selected inputs; sync's no-op path runs this checksum once.
        for folder, tag in segments:
            paths = [store.segment_path(folder, tag, table) for table in store.SCHEMAS]
            paths += [folder / f"jobs_{tag}.csv", folder.parent / "gfw_oil_sites.csv"]
            for path in paths:
                if path.exists():
                    path.stat()
        stat_seconds = time.perf_counter() - started
        hashing = time.perf_counter()
        for segment in segments:
            store._fingerprint(segment)
        hash_seconds = time.perf_counter() - hashing
        return _report(
            action,
            started,
            stat_seconds=round(stat_seconds, 6),
            hash_seconds=round(hash_seconds, 3),
            added_seconds=round(hash_seconds - stat_seconds, 3),
        )
    if action == "rebuild":
        store.rebuild(segments, ROOT)
        sizes = {
            table: sum(
                p.stat().st_size for p in (ROOT / table).glob("night=*/part.parquet")
            )
            for table in TABLES
        }
        return _report(action, started, bytes=sizes)
    if action == "store":
        frames = viewer.load_tables(BASE, ROOT)
        result = _report(
            action, started, rows={t: len(f) for t, f in zip(TABLES, frames)}
        )
        for table, frame in zip(TABLES, frames):
            expected = pd.read_pickle(ARTIFACTS / f"expected-{table}.pkl")
            _equal(frame, expected)
            print(
                f"PASS HEAD CSV/store {table}: {len(frame):,} rows; exact values, object scalar types, float bits, dtypes, columns, row order, index",
                flush=True,
            )
            del expected
            gc.collect()
        return result
    if action == "incremental":
        incremental = ARTIFACTS / "incremental"
        store.rebuild(segments[:3], incremental)
        before = {
            p: p.stat().st_mtime_ns
            for t in TABLES
            for p in (incremental / t).glob("night=*/part.parquet")
        }
        rebuilt = time.perf_counter()
        store.ingest(segments[3], incremental)
        ingested = time.perf_counter()
        unchanged = sum(
            p.exists() and p.stat().st_mtime_ns == mtime for p, mtime in before.items()
        )
        for table in TABLES:
            left = store.read(table, incremental)
            right = store.read(table, ROOT)
            _equal(left, right)
            print(
                f"PASS rebuild(first 3)+ingest(4) {table}: {len(left):,} rows",
                flush=True,
            )
            del left, right
            gc.collect()
        return _report(
            action,
            started,
            rebuild_seconds=round(rebuilt - started, 3),
            ingest_seconds=round(ingested - rebuilt, 3),
            unchanged_prior_files=unchanged,
            prior_files=len(before),
        )
    raise ValueError(action)


if __name__ == "__main__":
    main()
