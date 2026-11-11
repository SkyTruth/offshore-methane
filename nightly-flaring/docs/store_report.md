# Storage implementation and acceptance report

2026-11-11, this checkout only. No network, real enrichment, publication,
real update run, checkout branch/commit/push, PR or delegated work was performed.
The existing publish-retention test uses only its temporary local bare Git
fixture. The existing CSVs and caches were retained.

## Files changed

- `store.py`: explicit schemas, legacy conversions/readers, atomic zstd exports,
  verified converter, partitioned rebuild/ingest/sync/read, filesystem adapter.
- `pipeline.py`: three large exports use Parquet; processing and caches retain
  their existing logic.
- `build_viewer.py`: both-format discovery/readers, private provenance ordinals,
  `load_tables`, and the default store build path.
- `update.py`: Parquet-aware last-night read and store build path.
- `validate_myc25.py`, `gates_30n.py`, `bench_speedup.py`: shared export readers.
  `backfill.py` needs no change: its validation call now reads either format.
- `requirements.txt`: `pyarrow>=17`.
- `tests/test_store.py`: offline format, ownership, schema, concurrent-process
  locking, independent reference and decoder-fault regression tests.
- `tests/test_granule_fixes.py`: existing output-path expectations use Parquet.
- `tests/test_darknights.py`: existing update/build test mocks `load_tables`.
- `tests/check_store_real.py`: repeatable sequential real-fixture acceptance.
- `docs/explainer/make_figures.py`: copies use the shared segment reader;
  Ruff also formatted this previously unformatted figure script.
- `README.md`, `docs/decisions.md` (D69), and this report: operations, choices,
  measurements and limits.

## Verification

Run from `nightly-flaring/`:

```bash
env -u PROJ_DATA -u GDAL_DATA .venv/bin/python -W ignore -m pytest -q tests
env -u PROJ_DATA -u GDAL_DATA .venv/bin/python -W ignore tests/check_store_real.py
```

Final tails:

```text
356 passed in 18.34s
All checks passed!
12 files already formatted
```

The 291 baseline tests and 65 storage tests pass (35 added in this fix round).
Ruff check and Ruff format check
used `~/miniforge3/envs/methane/bin/python -m ruff` on all 12 touched Python
files, with `PROJ_DATA` and `GDAL_DATA` unset. `git diff --check` is clean.
`git status --short` has no untracked data files; only source/docs/tests and
docs appear. `git check-ignore` confirms the new
segment Parquet files and `outputs/store/` are ignored.

The real comparison executes
`git show 7bc1699:nightly-flaring/build_viewer.py` (pinned in review round 2)
as a separate module and uses that committed loader to read CSVs even after
conversion. Expected frames are pickles under `outputs/store_acceptance/`,
outside the encoder under test. It checks
`assert_frame_equal(check_exact=True)` against `build_viewer.load_tables()`,
every object cell's scalar type, and float bits (including floats in object
columns). Fault-injection tests prove that −0.0 → 0.0, `1` → `1.0` and
`np.nan` → `None` fail. A missing reference commit exits with a clear error.
The reference is `7bc16991feb1c59669746f27e234c1ec6c628fc6`. The figures in this
report are from the round 2 run, when the script read that commit as `HEAD`
and wrote its artifacts under `outputs/store/_acceptance/`.
Values, missing cells, dtypes,
column order, row order and RangeIndex match. The comparison workers are fresh,
sequential processes; no two heavy runs execute together. This fix round waited
15 minutes before starting its first real-fixture run, then force-converted
the timestamp exports and performed a fresh rebuild.

```text
PASS timestamps: all real passes/copies are whole seconds; microsecond storage loses no time
PASS HEAD CSV/store site_nights: 5,139,701 rows; exact values, object scalar types, float bits, dtypes, columns, row order, index
PASS HEAD CSV/store overpasses: 18,751,171 rows; exact values, object scalar types, float bits, dtypes, columns, row order, index
PASS HEAD CSV/store copies: 330,299 rows; exact values, object scalar types, float bits, dtypes, columns, row order, index
PASS rebuild(first 3)+ingest(4) site_nights: 5,139,701 rows
PASS rebuild(first 3)+ingest(4) overpasses: 18,751,171 rows
PASS rebuild(first 3)+ingest(4) copies: 330,299 rows
PASS: exact CSV/store frames and rebuild/ingest equivalence
```

## Sizes

| Segment | Table | CSV bytes | Parquet bytes |
| --- | --- | ---: | ---: |
| v5_backlog_2025 | overpasses | 1,302,154,710 | 133,631,709 |
| v5_backlog_2025 | site_nights | 250,970,589 | 19,095,950 |
| v5_backlog_2025 | copies | 67,192,797 | 21,357,743 |
| v5_backlog_2026 | overpasses | 749,291,608 | 77,222,237 |
| v5_backlog_2026 | site_nights | 151,666,716 | 11,580,508 |
| v5_backlog_2026 | copies | 38,838,660 | 12,994,163 |
| v5_backfill | overpasses | 206,374,011 | 21,408,838 |
| v5_backfill | site_nights | 40,685,762 | 3,155,260 |
| v5_backfill | copies | 10,591,890 | 3,450,435 |
| v5_30n | overpasses | 104,198,826 | 10,998,045 |
| v5_30n | site_nights | 20,895,056 | 1,707,527 |
| v5_30n | copies | 5,444,064 | 1,775,743 |
| All four | overpasses | 2,362,019,155 | 243,260,829 |
| All four | site_nights | 464,218,123 | 35,539,245 |
| All four | copies | 122,067,411 | 39,578,084 |
| Total | | 2,948,304,689 | 318,378,158 |

Segment Parquet uses 89.2% fewer bytes than CSV. Night files contain
136,841,556 bytes for site-nights, 336,147,908 for overpasses and 54,332,761
for copies, 527,322,225 total, excluding the manifest and acceptance artifacts.
Site-nights have 673 partitions; overpasses and copies each have 675, including
boundary passes outside the requested local-night grid. Partition files repeat
schema/dictionary overhead; they are larger in total than segment Parquet.

## Timing and memory

| Operation | Seconds | Peak RSS, MiB |
| --- | ---: | ---: |
| Convert and verify all segments | 46.718 | 10,100.0 |
| Load all three frames with HEAD CSV loader | 37.389 | 9,488.3 |
| SHA-256 fingerprint probe (including stat comparison) | 0.123 | 150.7 |
| Full store rebuild | 59.909 | 10,747.9 |
| load_tables, up-to-date store | 20.232 | 10,220.4 |
| First-three rebuild, fourth ingest, equality checks | 160.570 | 12,446.7 |

Times are single local runs with warm filesystem caches. CSV and store load
measurements include all three returned frames and exclude later equality
checks and reference serialization. Peak RSS is measured independently in each
worker with `resource.getrusage`, converted from macOS bytes to MiB. Store
loading is 1.85 times faster in this run; its peak memory is similar to CSV.

SHA-256 fingerprinting took 0.122 s versus 0.000487 s for size/mtime probes
on the same reader-selected inputs: about 0.122 s added to an up-to-date sync.
This hashes the selected Parquet files, jobs manifests and optional catalog,
without decoding export frames. The measurement is a single warm-cache run.

The incremental worker spent 54.711 s rebuilding the first three segments and
50.012 s ingesting the fourth; its total includes reading/comparing the two
stores. Of 1,933 prior night files, 1,929 kept their original bytes and mtimes.
A rerun writes raw results to `outputs/store_acceptance/report.json`.

In the original implementation round, a rebuild attempt was stopped after date formatting proved slow.
On 100,000 values, pandas' UTC-aware `%Y-%m-%d` formatting took 6.081 s;
formatting after removing the UTC timezone marker took 0.016 s. The final
night calculation converts to UTC first, removes that marker, then applies
longitude and the twelve-hour evening offset. The final measurements above
use this path; the aborted attempt is excluded.

## Schema exceptions and choices

- All four complete fixture segments have matching column sets: overpasses
  33, site-nights 16, copies 40. Tests also cover different column sets.
- Counts, row/column coordinates and assigned-site IDs are nullable int64 on
  disk. CSV blanks make overpass `peak_row`/`peak_col` and copy `assigned_site`
  float64 in pandas; per-frame metadata restores that exact dtype.
- Nullable flags are Arrow boolean and restore object plus `np.nan` when CSV
  inferred that combination. All-empty optional columns retain their physical
  type and restore the CSV-inferred float64 or object dtype.
- `split_component_ids`, `bands`, `fit_flag`, `pair_distance_px`, `pair_dcol`,
  `pair_drow` and `pair_rh_ratio` use struct children `text:string`,
  `number:float64`, `integer:int64`. Numeric/string distinctions survive
  concatenation; pair-chain semicolon strings cannot use one scalar numeric
  type. The real complete exports have string split/band/fit fields and numeric
  pair fields; mixed/empty variants are exercised by tests.
- `time_utc` is a UTC microsecond timestamp with a string companion preserving
  its exact spelling; date32 stores site-night dates. Metadata preserves the
  original column set, order, dtypes and RangeIndex. Unknown fields and values
  that cannot be represented safely raise; negative zero in integer fields
  raises rather than losing its sign bit. Writes use `coerce_timestamps="us"`
  and reject sub-microsecond values. Flags only accept true/false/blank
  booleans; numeric flags (including −0.0, 0.0, 1.0 and 2.0), boolean audit
  values, uint64 IDs and boolean measurements raise errors naming table,
  column and offending value. Boolean-plus-blank concat frames that pandas
  promotes to numeric flags are likewise rejected. All-empty nights round-trip.
  No new companion encoding was added for those unsupported inputs.
- Copies use centroid longitude for their local evening, including unassigned
  copies. Passes use the catalog longitude for that site, else that site's
  longitude in the row's source segment's site-nights, else the UTC date.
  Missing sites remain in the frames, as at HEAD; `build_data` drops them later.
  Fingerprints also cover the optional catalog.
- Stable segment and original-row ordinals reproduce the canonical merged
  order without rewriting order tokens when earlier rows disappear.
- Fingerprints hash the contents of reader-selected exports with SHA-256
  (Parquet if present, otherwise CSV), jobs manifests and the optional site
  catalog. They detect same-size, same-mtime rewrites, resumed rewrites,
  new preferred files, ownership changes and source-list changes.
- Ingest recomputes the existing canonical merge and compares nights. It only
  replaces changed files, including deletions required by snapshot ownership.
  It still reads all exports. This choice keeps ownership and pandas' empty
  frame/dtype behavior in one implementation.
- The completed manifest is removed before materialization and published
  last. A failure leaves no completed manifest, so sync rebuilds instead of
  certifying a partially materialized store. Night files use temp-write/move;
  readers ignore temporary siblings. Local writers wait on an exclusive
  `<root>/.lock` held across the entire rebuild/ingest/sync. Plain readers hold
  shared locks; `load_tables` holds an exclusive lock across sync and all three
  reads. Spawned-process tests pause writes and reads to verify no mixed store
  or mismatched manifest can be returned. Temp names contain PID plus UUID.
  Non-local roots require one writer and readers waiting for its completion.
- The pyarrow floor is 17 to exclude the base environment's 8 and use a modern
  Arrow/pandas/filesystem stack. Acceptance used pyarrow 25, pandas 2.2.3,
  NumPy 1.26 and Python 3.10. The floor itself was not installed or tested.

## Readers retained on CSV

Production segment readers in the website, update, validation, gates and
benchmark planner use the shared helper. Direct CSV reads remain in:

- `store.py`: the legacy fallback, verified converter and unnamed temporary
  export-normalization stream.
- `pipeline.py`: per-granule caches, site catalog and small jobs manifest;
  `build_viewer.py`, `update.py`, `validate_myc25.py`, `gates_30n.py` and
  `bench_speedup.py`: small manifests, validation/reference/catalog inputs.
- `dev_v4.py`, `compare_versions.py`, `report_v5.py`, `verify_qf1.py`:
  historical development-night comparisons and raw-SDR diagnostics, outside
  the production segment reader path. `dev_v5.py` retains diagnostic CSV
  exports and per-granule cache reads.
- `docs/explainer/make_chips.py`: fixed-tag historical CSV reconstruction for
  checked-in scientific figures. `make_figures.py` discovers current segments
  and uses `store.read_segment` for copies, tested with Parquet-only exports.
- `docs/flareout_research/scripts_sol/{r2_evidence,spatial_and_peers,
  r2_primary_check,r2_dependence,r3_raw_audit}.py` and
  `docs/flareout_research/scripts_opus/s00_cache.py`: archived research
  provenance, kept as run (also excluded from Ruff in the repository config).

## Deviations and untested behavior

The spec calls the export entry point `pipeline.run`; this checkout exposes
`pipeline.main`, which was changed. Pipeline export writing uses an unnamed
CSV stream before Parquet to reproduce the old CSV parser exactly. No segment
CSV export is published by that path.
The final normalization step measured +443 MiB peak for the 30-night
overpasses (828,621 rows), versus +2 MiB for HEAD's CSV write. About +5.5 GB
for a 2025-sized export is an extrapolation from that single ratio, not a
measurement. README and D69 record this retained memory cost.

Ingest's read side recomputes the full canonical merge; its write side only
replaces changed night files. The spec constrains the write set, and this
choice preserves all ownership, dtype and order behavior without a second
merge implementation. Absolute segment paths are local source references;
only the store root uses `pyarrow.fs`.

GCS, BigQuery external tables and the minimum pyarrow version are untested.
Non-local writes omit directory markers (covered with a simulated object
filesystem). BigQuery's source URI pattern is
`gs://bucket/prefix/overpasses/*.parquet` with Hive partitioning enabled.
`_segment`, `_row` and `_time_utc_text` are physical provenance columns.
GCS moves are copy plus delete, not atomic; the atomic rename guarantee and
store locks apply to local roots only. No distributed transaction is provided.
Only local filesystem roots (including a `file://` URI) were exercised. No real
pipeline processing, enrichment, full website rebuild or publication was run;
mocked pipeline export tests and exact website-input frame comparisons cover
this storage change. The requested spec file was kept.
