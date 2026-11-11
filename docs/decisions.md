# Decisions

## D01 — Keep the CSV store until Postgres

- **Choice:** Keep the four CSVs as the store until the move to Postgres on GCP;
  shelve interim SQLite on the local branch `sqlite-storage-shelved`.
- **Alternatives:** SQLite with manual CSV import/export/merge; SQLite with
  write-through CSV mirrors; move directly to Postgres.
- **Why:** A database plus a git-tracked CSV mirror needs one side declared the
  source of truth. Keeping CSVs avoids an interim synchronization layer.
- **Evidence:** Four council review rounds are recorded in [reviews.md](reviews.md).
  Its round 4 summary attributes every blocker in rounds 2–4 to database↔CSV
  synchronization (R2-01–R2-03, R2-05–R2-07, R3-01, R4-01–R4-04).
  The shelved `db.py` has 1,150 lines versus 938 for the pre-change `csv_utils.py`.
- **Status:** Accepted; SQLite was not merged. Carry forward the schema's primary
  keys, `windows.structure_id` and `process_runs.window_id` foreign keys, the `''`
  marker in `process_runs.system_index`, and SQLAlchemy Core upserts for Postgres.

## D02 — Store the pixel analysis table as Parquet

- **Choice:** Replace the pixel analysis CSV with
  `data/sgi_vs_sga_pixels.parquet`, using zstd compression.
- **Alternatives:** Keep the analysis table as CSV; put it in the joined store.
- **Why:** Parquet preserves the analysis table's rows and columns while reducing
  its size from 13,112,624 to 2,033,586 bytes.
- **Evidence:** The file contains 190,231 rows with columns `sgi`, `glint_alpha`,
  and `sid`. `tests/test_csv_utils.py` checks the row count and columns, and
  compares it exactly (`check_exact=True`) with the CSV at commit `57cffc3` when
  that Git object is available.
- **Status:** Accepted; `pyarrow` is declared in both environment dependency lists.

## D03 — Reject malformed CSV records before pandas parsing

- **Choice:** `_read_csv` checks field counts with Python's `csv` module and rejects
  NUL bytes, reporting the file name and physical line number. It then calls
  `pd.read_csv(path)` without changing pandas arguments or inferred dtypes.
- **Alternatives:** Warn and skip malformed rows; rely on pandas parsing alone.
- **Why:** Skipping a row before an upsert rewrites the file without that row.
  Pandas can also accept an extra field as an implicit index, shifting columns:
  `x1,x2,1,2,US` under `structure_id,lon,lat,country` loads as `structure_id=x2`.
- **Evidence:** `tests/test_csv_utils.py` checks extra and missing fields, NUL bytes,
  line numbers, pandas equivalence, and unchanged file bytes after a rejected
  `upsert_granule`, `upsert_process_run` and `update_run_metrics`. Lines that are
  empty or hold only spaces and tabs remain skipped, as in pandas; any other
  whitespace is a field. The `csv` module's 131,072-character field limit is
  raised to `2**31 - 1` once at import, since pandas has none; the setting is
  process-wide. A record ending in a bare carriage return is rejected, because
  pandas and the `csv` module tokenise the following record differently. The
  file is read once and both the check and pandas parse those bytes. Blank lines remain skipped; missing files return an empty
  frame; zero-byte files retain pandas' existing `EmptyDataError`.
- **Status:** Accepted; the malformed-line skip fallback is removed.

## D04 — Run integrity validation as a separate command

- **Choice:** Expose read-only `validate_data()` and
  `python -m offshore_methane.csv_utils`; report key, reference, numeric, and
  integer problems with offending row counts and up to five keys per check.
- **Alternatives:** Enforce integrity checks in every loader or upsert; rely on
  ad hoc inspection.
- **Why:** A separate command adds integrity diagnostics without changing pipeline
  load or write behavior beyond the malformed-record rejection in D03.
- **Evidence:** Validation of the repository data on 2026-11-11 returned `[]`:
  0/145 rows in `structures.csv`, 0/3,572 in `windows.csv`, 0/5,097 in
  `granules.csv`, and 0/8,894 in `process_runs.csv` failed checks. The module
  command printed no messages and exited 0. All four CSVs are byte-identical to
  HEAD. Tests use temporary CSVs for invalid fixtures and check the real data
  read-only.
- **Status:** Accepted; missing files are skipped, and validation is not called
  by loaders or upserts. Legacy column names the loaders accept (`event_id`,
  `window_id` in windows, `longitude`/`latitude`) are renamed before the checks.
