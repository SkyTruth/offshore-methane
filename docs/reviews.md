# Review log

Findings from independent reviews of changes in this repository. Each round
lists the reviewer, the verdict, and every finding with its resolution.
Design choices that follow from a finding are recorded in `decisions.md`.

## SQLite storage migration (`offshore_methane/db.py`)

The implementation is on the local branch `sqlite-storage-shelved` and was not merged.

### Round 1 — 2026-11-11

Reviewers: GPT-6.1 Sol (NO-GO, 3 blockers, 1 should-fix) and Claude Opus 5.5
(GO, 2 should-fix, 10 nits). Both ran `pytest -q` (38 passed, 2 `test_mbsp`
failures that predate the change) and confirmed the four CSV snapshots and the
Parquet file carry the data at `57cffc3` (145 / 3,572 / 5,097 / 8,894 rows;
190,231 pixel rows).

| # | Reviewer | Severity | Finding | Resolution |
|---|---|---|---|---|
| R1-01 | Sol | blocker | `import_csv` accepts a CSV row with an extra field: pandas treats the first field as an index and the importer discards it, so `x1,x2,1,2,US` imports as key `x2`. | Validated CSV field counts; `test_import_rejects_wrong_field_count` |
| R1-02 | Sol | blocker | Export → import → export is not byte-identical for 17-significant-digit floats: `pd.to_numeric` parsed `0.12345678901234568` to `0.1234567890123456`. | Parsed floats without precision loss; `test_import_float_roundtrip_17_digits` |
| R1-03 | Sol | blocker | `process_product()` with a site lacking `window_id` writes `window_id=-1`; the new foreign key rejects both run writes and the surrounding `try/except` swallows the errors while exports proceed. Opus traced the orchestrator's own call path and found `window_id` always set there; the failure needs an ad hoc caller. | Validated windows and logged write exceptions; `test_process_product_requires_real_window_before_exports`, `test_process_product_reports_persistence_failures` |
| R1-04 | Sol, Opus | should-fix | On Postgres (`READ COMMITTED`) the four export reads are not one snapshot; an insert between reads can export a child row without its parent. Not run: no Postgres server available. | Applied REPEATABLE READ; `test_export_uses_repeatable_read` |
| R1-05 | Opus | should-fix | `export_csv` from a database bootstrapped before a `git pull` overwrites the pulled CSVs: 3,573 window rows → 3,572 after export. | Guarded snapshot hashes with force override; `test_export_refuses_stale_snapshot_and_force` |
| R1-06 | Opus | nit | Text primary-key order follows database collation on Postgres, so switching backends reorders the CSVs. | Sorted text keys by UTF-8 bytes; `test_export_sorts_text_keys_by_bytes` |
| R1-07 | Opus | nit | Concurrent first use on an empty Postgres database can fail in `create_all`. | Required explicit remote init; `test_non_sqlite_requires_explicit_init` |
| R1-08 | Opus | nit | A crash during bootstrap leaves `.offshore_methane.db-*` temp files in `data/`. Recovery succeeded in all three crash points tested by Sol. | Cleaned stale unlocked bootstrap files; `test_bootstrap_cleans_only_stale_unused_files` |
| R1-09 | Opus | nit | Exported CSVs and the bootstrapped database are mode 0600 (`mkstemp`), previously umask default. | Applied umask on exclusive file creation; `test_snapshot_and_bootstrap_permissions_follow_umask` |
| R1-10 | Opus | nit | `''` written to a text column through an upsert reads back as `''` from the live database and as NULL after export → import. | Normalized blank nullable text; `test_upsert_blank_nullable_text_is_null` |
| R1-11 | Opus | nit | `notebooks/hitl.py` fallback path passes `int(lon)` as `window_id`; the foreign key now rejects it, a bare `except` hides the error, and the status reads "Saved". Before the change the same path wrote a row with a wrong window id. | Reserved optional window id and reported failures; `test_hitl_coordinates_never_become_window_id`, `test_hitl_reports_database_write_failure` |
| R1-12 | Opus | nit | The Postgres compile test builds its own statement instead of going through `db._upsert`. | Compiled production upsert builder; `test_postgres_schema_compiles_without_server` |
| R1-13 | Opus | nit | `tests/test_db_equivalence.py` fails in a shallow clone because it reads the old module with `git show`. | Skipped unavailable baseline; `test_unavailable_baseline_skips` |
| R1-14 | Opus | nit | `decisions.md` refers to the working spec file, which is deleted before commit. | Removed the reference |
| R1-15 | Opus | nit | An existing empty default database is used as-is; the orchestrator reports no windows without naming the file. | Warned on empty default DB; `test_empty_default_database_warns_with_import_command` |
| R1-16 | main session | nit | `upsert_granule` stores full float precision while the snapshot holds 4 decimals, so live reads differ from reads after `import --replace` (`1.23456789` vs `1.2346`). | Rounded granule floats on write; `test_granule_rounds_on_write` |

### Round 2 — 2026-11-11

Reviewers: GPT-6.1 Sol (NO-GO, 5 blockers) and Claude Opus 5.5 (NO-GO, 1
blocker). Both reverted the round 1 fixes one at a time and confirmed the
regression tests fail without them (15/16; R1-14 is a doc edit with no test).
`pytest -q`: 69 passed, 2 `test_mbsp` failures that predate the change. The
snapshots round-trip byte-identically and the Parquet file matches all 190,231
rows.

| # | Reviewer | Severity | Finding | Resolution |
|---|---|---|---|---|
| R2-01 | Opus | blocker | The export guard's error message recommends `import --replace`, which deletes database writes that were never exported: 8,895 → 8,894 runs after a local HITL write plus a pulled `windows.csv` row. `export --force` drops the pulled row instead. | Guarded replacement and merged the union; `test_replace_refuses_unexported_changes_and_merge_keeps_union` |
| R2-02 | Sol | blocker | A snapshot CSV that was absent at bootstrap and appears later has no recorded hash, so export overwrites it: 4 granule rows → 0. | Recorded absent snapshots and refused new files; `test_export_refuses_newly_appeared_snapshot` |
| R2-03 | Sol | blocker | Export does not serialise against other imports/exports or revalidate before replacing files: a row added between hash validation and table capture is lost (4 → 3 windows), and two databases exporting to the same paths leave a snapshot that fails import. | Locked snapshot operations, tracked database identity, and rechecked before replacement; `test_export_refuses_different_database`, `test_export_rechecks_before_any_replace`, `test_concurrent_snapshot_imports_and_exports` |
| R2-04 | Sol, Opus | should-fix | An export interrupted after replacing some files leaves their hashes unrecorded; the retry refuses with a false conflict. | Accepted files already matching intended output on retry; `test_interrupted_export_retry_without_force` |
| R2-05 | Sol | blocker | A text field containing NUL passes field-count validation and is truncated by the pandas parser: `US\x00suffix` imports as `US`. | Used validated CSV records and rejected NUL with line numbers; `test_import_rejects_nul_with_line` |
| R2-06 | Sol | blocker | Integer columns pass through float conversion: `1.0000000000000001` imports as id 1 and `9007199254740993.0` as 9007199254740992. | Parsed signed 64-bit integers exactly; `test_import_rejects_inexact_or_out_of_range_integer`, `test_import_preserves_exact_large_integer`, `test_export_nullable_integer_preserves_large_values` |
| R2-07 | Sol | blocker | `upsert_granule` accepts `system_index=""`; the exported snapshot then fails `import --replace` with a NULL key. | Rejected missing or blank granule keys; `test_upsert_granule_rejects_blank_key` |
| R2-08 | Opus | should-fix | Snapshot hashes describe one checkout but are stored in the database, so on a shared Postgres URL one user's export makes another user's export refuse and point at `import --replace`. Read from code; no Postgres server available. | Kept sync state per checkout; `test_checkouts_share_database_with_independent_state` |
| R2-09 | Opus | nit | The field-count check rejects blank lines, which the old loader skipped; a zero-byte CSV raises `EmptyDataError` at bootstrap. | Skipped blank lines and rejected zero-byte CSVs with a header error; `test_import_skips_blank_lines`, `test_import_rejects_zero_byte_file` |
| R2-10 | Sol | nit | `decisions.md` cites a regression test for R1-14 that the main session removed. | Removed stale citation and checked test references; `test_decision_test_references_exist` |

### Round 3 — 2026-11-11

Reviewers: GPT-6.1 Sol (GO, no findings) and Claude Opus 5.5 (NO-GO, 1
blocker). Both reverted the round 2 fixes in copies of the repo and confirmed
the regression tests fail without them (15 mutations each). `pytest -q`: 101
passed, 2 `test_mbsp` failures that predate the change.

Loss of data between the database and the CSV snapshots was found in each of
the three rounds in a different form (R1-05, R2-01, R3-01). Round 4 replaces
the manual export/import/merge workflow with write-through snapshots; see
`decisions.md`.

| # | Reviewer | Severity | Finding | Resolution |
|---|---|---|---|---|
| R3-01 | Opus | blocker | `import --merge` applies every non-null CSV cell, so local unexported updates to rows the teammate never touched revert to the last-synced values: `sgi_median` 0.9 → 0.5 and `hitl_value` 0 → 1 after the documented pull → `import --merge` → `export`. | Local writes publish before commit; pulls retain the local update plus unrelated rows; `test_local_update_survives_pulled_unrelated_rows` |
| R3-02 | Opus | should-fix | Merge cannot propagate upstream deletions; the following export writes the deleted rows back. With an unexported local write, both `export` and `import --replace` refuse, so only the `--force` variants keep the deletion. | Reloaded complete validated CSVs on the next call, including deletions; `test_csv_deletion_propagates_on_next_read` |
| R3-03 | Opus | should-fix | Database identity is the URL string. A database recreated at the same explicit URL passes the export guard and overwrites the CSVs with header-only files: `windows.csv` 3,573 lines → 1. | Remote export refuses an empty database when CSVs contain rows; no database identity or remote state; `test_remote_export_refuses_empty_database_with_populated_csvs` |
| R3-04 | Opus | nit | `import --replace` on a recreated empty default database refuses with "unexported changes". | Local replacement bypasses sync/bootstrap and needs no force, including a missing database file; `test_replace_after_default_database_deleted` |

### Round 4 — 2026-11-11

Reviewers: GPT-6.1 Sol (NO-GO, 3 blockers) and Claude Opus 5.5 (NO-GO, 1
blocker). Both confirmed R3-01 … R3-04 closed by reverting the fixes. `pytest
-q`: 112 passed, 2 `test_mbsp` failures that predate the change. Concurrency
checks passed for both reviewers (8 processes × 8 threads; 32 threads plus 4
writer processes: 0 errors, 0 spurious reloads). Seven crash points left the
CSVs equal to the database.

| # | Reviewer | Severity | Finding | Resolution |
|---|---|---|---|---|
| R4-01 | Opus | blocker | `AGENTS.md` tells contributors to run `export` before committing, and local `export` skips the sync check, so pull → export → commit removes the pulled rows: a teammate's granule row went 1 → 0 on origin in a two-clone reproduction. Export also overwrites a CSV holding merge markers. | Not fixed; implementation shelved |
| R4-02 | Sol | blocker | `import --replace` after deleting the default database publishes an empty database before importing; a crash at that point leaves the old state file, the fast path accepts the empty database, and the next granule write replaces 4 rows with 1. | Not fixed; implementation shelved |
| R4-03 | Sol | blocker | Crash after COMMIT and before the state write, then `git restore` of the CSV: file equals recorded state, the rendering is never compared, and the next write puts the reverted values back (`0.5, 1` → `0.9, 0`). | Not fixed; implementation shelved |
| R4-04 | Sol | blocker | `load_events`, `unified_db` and `virtual_db` read their tables through separate connections; a checkout plus `import --replace` between the reads returns a window with `lon`/`lat` NaN. | Not fixed; implementation shelved |
| R4-05 | Opus | should-fix | The engine cache is keyed on the path. A kernel holding an engine to a deleted database file keeps writing its stale view into the CSV after another process rebuilds the database. | Not fixed; implementation shelved |
| R4-06 | Sol, Opus | should-fix | An empty, truncated or old-format state file makes every call raise `JSONDecodeError`/`AttributeError` with no recovery text. | Not fixed; implementation shelved |
| R4-07 | Opus | should-fix | `import fcntl` at module top fails on Windows; the old module imported it lazily with a fallback. Two notebooks use Windows paths. | Not fixed; implementation shelved |
| R4-08 | Opus | should-fix | Reload rewrites a checked-out CSV into canonical order, which dirties the tree: `structures.csv` at `57cffc3` is in numeric order, and checkout back is refused. | Not fixed; implementation shelved |
| R4-09 | Opus | nit | `test_replace_after_default_database_deleted` passes when the sync bypass for `--replace` is reverted. | Not fixed; implementation shelved |

Status after round 4: SQLite was shelved (`decisions.md` D01). Every blocker in rounds 2–4 lies in
keeping the database and the git-tracked CSVs consistent; none lies in the
schema, the upserts, the read API, or concurrency.

## Parquet pixel table and strict CSV validation (`offshore_methane/csv_utils.py`)

### Round 1 — 2026-11-11

Reviewers: GPT-6.1 Sol (NO-GO, 2 blockers, 1 should-fix) and Claude Opus 5.5
(GO, 3 should-fix, 7 nits). `pytest -q`: 53 passed, 2 `test_mbsp` failures that
predate the change. Both confirmed `_read_csv` equals `pd.read_csv` on the four
store CSVs (145 / 3,572 / 5,097 / 8,894 rows), `validate_data()` returns `[]`
on them, and the Parquet file equals the CSV at `57cffc3` (190,231 × 3). Opus
ran 6 upsert cases against malformed files: all 6 raise and leave the bytes
unchanged; at `57cffc3` all 6 rewrote the file.

| # | Reviewer | Severity | Finding | Resolution |
|---|---|---|---|---|
| V1-01 | Sol, Opus | blocker (Sol), nit (Opus) | The `csv` module's 131,072-character field limit rejects a file the writer produced: a 131,073-character field written by an upsert fails the next read. | Limit raised to `sys.maxsize`; `test_read_csv_accepts_field_longer_than_csv_module_limit` |
| V1-02 | Sol | blocker | `str.strip()` treated a row holding only a non-breaking space as blank, while pandas reads it as a one-field row; the upsert then rewrote the file. | Only space/tab-only lines count as blank, as in pandas; `test_non_space_whitespace_row_is_a_short_row`, `test_space_or_tab_row_is_blank_as_in_pandas` |
| V1-03 | Sol, Opus | should-fix | `validate_data` reported missing columns for legacy names the loaders accept (`event_id`, `window_id` in windows, `longitude`/`latitude`). | Same renames applied before the checks; `test_validate_data_accepts_legacy_column_names` |
| V1-04 | Opus | should-fix | The Parquet test compared with `rtol=1e-5`; rounding every float to 6 significant digits still passed. After the commit `HEAD` no longer holds the CSV, so the comparison would always skip. | `check_exact=True` and the baseline pinned to `57cffc3`; `test_pixel_parquet_matches_original_csv` |
| V1-05 | Opus | should-fix | A hand-edited row that omits trailing empty fields loaded before and now raises (`windows.csv: line 3: expected 5 fields, got 4`). Intended; undocumented. | README line under the validation subsection |
| V1-06 | Opus | nit | The `structure_id` reference check compared raw dtypes; the loader joins as strings. | Compared as strings; `test_validate_data_compares_structure_ids_as_strings` |
| V1-07 | Opus | nit | Duplicate keys repeated in the example list. | De-duplicated; `test_validate_data_lists_each_duplicate_key_once` |
| V1-08 | Opus | nit | `main()` raised `PermissionError` on an unreadable file. | `OSError` reported as a message |
| V1-09 | Opus | nit | No test that `upsert_process_run` and `update_run_metrics` leave a malformed file unchanged, and none comparing `_read_csv` with pandas on the real files. | `test_run_writers_do_not_rewrite_malformed_file`, `test_real_store_csvs_match_pandas` |
| V1-10 | Opus | nit | The integer check goes through float64, so `1.0000000000000001` passes as an integer. | Not fixed; the pipeline does not write such values |
| V1-11 | Opus | nit | 46,160/190,231 `sgi` values in the Parquet file differ by 1 ULP from a round-trip parse of the CSV text; they equal the default `pd.read_csv` output earlier readers received. | No change |

### Round 2 — 2026-11-11

Reviewers: GPT-6.1 Sol (GO, 4 should-fix) and Claude Opus 5.5 (GO, 4
should-fix, 3 nits). Both reverted the round 1 fixes and confirmed the
regression tests fail without them. `pytest -q`: 65 passed, 2 `test_mbsp`
failures that predate the change. Opus fuzzed `_atomic_write_csv` with 20,000
random frames: 19,998 read identically through `_read_csv` and `pd.read_csv`,
and 2 raised `EmptyDataError` in both. Against a port of the pandas tokenizer,
30,000 random inputs without bare carriage returns gave 0 divergences; with
bare carriage returns, 102 malformed files were accepted and 62 valid files
rejected (V2-01). The fixes below were applied after the verdicts; the suite
then gave 73 passed.

| # | Reviewer | Severity | Finding | Resolution |
|---|---|---|---|---|
| V2-01 | Sol, Opus | should-fix | A blank line ending in a bare `\r` followed by a row starting with a comma passes the width check while pandas reads the row one field short; an upsert then rewrites it shifted. Needs a hand-edited or legacy Mac file. | A record ending in a bare `\r` is rejected; `\r` inside a quoted field is accepted; `test_bare_carriage_return_row_is_rejected_before_rewrite`, `test_carriage_return_inside_quoted_field_is_accepted` |
| V2-02 | Sol, Opus | should-fix | `validate_data` raised `TypeError` when a file holds both a legacy and a current column name (`window_id,event_id`), which the loaders accept. | Legacy name used only when the current one is absent; leftover dropped; `test_validate_data_ignores_legacy_column_next_to_current` |
| V2-03 | Opus | should-fix | `csv.field_size_limit(sys.maxsize)` overflows a C long on Windows with Python 3.10–3.12. Not run on Windows. | Capped at `2**31 - 1`; `test_field_limit_fits_a_32_bit_c_long` |
| V2-04 | Sol | should-fix | Each read changed the process-wide `csv` field limit. | Set once at import; the limit stays process-wide (`decisions.md` D03) |
| V2-05 | Opus | should-fix | `csv_migrate.ensure_windows_header()` wrote a 6-field header onto a 5-column file, which the strict reader rejects. | Header sized to the first row; `test_ensure_windows_header_matches_row_width` |
| V2-06 | Sol | should-fix | A permission error from `path.is_file()` escaped `validate_data`. | Caught and reported as `read failed`; `test_validate_data_reports_unreadable_directory` |
| V2-07 | Opus | nit | The file was opened once for the check and again for the parse. | Bytes read once and parsed from memory |
| V2-08 | Opus | nit | Every read failure was labelled "CSV parse failed: 1 offending row". | Message is `<file>: read failed: <error>` |
| V2-09 | Opus | should-fix | `validate_data` reports missing columns for header spellings `load_events` accepts through lower-casing and token aliases (`Lon`, `longitude_dd`). | Not fixed; the store files use the current names |
| V2-10 | Opus | nit | `orchestrator.py:398-405` wraps `upsert_process_run` in `except Exception: pass`, so a malformed `process_runs.csv` now skips the run-metadata write without a message; at `57cffc3` it warned and rewrote the file without the bad rows. | Not fixed in this change |
| V2-11 | Opus | nit | `upsert_process_run(w, "")` called twice appends two marker rows, because the loaded `<NA>` never equals `""`. Exists at `57cffc3`; `validate_data` reports the duplicate. | Not fixed in this change |
