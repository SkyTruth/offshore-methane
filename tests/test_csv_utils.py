import io
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from offshore_methane import csv_utils

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    "structures": "structure_id,lon,lat\nx1,1,2\nx2,3,4\n",
    "windows": "id,structure_id\n1,x1\n2,x2\n",
    "granules": "system_index,sga_scene,cloudiness\ng1,10,20\ng2,,\n",
    "process_runs": (
        "window_id,system_index,sga_local_median,sgi_median,"
        "valid_pixel_c,valid_pixel_mbsp,hitl_value\n"
        "1,g1,10,0.2,0.8,0.9,1\n2,,,,,,\n"
    ),
}


@pytest.fixture
def data_files(tmp_path, monkeypatch):
    monkeypatch.setattr(csv_utils, "DATA_DIR", tmp_path)
    monkeypatch.setattr(csv_utils, "EVENTS_CSV", tmp_path / "events.csv")
    paths = {}
    for name, text in FIXTURES.items():
        path = tmp_path / f"{name}.csv"
        path.write_text(text)
        monkeypatch.setattr(csv_utils, f"{name.upper()}_CSV", path)
        paths[name] = path
    return paths


@pytest.mark.parametrize("prefix", ["", "\ufeff"])
def test_read_csv_matches_pandas(tmp_path, prefix):
    path = tmp_path / "valid.csv"
    path.write_text(
        prefix + 'key,number,fraction,note\r\nx1,1,0.12345678901234568,"a,b"\r\n'
        'x2,2,,"two\nlines"\r\nx3,3,1.5,""\r\n',
        encoding="utf-8",
    )
    pd.testing.assert_frame_equal(csv_utils._read_csv(path), pd.read_csv(path))


@pytest.mark.parametrize(
    "text,detail",
    [
        ("a,b\n1,2,3\n", "line 2: expected 2 fields, got 3"),
        ("a,b\n1,2\n3,4,5\n", "line 3: expected 2 fields, got 3"),
        ("a,b\n1\n", "line 2: expected 2 fields, got 1"),
        ('a,b\n""\n', "line 2: expected 2 fields, got 1"),
        ('a,b\n"two\nlines",2,3\n', "line 2: expected 2 fields, got 3"),
        ("a,b\n1,US\x00suffix\n", "line 2: NUL byte"),
        ("a,b\x00\n1,2\n", "line 1: NUL byte"),
        ('a,b\n1,"two\nlines\x00"\n', "line 3: NUL byte"),
    ],
)
def test_read_csv_rejects_malformed_rows(tmp_path, text, detail):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=re.escape(f"bad.csv: {detail}")):
        csv_utils._read_csv(path)


def test_read_csv_skips_blank_lines(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("\n \t\na,b\n1,2\n\n \t\n3,4\n\n")
    result = csv_utils._read_csv(path)
    pd.testing.assert_frame_equal(result, pd.read_csv(path))
    assert len(result) == 2


def test_read_csv_missing_file(tmp_path):
    pd.testing.assert_frame_equal(
        csv_utils._read_csv(tmp_path / "missing.csv"), pd.DataFrame()
    )


def test_read_csv_zero_byte_preserves_empty_data_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.touch()
    with pytest.raises(pd.errors.EmptyDataError):
        csv_utils._read_csv(path)


def test_upsert_granule_does_not_rewrite_malformed_file(data_files):
    path = data_files["granules"]
    path.write_text(FIXTURES["granules"] + "g3,30,40,extra\n")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="granules.csv: line 4"):
        csv_utils.upsert_granule({"system_index": "g1", "cloudiness": 5})
    assert path.read_bytes() == original


def test_validate_data_clean_and_read_only(data_files):
    before = {path: path.read_bytes() for path in data_files.values()}
    assert csv_utils.validate_data() == []
    assert {path: path.read_bytes() for path in data_files.values()} == before
    assert set(next(iter(data_files.values())).parent.iterdir()) == set(before)


@pytest.mark.parametrize(
    "table,column,value,check,key",
    [
        ("structures", "duplicate", 0, "structure_id unique", "'x1'"),
        ("structures", "append_null", "structure_id", "structure_id non-null", "''"),
        ("structures", "lon", "bad", "lon numeric and non-null", "'x1'"),
        ("structures", "lon", None, "lon numeric and non-null", "'x1'"),
        ("structures", "lat", "bad", "lat numeric and non-null", "'x1'"),
        ("structures", "lat", None, "lat numeric and non-null", "'x1'"),
        ("windows", "duplicate", 0, "id unique", "1"),
        ("windows", "append_null", "id", "id integer", "''"),
        ("windows", "id", "bad", "id integer", "'bad'"),
        ("windows", "id", 1.5, "id integer", "1.5"),
        ("windows", "structure_id", None, "structure_id non-null", "1"),
        ("windows", "structure_id", "missing", "present in structures.csv", "1"),
        ("granules", "duplicate", 0, "system_index unique", "'g1'"),
        ("granules", "system_index", None, "system_index non-null", "''"),
        ("granules", "sga_scene", "bad", "sga_scene numeric where present", "'g1'"),
        ("granules", "cloudiness", "bad", "cloudiness numeric where present", "'g1'"),
        ("process_runs", "duplicate", 0, "window_id, system_index unique", "(1, 'g1')"),
        ("process_runs", "duplicate", 1, "window_id, system_index unique", "(2, '')"),
        ("process_runs", "window_id", "bad", "window_id integer", "('bad', 'g1')"),
        ("process_runs", "window_id", None, "window_id integer", "('', 'g1')"),
        ("process_runs", "window_id", 1.5, "window_id integer", "(1.5, 'g1')"),
        ("process_runs", "window_id", 999, "present in windows.csv", "(999, 'g1')"),
        ("process_runs", "sga_local_median", "bad", "sga_local_median numeric", "'g1'"),
        ("process_runs", "sgi_median", "bad", "sgi_median numeric", "'g1'"),
        ("process_runs", "valid_pixel_c", "bad", "valid_pixel_c numeric", "'g1'"),
        ("process_runs", "valid_pixel_mbsp", "bad", "valid_pixel_mbsp numeric", "'g1'"),
        ("process_runs", "hitl_value", "bad", "hitl_value integer", "'g1'"),
        ("process_runs", "hitl_value", 1.5, "hitl_value integer", "'g1'"),
    ],
)
def test_validate_data_reports_each_check(data_files, table, column, value, check, key):
    path = data_files[table]
    frame = pd.read_csv(path).astype(object)
    if column == "duplicate":
        frame = pd.concat([frame, frame.iloc[[value]]], ignore_index=True)
    elif column == "append_null":
        row = frame.iloc[[0]].copy()
        row[value] = None
        frame = pd.concat([frame, row], ignore_index=True)
    elif table == "windows" and column == "id":
        row = frame.iloc[[0]].copy()
        row[column] = value
        frame = pd.concat([frame, row], ignore_index=True)
    else:
        frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    before = {path: path.read_bytes() for path in data_files.values()}
    problems = csv_utils.validate_data()
    assert len(problems) == 1, problems
    message = problems[0]
    assert message.startswith(f"{table}.csv:")
    assert check in message
    count = 2 if column == "duplicate" else 1
    assert f"{count}/{len(frame)} rows" in message
    assert key in message.split("keys: ")[1]
    assert {path: path.read_bytes() for path in data_files.values()} == before


def test_validate_data_limits_examples_to_five_keys(data_files):
    path = data_files["granules"]
    path.write_text(
        "system_index,sga_scene,cloudiness\n"
        + "".join(f"g{i},bad,\n" for i in range(7))
    )
    problems = csv_utils.validate_data()
    assert len(problems) == 1
    assert "7/7 rows" in problems[0]
    assert "keys: ['g0', 'g1', 'g2', 'g3', 'g4']" in problems[0]


def test_validate_data_skips_missing_files(data_files):
    data_files["structures"].unlink()
    assert csv_utils.validate_data() == []
    for path in data_files.values():
        path.unlink(missing_ok=True)
    assert csv_utils.validate_data() == []


def test_validate_data_reports_parse_failure(data_files):
    data_files["granules"].write_text(FIXTURES["granules"] + "g3,30,40,extra\n")
    problems = csv_utils.validate_data()
    assert len(problems) == 1
    assert problems[0].startswith("granules.csv: read failed: ")
    assert "line 4: expected 3 fields, got 4" in problems[0]


def test_validate_repository_data_is_clean_and_read_only():
    paths = [ROOT / "data" / f"{name}.csv" for name in FIXTURES]
    before = {path: path.read_bytes() for path in paths}
    assert csv_utils.validate_data() == []
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize("invalid", [False, True])
def test_main_return_codes_and_output(data_files, capsys, invalid):
    if invalid:
        path = data_files["granules"]
        path.write_text(FIXTURES["granules"].replace("g1,10", "g1,bad"))
    expected = csv_utils.validate_data()
    assert csv_utils.main() == int(invalid)
    assert capsys.readouterr().out == "".join(f"{message}\n" for message in expected)


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "offshore_methane.csv_utils"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_pixel_parquet_row_count_and_columns():
    frame = pd.read_parquet(ROOT / "data/sgi_vs_sga_pixels.parquet")
    assert len(frame) == 190_231
    assert frame.columns.tolist() == ["sgi", "glint_alpha", "sid"]


def test_pixel_parquet_matches_original_csv():
    try:
        original = subprocess.run(
            ["git", "show", "57cffc3:data/sgi_vs_sga_pixels.csv"],
            cwd=ROOT,
            capture_output=True,
        )
    except OSError:
        pytest.skip("Git is unavailable for the original pixel CSV comparison")
    if original.returncode:
        pytest.skip("commit 57cffc3 with the original pixel CSV is unavailable")
    pd.testing.assert_frame_equal(
        pd.read_parquet(ROOT / "data/sgi_vs_sga_pixels.parquet"),
        pd.read_csv(io.BytesIO(original.stdout)),
        check_exact=True,
    )


def test_read_csv_accepts_field_longer_than_csv_module_limit(data_files):
    path = data_files["granules"]
    csv_utils.upsert_granule({"system_index": "big", "timestamp": "x" * 131_073})
    pd.testing.assert_frame_equal(csv_utils._read_csv(path), pd.read_csv(path))
    csv_utils.upsert_granule({"system_index": "g1", "cloudiness": 5})
    assert len(csv_utils.load_granules()) == 3


@pytest.mark.parametrize("blank", ["\u00a0", "\f", "\v"])
def test_non_space_whitespace_row_is_a_short_row(data_files, blank):
    path = data_files["granules"]
    path.write_text(FIXTURES["granules"] + blank + "\n")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="granules.csv: line 4: expected 3 fields"):
        csv_utils.upsert_granule({"system_index": "g1", "cloudiness": 5})
    assert path.read_bytes() == original


@pytest.mark.parametrize("blank", [" ", "\t", " \t "])
def test_space_or_tab_row_is_blank_as_in_pandas(tmp_path, blank):
    path = tmp_path / "blank.csv"
    path.write_text("a,b\n1,2\n" + blank + "\n3,4\n")
    pd.testing.assert_frame_equal(csv_utils._read_csv(path), pd.read_csv(path))


def test_run_writers_do_not_rewrite_malformed_file(data_files):
    path = data_files["process_runs"]
    path.write_text(FIXTURES["process_runs"] + "3,g3\n")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="process_runs.csv: line 4"):
        csv_utils.upsert_process_run(1, "g1", git_hash="h")
    with pytest.raises(ValueError, match="process_runs.csv: line 4"):
        csv_utils.update_run_metrics(1, "g1", hitl_value=0)
    assert path.read_bytes() == original


def test_validate_data_accepts_legacy_column_names(data_files):
    for name, old, new in (
        ("process_runs", "window_id", "event_id"),
        ("windows", "id,", "window_id,"),
        ("structures", "lon,lat", "longitude,latitude"),
    ):
        text = data_files[name].read_text()
        data_files[name].write_text(text.replace(old, new, 1))
    assert csv_utils.load_process_runs()["window_id"].tolist() == [1, 2]
    assert csv_utils.validate_data() == []


def test_validate_data_compares_structure_ids_as_strings(data_files):
    data_files["structures"].write_text("structure_id,lon,lat\nx1,1,2\n123,3,4\n")
    data_files["windows"].write_text("id,structure_id\n1,123\n2,123\n")
    assert csv_utils.validate_data() == []


def test_validate_data_lists_each_duplicate_key_once(data_files):
    data_files["granules"].write_text(
        "system_index,sga_scene,cloudiness\n" + "g1,1,1\n" * 3 + "g2,1,1\n" * 2
    )
    (message,) = csv_utils.validate_data()
    assert "5/5 rows" in message and message.endswith("keys: ['g1', 'g2']")


def test_real_store_csvs_match_pandas():
    for name in ("structures", "windows", "granules", "process_runs"):
        path = ROOT / "data" / f"{name}.csv"
        pd.testing.assert_frame_equal(csv_utils._read_csv(path), pd.read_csv(path))


def test_bare_carriage_return_row_is_rejected_before_rewrite(data_files):
    path = data_files["granules"]
    path.write_text(FIXTURES["granules"] + "\r,g3,3\n", newline="")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="granules.csv: line 4: bare carriage"):
        csv_utils.upsert_granule({"system_index": "g1", "cloudiness": 5})
    assert path.read_bytes() == original


def test_carriage_return_inside_quoted_field_is_accepted(tmp_path):
    path = tmp_path / "quoted.csv"
    pd.DataFrame({"a": ["x\ry", "z"], "b": [1, 2]}).to_csv(path, index=False)
    pd.testing.assert_frame_equal(csv_utils._read_csv(path), pd.read_csv(path))


def test_field_limit_fits_a_32_bit_c_long():
    import csv

    assert csv.field_size_limit() == min(sys.maxsize, 2**31 - 1)


@pytest.mark.parametrize(
    "name, header, row",
    [
        ("process_runs", "window_id,event_id,system_index", "1,1,g1"),
        ("windows", "id,window_id,structure_id", "1,1,x1\n2,2,x2"),
        (
            "structures",
            "structure_id,lon,lat,longitude,latitude",
            "x1,1,2,1,2\nx2,3,4,3,4",
        ),
    ],
)
def test_validate_data_ignores_legacy_column_next_to_current(
    data_files, name, header, row
):
    data_files[name].write_text(f"{header}\n{row}\n")
    assert csv_utils.validate_data() == []


def test_validate_data_reports_unreadable_directory(data_files, monkeypatch):
    def denied(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(type(data_files["granules"]), "is_file", denied)
    messages = csv_utils.validate_data()
    assert len(messages) == 4 and all("read failed" in m for m in messages)


def test_ensure_windows_header_matches_row_width(data_files, monkeypatch):
    from offshore_methane import config as cfg
    from offshore_methane import csv_migrate

    path = data_files["windows"]
    path.write_text("0,x1,2022-09-29,2022-09-30,https://example\n")
    monkeypatch.setattr(cfg, "WINDOWS_CSV", path)
    assert csv_migrate.ensure_windows_header()
    frame = csv_utils._read_csv(path)
    assert frame.columns.tolist() == ["id", "structure_id", "start", "end", "citation"]
