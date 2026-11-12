"""Offline format, ownership and partition acceptance tests."""

import io
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

import build_viewer as viewer
import pipeline
import store
import update


def legacy(frame):
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)), low_memory=False)


@pytest.fixture(params=list(store.SCHEMAS))
def sample(request):
    table = request.param
    common = dict(site_id=[1.0, np.nan, 3.0], rh_mw=[1.2345678901234567, np.nan, -0.0])
    if table == "site_nights":
        frame = pd.DataFrame(
            common
            | dict(night=["2026-01-01"] * 3, n_clear=[1.0, np.nan, 0.0], temp_k=np.nan)
        )
    else:
        frame = pd.DataFrame(
            dict(
                time_utc=[
                    "2026-01-02T01:02:03",
                    "2026-01-02 01:02:03.000000Z",
                    "2026-01-02T02:02:03+01:00",
                ],
                peak_row=[1.0, np.nan, 2.0],
                bands=["M10+M11", np.nan, "M10"],
                fit_flag=np.nan,
            )
        )
        if table == "overpasses":
            frame = frame.assign(
                **common,
                detected=[True, np.nan, False],
                split_component_ids=["c1;c2", np.nan, "c3"],
            )
        else:
            frame = frame.assign(
                assigned_site=[1.0, np.nan, 3.0],
                band_rule_changed=[True, np.nan, False],
                pair_drow=["1;2", np.nan, "3"],
            )
    return table, legacy(frame)


def test_round_trip(sample, tmp_path):
    table, frame = sample
    target = tmp_path / "frame.parquet"
    store.write_frame(frame, target, table)
    actual = store.from_arrow(pq.ParquetFile(target).read())
    assert_frame_equal(frame, actual, check_exact=True)
    for col in frame.select_dtypes("float64"):
        np.testing.assert_array_equal(
            frame[col].to_numpy().view("uint64"), actual[col].to_numpy().view("uint64")
        )
    assert pq.ParquetFile(target).schema_arrow.remove_metadata() == store.SCHEMAS[table]
    assert pq.ParquetFile(target).metadata.row_group(0).column(0).compression == "ZSTD"


def test_schema_stable_with_empty_optionals(sample, tmp_path):
    table, frame = sample
    empty = frame.copy()
    for col in empty.columns:
        if col not in {"time_utc", "night"}:
            empty[col] = np.nan
    for name, value in (("full", frame), ("empty", empty), ("no_rows", frame.iloc[:0])):
        store.write_frame(
            value.reset_index(drop=True), tmp_path / f"{name}.parquet", table
        )
    schemas = [
        pq.ParquetFile(tmp_path / f"{name}.parquet").schema_arrow.remove_metadata()
        for name in ("full", "empty", "no_rows")
    ]
    assert schemas[0] == schemas[1] == schemas[2]


@pytest.mark.parametrize("column", ["split_component_ids", "bands", "fit_flag"])
def test_mixed_audit_scalars(column):
    frame = pd.DataFrame({column: ["c1;c2", 1.125, 2, np.nan]})
    assert_frame_equal(
        frame, store.from_arrow(store.to_arrow(frame, "overpasses")), check_exact=True
    )


def test_reject_incompatible_schema():
    with pytest.raises(ValueError, match=r"site_nights.n_clear: 1.5"):
        store.to_arrow(pd.DataFrame({"n_clear": [1.5]}), "site_nights")
    with pytest.raises(ValueError, match="unexpected"):
        store.to_arrow(pd.DataFrame({"unknown": [1]}), "site_nights")
    with pytest.raises(ValueError, match="ISO"):
        store.to_arrow(pd.DataFrame({"night": ["2026-1-1"]}), "site_nights")


def test_reader_prefers_parquet_with_csv_fallback(sample, tmp_path):
    table, frame = sample
    frame.to_csv(tmp_path / f"{table}_tag.csv", index=False)
    csv = store.read_segment(tmp_path, "tag", table)
    store.write_frame(csv, tmp_path / f"{table}_tag.parquet", table)
    assert_frame_equal(
        csv, store.read_segment(tmp_path, "tag", table), check_exact=True
    )
    (tmp_path / f"{table}_tag.csv").write_text("invalid CSV")
    assert_frame_equal(
        csv, store.read_segment(tmp_path, "tag", table), check_exact=True
    )
    assert_frame_equal(
        csv[[csv.columns[0]]],
        store.read_segment(tmp_path, "tag", table, columns=[csv.columns[0]]),
        check_exact=True,
    )


def write_segment(base, number, nights, passes, copies, jobs=None):
    tag = f"2026-08-0{number}_2026-08-0{number + 1}"
    folder = base / f"v5_{number}"
    folder.mkdir()
    pd.DataFrame(nights, columns=["site_id", "night", "lon", "status"]).to_csv(
        folder / f"site_nights_{tag}.csv", index=False
    )
    pd.DataFrame(
        passes, columns=["satellite", "granule", "site_id", "time_utc", "rh_mw"]
    ).to_csv(folder / f"overpasses_{tag}.csv", index=False)
    pd.DataFrame(
        copies,
        columns=[
            "satellite",
            "granule",
            "time_utc",
            "group_id",
            "peak_row",
            "peak_col",
            "centroid_lon",
        ],
    ).to_csv(folder / f"copies_{tag}.csv", index=False)
    if jobs is not None:
        pd.DataFrame(jobs, columns=["satellite", "granule", "complete"]).to_csv(
            folder / f"jobs_{tag}.csv", index=False
        )
    return folder, tag


@pytest.fixture
def segments(tmp_path):
    # Same granule has a subset in the newer snapshot; old site 2 disappears.
    # The incomplete retry cannot remove the older measured granule.
    return [
        write_segment(
            tmp_path,
            1,
            [(1, "2026-08-01", 30.0, "cloudy"), (2, "2026-08-01", -30.0, "flaring")],
            [
                ("SNPP", "boundary", 1, "2026-08-02T01:00:00", 5.0),
                ("SNPP", "boundary", 2, "2026-08-02T02:00:00", 4.0),
                ("SNPP", "retry", 1, "2026-08-02T02:00:00", 3.0),
            ],
            [
                ("SNPP", "boundary", "2026-08-02T01:00:00", 0, 1, 2, 30.0),
                ("SNPP", "retry", "2026-08-02T02:00:00", 0, 3, 4, -30.0),
            ],
            [("SNPP", "boundary", True), ("SNPP", "retry", True)],
        ),
        write_segment(
            tmp_path,
            2,
            [(1, "2026-08-01", 30.0, "flaring"), (1, "2026-08-02", 30.0, "cloudy")],
            [
                ("SNPP", "boundary", 1, "2026-08-02T01:00:00", 9.0),
                ("SNPP", "boundary", 1, "2026-08-02T01:00:00", 10.0),
                ("SNPP", "retry", 1, "2026-08-02T02:00:00", 0.0),
            ],
            [("SNPP", "boundary", "2026-08-02T01:00:00", 0, 1, 2, 30.0)],
            [("SNPP", "boundary", True), ("SNPP", "retry", False)],
        ),
        write_segment(
            tmp_path,
            3,
            [(1, "2026-08-03", 30.0, "clear_no_flare")],
            [("SNPP", "legacy", 1, "2026-08-04T01:00:00", 0.0)],
            [("SNPP", "legacy", "2026-08-04T01:00:00", 0, 1, 2, 30.0)],
        ),
        write_segment(
            tmp_path,
            4,
            [(1, "2026-08-04", 30.0, "clear_no_flare")],
            [("SNPP", "boundary", 1, "2026-08-02T01:00:00", 0.0)],
            [],
            [("SNPP", "boundary", True)],
        ),
    ]


def expected(segments):
    n, o = viewer.load_segments(segments)
    return dict(site_nights=n, overpasses=o, copies=viewer.load_copies(segments))


def test_converter_verifies_and_keeps_csvs(segments):
    folder, tag = segments[0]
    before = {p: p.read_bytes() for p in folder.glob("*.csv")}
    sizes = store.convert(segments[0])
    assert len(sizes) == 3
    for table in store.SCHEMAS:
        assert_frame_equal(
            pd.read_csv(folder / f"{table}_{tag}.csv", low_memory=False),
            store.read_segment(folder, tag, table),
            check_exact=True,
        )
    assert all(p.read_bytes() == value for p, value in before.items())


def test_converter_mismatch_is_not_published(segments, monkeypatch):
    folder, tag = segments[0]

    def mismatch(*args, **kwargs):
        raise AssertionError("mismatch")

    monkeypatch.setattr(store, "assert_frame_equal", mismatch)
    with pytest.raises(AssertionError, match="mismatch"):
        store.convert(segments[0])
    assert not (folder / f"overpasses_{tag}.parquet").exists()
    assert (folder / f"overpasses_{tag}.csv").exists()


def test_find_segments_all_formats(segments):
    store.convert(segments[1])
    store.convert(segments[2])
    for p in segments[2][0].glob("*.csv"):
        p.unlink()
    assert viewer.find_segments(segments[0][0].parent) == segments


def test_store_exactly_matches_legacy(segments, tmp_path):
    frames = expected(segments)
    root = tmp_path / "store"
    store.rebuild(segments, root)
    for table, frame in frames.items():
        assert_frame_equal(frame, store.read(table, root), check_exact=True)
        assert all(
            "night" not in pq.ParquetFile(p).schema_arrow.names
            for p in (root / table).glob("night=*/part.parquet")
        )
    assert (root / "_segments.json").exists()
    assert len(frames["overpasses"]) == 3
    assert set(frames["copies"].granule) == {"retry", "legacy"}


def test_ingest_equals_rebuild_and_preserves_untouched_parts(segments, tmp_path):
    root = tmp_path / "incremental"
    store.rebuild(segments[:3], root)
    untouched = root / "overpasses/night=2026-08-03/part.parquet"
    before = untouched.stat().st_mtime_ns
    store.ingest(segments[3], root)
    assert untouched.stat().st_mtime_ns == before
    full = tmp_path / "full"
    store.rebuild(segments, full)
    for table in store.SCHEMAS:
        assert_frame_equal(
            store.read(table, root), store.read(table, full), check_exact=True
        )
    with pytest.raises(ValueError, match="rebuild"):
        store.ingest(segments[0], root)


def test_sync_actions_and_fingerprints(segments, tmp_path):
    root = tmp_path / "store"
    assert store.sync(segments[:3], root) == "rebuild"
    assert store.sync(segments[:3], root) == "noop"
    assert store.sync(segments, root) == "ingest"
    folder, tag = segments[0]
    with (folder / f"site_nights_{tag}.csv").open("a") as stream:
        stream.write("1,2026-08-01,30.0,cloudy\n")
    assert store.sync(segments, root) == "rebuild"
    manifest = json.loads((root / "_segments.json").read_text())
    assert (
        manifest["segments"][0]["files"][f"site_nights_{tag}.csv"]
        == hashlib.sha256((folder / f"site_nights_{tag}.csv").read_bytes()).hexdigest()
    )
    assert store.sync(segments[:-1], root) == "rebuild"


def test_manifest_only_change_invalidates_store(segments, tmp_path):
    root = tmp_path / "store"
    store.rebuild(segments, root)
    folder, tag = segments[-1]
    pd.DataFrame(
        dict(satellite=["SNPP"], granule=["boundary"], complete=[False])
    ).to_csv(folder / f"jobs_{tag}.csv", index=False)
    assert store.sync(segments, root) == "rebuild"
    assert "boundary" in set(store.read("copies", root).granule)


def test_ingest_changed_history_requires_rebuild(segments, tmp_path):
    root = tmp_path / "store"
    store.rebuild(segments[:3], root)
    folder, tag = segments[0]
    (folder / f"jobs_{tag}.csv").write_text("satellite,granule,complete\n")
    with pytest.raises(ValueError, match="rebuild"):
        store.ingest(segments[-1], root)


def test_evening_nights_east_and_west():
    # Local midnight still belongs to the preceding evening, on both sides.
    frame = pd.DataFrame(
        dict(
            site_id=[1, 1, 2, 2],
            time_utc=[
                "2026-08-01T21:59:59Z",
                "2026-08-01T22:00:01Z",
                "2026-08-02T01:59:59Z",
                "2026-08-02T02:00:01Z",
            ],
            centroid_lon=[30.0, 30.0, -30.0, -30.0],
        )
    )
    sites = pd.Series({1: 30.0, 2: -30.0})
    for table in ("overpasses", "copies"):
        assert store.row_nights(frame, table, sites).tolist() == ["2026-08-01"] * 4
    frame.time_utc = [
        "2026-08-02T09:59:59Z",
        "2026-08-02T10:00:01Z",
        "2026-08-02T13:59:59Z",
        "2026-08-02T14:00:01Z",
    ]
    assert store.row_nights(frame, "overpasses", sites).tolist() == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-01",
        "2026-08-02",
    ]


def test_read_subset(segments, tmp_path):
    root = tmp_path / "store"
    store.rebuild(segments, root)
    frame = expected(segments)["site_nights"]
    assert_frame_equal(
        frame.loc[frame.night.eq("2026-08-03"), ["status", "night"]].reset_index(
            drop=True
        ),
        store.read(
            "site_nights", root, nights=["2026-08-03"], columns=["status", "night"]
        ),
        check_exact=True,
    )
    assert_frame_equal(
        frame.iloc[:0][["night"]],
        store.read("site_nights", root, nights=[], columns=["night"]),
        check_exact=True,
    )
    with pytest.raises(KeyError):
        store.read("site_nights", root, columns=["absent"])


def test_crash_before_move_keeps_existing_file_and_ignores_temp(
    segments, tmp_path, monkeypatch
):
    root = tmp_path / "store"
    store.rebuild(segments, root)
    storage = store._Store(root)
    path = storage.path("overpasses", "night=2026-08-03", "part.parquet")
    original = Path(path).read_bytes()

    class FailMove:
        def __getattr__(self, name):
            return getattr(storage.fs, name)

        def move(self, *args):
            raise RuntimeError("crash")

    broken = store._Store(root)
    broken.fs = FailMove()
    with pytest.raises(RuntimeError, match="crash"):
        broken.atomic(path, lambda stream: stream.write(b"partial parquet"))
    assert Path(path).read_bytes() == original
    assert list(Path(path).parent.glob("part.parquet.*.tmp"))
    assert_frame_equal(
        store.read("overpasses", root),
        expected(segments)["overpasses"],
        check_exact=True,
    )


def test_load_tables_and_last_processed_parquet_only(segments, tmp_path):
    frames = expected(segments)
    for segment in segments:
        store.convert(segment)
        for p in segment[0].glob("*.csv"):
            if not p.name.startswith("jobs_"):
                p.unlink()
    for frame, table in zip(
        viewer.load_tables(tmp_path), ("site_nights", "overpasses", "copies")
    ):
        assert_frame_equal(frame, frames[table], check_exact=True)
    assert update.last_processed(segments).isoformat() == "2026-08-04"


def test_pipeline_exports_legacy_frames_as_parquet(tmp_path, monkeypatch):
    sites = pd.DataFrame(dict(site_id=[1], lat=[0.0], lon=[0.0]))
    sites.to_csv(tmp_path / "sites.csv", index=False)
    from test_granule_fixes import observations

    passes = observations()
    job = dict(
        satellite="SNPP",
        gid="fixture",
        start=pd.Timestamp("2026-10-30T01:00:00").to_pydatetime(),
        sites=[(1, 0.0, 0.0)],
        keys={},
    )
    monkeypatch.setattr(pipeline, "build_jobs", lambda *a: [job])

    def run(*args):
        cache = args[1]
        pipeline._cache_path(job, cache).parent.mkdir(parents=True)
        passes.to_csv(pipeline._cache_path(job, cache), index=False)
        pd.DataFrame(columns=pipeline.granule.COPY_COLUMNS).to_csv(
            pipeline._copies_path(job, cache), index=False
        )
        return passes

    monkeypatch.setattr(pipeline, "run_jobs", run)
    nights = pipeline.main(
        start="2026-10-29",
        end="2026-10-29",
        sites_csv=tmp_path / "sites.csv",
        out_dir=tmp_path,
    )
    tag = "2026-10-29_2026-10-29"
    assert_frame_equal(
        legacy(passes),
        store.read_segment(tmp_path, tag, "overpasses"),
        check_exact=True,
    )
    assert_frame_equal(
        legacy(nights),
        store.read_segment(tmp_path, tag, "site_nights"),
        check_exact=True,
    )
    assert store.read_segment(tmp_path, tag, "copies").empty
    assert not list(tmp_path.glob("overpasses_*.csv"))


def test_column_and_dtype_changes_preserve_untouched_partition(segments, tmp_path):
    folder, tag = segments[0]
    path = folder / f"copies_{tag}.csv"
    copies = pd.read_csv(path)
    copies.loc[copies.granule.eq("retry"), "time_utc"] = "2026-08-04T02:00:00"
    copies["pair_drow"] = [1, 2]
    copies["band_rule_changed"] = [True, False]
    copies.to_csv(path, index=False)
    folder, tag = segments[1]
    path = folder / f"copies_{tag}.csv"
    copies = pd.read_csv(path)
    copies["pair_drow"] = [1.5]
    copies.to_csv(path, index=False)
    root = tmp_path / "store"
    store.rebuild(segments[:1], root)
    untouched = root / "copies/night=2026-08-03/part.parquet"
    before = untouched.stat().st_mtime_ns
    store.ingest(segments[1], root)
    assert untouched.stat().st_mtime_ns == before
    for table, frame in expected(segments[:2]).items():
        assert_frame_equal(frame, store.read(table, root), check_exact=True)


def test_all_empty_copies_round_trip(segments, tmp_path):
    for folder, tag in segments:
        path = folder / f"copies_{tag}.csv"
        pd.read_csv(path).iloc[:0].to_csv(path, index=False)
    root = tmp_path / "store"
    store.rebuild(segments, root)
    assert_frame_equal(
        viewer.load_copies(segments), store.read("copies", root), check_exact=True
    )


def test_local_filesystem_uri(segments, tmp_path):
    root = (tmp_path / "store").as_uri()
    store.rebuild(segments, root)
    assert_frame_equal(
        expected(segments)["site_nights"],
        store.read("site_nights", root),
        check_exact=True,
    )


def test_converter_skips_existing_parquet_unless_forced(segments):
    folder, tag = segments[0]
    target = folder / f"overpasses_{tag}.parquet"
    fresh = pd.read_csv(folder / f"overpasses_{tag}.csv").assign(rh_mw=99.0)
    store.write_frame(fresh, target, "overpasses")
    before = target.read_bytes()
    sizes = store.convert(segments[0])
    assert {row["table"] for row in sizes} == {"site_nights", "copies"}
    assert target.read_bytes() == before
    assert_frame_equal(
        fresh, store.read_segment(folder, tag, "overpasses"), check_exact=True
    )
    assert len(store.convert(segments[0], force=True)) == 3
    assert_frame_equal(
        pd.read_csv(folder / f"overpasses_{tag}.csv"),
        store.read_segment(folder, tag, "overpasses"),
        check_exact=True,
    )


def test_converter_skips_missing_csv(segments):
    folder, tag = segments[0]
    store.convert(segments[0])
    for path in folder.glob("*.csv"):
        if not path.name.startswith("jobs_"):
            path.unlink()
    assert store.convert(segments[0], force=True) == []
    assert store.convert(segments[1])  # A later CSV segment still converts.


def test_content_fingerprint_same_stat_rewrite(segments, tmp_path):
    root = tmp_path / "store"
    store.rebuild(segments, root)
    folder, tag = segments[2]
    source = folder / f"overpasses_{tag}.csv"
    stat = source.stat()
    original = source.read_text()
    changed = original.replace(",0.0", ",9.0")
    assert changed != original and len(changed) == len(original)
    source.write_text(changed)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source.stat().st_size == stat.st_size
    assert source.stat().st_mtime_ns == stat.st_mtime_ns
    assert store.sync(segments, root) == "rebuild"
    assert_frame_equal(
        viewer.load_segments(segments)[1],
        store.read("overpasses", root),
        check_exact=True,
    )


def test_fingerprint_hashes_only_preferred_exports(segments):
    store.convert(segments[0])
    before = store._fingerprint(segments[0])
    folder, tag = segments[0]
    (folder / f"overpasses_{tag}.csv").write_text("stale and ignored")
    assert store._fingerprint(segments[0]) == before


def test_overpass_missing_catalog_sites_use_segment_then_utc(tmp_path):
    pd.DataFrame(dict(site_id=[1], lon=[150.0])).to_csv(
        tmp_path / "gfw_oil_sites.csv", index=False
    )
    segments = [
        write_segment(
            tmp_path,
            1,
            [(1, "2026-08-01", -150.0, "cloudy"), (2, "2026-08-01", -30.0, "cloudy")],
            [("SNPP", "first", sid, "2026-08-02T03:00:00Z", 0.0) for sid in (1, 2, 3)],
            [],
        ),
        write_segment(
            tmp_path,
            2,
            [(2, "2026-08-01", 150.0, "cloudy")],
            [],
            [],
        ),
    ]
    root = tmp_path / "store"
    store.rebuild(segments, root)
    assert store.read("overpasses", root, nights=["2026-08-01"]).site_id.tolist() == [2]
    assert store.read("overpasses", root, nights=["2026-08-02"]).site_id.tolist() == [
        1,
        3,
    ]
    assert_frame_equal(
        viewer.load_segments(segments)[1],
        store.read("overpasses", root),
        check_exact=True,
    )


def test_zero_row_overpasses_concat_matches_csv(segments, tmp_path):
    for i, (folder, tag) in enumerate(segments):
        source = folder / f"overpasses_{tag}.csv"
        frame = pd.read_csv(source).assign(temp_k=np.nan, area_m2=np.nan)
        if i == 1:
            frame = frame.iloc[:0]
        frame.to_csv(source, index=False)
    csv = expected(segments)
    for segment in segments:
        store.convert(segment)
    for table, frame in expected(segments).items():
        assert_frame_equal(csv[table], frame, check_exact=True)
    root = tmp_path / "store"
    store.rebuild(segments, root)
    for table, frame in csv.items():
        assert_frame_equal(frame, store.read(table, root), check_exact=True)


@pytest.mark.parametrize("value", [-0.0, 0.0, 1.0, 2.0, -1.0, "True", "bad"])
def test_flags_reject_nonboolean_values(value):
    with pytest.raises(ValueError) as caught:
        store.to_arrow(pd.DataFrame({"detected": [value, np.nan]}), "overpasses")
    assert f"overpasses.detected: {value!r}:" in str(caught.value)


@pytest.mark.parametrize(
    "frame,table,column,value",
    [
        (
            pd.DataFrame({"fit_flag": [True, np.nan, False]}),
            "copies",
            "fit_flag",
            "True",
        ),
        (
            pd.DataFrame({"site_id": pd.Series([2**63], dtype="uint64")}),
            "site_nights",
            "site_id",
            str(2**63),
        ),
        (pd.DataFrame({"rh_mw": [True, np.nan, False]}), "overpasses", "rh_mw", "True"),
        (
            pd.concat(
                [
                    pd.DataFrame({"detected": [True, False]}),
                    pd.DataFrame({"detected": [np.nan]}),
                ],
                ignore_index=True,
            ),
            "overpasses",
            "detected",
            "1.0",
        ),
    ],
)
def test_unrepresentable_values_name_table_column_value(frame, table, column, value):
    with pytest.raises(ValueError) as caught:
        store.to_arrow(frame, table)
    assert f"{table}.{column}: {value}:" in str(caught.value)


def test_all_blank_nights_round_trip():
    frame = legacy(pd.DataFrame(dict(site_id=[1, 2], night=[np.nan, np.nan])))
    assert_frame_equal(
        frame, store.from_arrow(store.to_arrow(frame, "site_nights")), check_exact=True
    )


def test_object_audit_retains_integer_and_float_scalar_types():
    from check_store_real import _equal

    frame = pd.DataFrame({"fit_flag": pd.Series([1, 1.0, np.nan], dtype=object)})
    _equal(frame, store.from_arrow(store.to_arrow(frame, "overpasses")))


def test_write_export_preserves_csv_inference_and_float_rounding(tmp_path):
    frame = pd.DataFrame({"rh_mw": [0.30000000000000004, -0.0], "bands": [1, 2]})
    csv = legacy(frame)
    assert not np.array_equal(
        frame.rh_mw.to_numpy().view("uint64"), csv.rh_mw.to_numpy().view("uint64")
    )
    target = tmp_path / "export.parquet"
    store.write_export(frame, target, "overpasses")
    from check_store_real import _equal

    _equal(csv, store.read_export(target))


def test_real_reference_uses_head_csv_loader_and_independent_pickles(
    segments, tmp_path, monkeypatch
):
    import check_store_real as check

    try:
        head = check._head_loader()
    except SystemExit as missing:  # shallow clone or a copy outside git
        pytest.skip(str(missing))
    references = dict(
        zip(check.TABLES, (*head.load_segments(segments), head.load_copies(segments)))
    )
    for segment in segments:
        store.convert(segment)

    def forbidden(*args, **kwargs):
        raise AssertionError("working-tree loader or encoder used for reference")

    monkeypatch.setattr(viewer, "load_segments", forbidden)
    monkeypatch.setattr(store, "to_arrow", forbidden)
    monkeypatch.setattr(store, "from_arrow", forbidden)
    monkeypatch.setattr(check, "BASE", tmp_path)
    monkeypatch.setattr(check, "ARTIFACTS", tmp_path / "store_acceptance")
    check.main("csv")
    for table, frame in references.items():
        check._equal(frame, pd.read_pickle(check.ARTIFACTS / f"expected-{table}.pkl"))


def test_microsecond_timestamps_and_loss_guard(tmp_path):
    frame = pd.DataFrame({"time_utc": ["2026-01-01T01:02:03.123456Z"]})
    target = tmp_path / "frame.parquet"
    store.write_frame(frame, target, "overpasses")
    assert pq.ParquetFile(target).schema_arrow.field("time_utc").type == pa.timestamp(
        "us", tz="UTC"
    )
    assert_frame_equal(frame, store.read_export(target), check_exact=True)
    bad = "2026-01-01T01:02:03.123456001Z"
    with pytest.raises(ValueError) as caught:
        store.write_frame(pd.DataFrame({"time_utc": [bad]}), target, "overpasses")
    assert f"overpasses.time_utc: {bad!r}:" in str(caught.value)


def test_read_accepts_date_nights(segments, tmp_path):
    root = tmp_path / "store"
    store.rebuild(segments, root)
    assert_frame_equal(
        store.read("site_nights", root, nights=["2026-08-03"]),
        store.read("site_nights", root, nights=[date(2026, 8, 3)]),
        check_exact=True,
    )


def test_main_empty_segments_means_discover_all(tmp_path, monkeypatch):
    frames = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(viewer, "load_tables", lambda base, root: frames)

    class ReachedBuildData(Exception):
        pass

    def build(nights, over):
        assert nights is frames[0] and over is frames[1]
        raise ReachedBuildData

    monkeypatch.setattr(viewer, "build_data", build)
    with pytest.raises(ReachedBuildData):
        viewer.main(segments=[], base=tmp_path)


@pytest.mark.parametrize(
    "fault", ["signed_zero", "object_type", "object_null", "object_signed_zero"]
)
def test_real_comparison_catches_decoder_faults(fault, tmp_path, monkeypatch):
    from check_store_real import _equal

    frames = {
        "signed_zero": (pd.DataFrame({"rh_mw": [-0.0, 1.0]}), "rh_mw", 0.0),
        "object_type": (
            pd.DataFrame({"fit_flag": pd.Series([1, "ok"], dtype=object)}),
            "fit_flag",
            1.0,
        ),
        "object_null": (
            pd.DataFrame({"fit_flag": pd.Series([np.nan, "ok"], dtype=object)}),
            "fit_flag",
            None,
        ),
        "object_signed_zero": (
            pd.DataFrame({"fit_flag": pd.Series([-0.0, "ok"], dtype=object)}),
            "fit_flag",
            0.0,
        ),
    }
    frame, col, replacement = frames[fault]
    reference = tmp_path / "reference.pkl"
    frame.to_pickle(reference)
    original = store.from_arrow

    def faulty(*args, **kwargs):
        decoded = original(*args, **kwargs)
        decoded.loc[0, col] = replacement
        return decoded

    monkeypatch.setattr(store, "from_arrow", faulty)
    decoded = store.from_arrow(store.to_arrow(frame, "overpasses"))
    with pytest.raises(AssertionError):
        _equal(pd.read_pickle(reference), decoded)


def test_head_loader_unavailable_is_clear(monkeypatch):
    import check_store_real as check
    from types import SimpleNamespace

    monkeypatch.setattr(
        check.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    with pytest.raises(SystemExit, match="HEAD.*unavailable"):
        check._head_loader()


def test_explainer_reads_parquet_only_copies(segments, tmp_path, monkeypatch):
    # Exercise the figure data loader without imagery or validation inputs.
    for folder, tag in segments:
        path = folder / f"copies_{tag}.csv"
        pd.read_csv(path).assign(scan=0).to_csv(path, index=False)
        path = folder / f"overpasses_{tag}.csv"
        pd.read_csv(path).assign(sza=100.0, detected=False).to_csv(path, index=False)
        store.convert((folder, tag))
        (folder / f"copies_{tag}.csv").unlink()
    monkeypatch.setattr(viewer, "find_segments", lambda: segments)
    spec = importlib.util.spec_from_file_location(
        "figure_fixture",
        Path(__file__).resolve().parents[1] / "docs/explainer/make_figures.py",
    )
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    catalog = tmp_path / "sites.csv"
    pd.DataFrame(dict(site_id=[1, 2], lon=[30.0, -30.0])).to_csv(catalog, index=False)
    monkeypatch.setattr(figures.p, "SITES_CSV", catalog)
    monkeypatch.setattr(figures.p, "dedupe_overpasses", lambda frame: frame)
    monkeypatch.setattr(
        figures.p, "strict_clear", lambda frame: pd.Series(True, index=frame.index)
    )
    monkeypatch.setattr(figures, "_lh", lambda run: None)
    monkeypatch.setattr(figures.V, "load_myc", lambda year: None)
    monkeypatch.setattr(figures.V, "match_sites", lambda *args: None)
    copies = figures.load()[-1]
    assert not copies.empty and "scan" in copies


def test_nonlocal_writes_create_no_directory_markers(segments, tmp_path):
    storage = store._Store(tmp_path / "object-store")
    local = storage.fs

    class ObjectKeys:
        def __getattr__(self, name):
            return getattr(local, name)

        def create_dir(self, *args, **kwargs):
            raise AssertionError("nonlocal directory marker")

        def open_output_stream(self, path):
            # Simulate an object store, where keys need no parent markers.
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return local.open_output_stream(path)

    storage.fs = ObjectKeys()
    store._materialize(segments, storage, replace=True)
    assert storage.manifest()


def _sync_process(segments, root, attempted, done, paused=None, release=None):
    if paused is not None:
        original = store._Store.atomic
        stopped = False

        def atomic(self, path, writer):
            nonlocal stopped
            original(self, path, writer)
            if not stopped and "night=" in path:
                stopped = True
                paused.set()
                if not release.wait(20):
                    raise RuntimeError("writer test barrier timed out")

        store._Store.atomic = atomic
    attempted.set()
    store.sync(segments, root)
    done.set()


def _read_process(
    segments, root, mode, attempted, done, results, paused=None, release=None
):
    if paused is not None:
        original = store._read
        stopped = False

        def read(*args, **kwargs):
            nonlocal stopped
            frame = original(*args, **kwargs)
            if not stopped:
                stopped = True
                paused.set()
                if not release.wait(20):
                    raise RuntimeError("reader test barrier timed out")
            return frame

        store._read = read
    attempted.set()
    frames = (
        (store.read("overpasses", root),)
        if mode == "read"
        else store.load_tables(segments, root)
    )
    results.put(frames)
    done.set()


def _finish_processes(processes, release):
    release.set()
    for process in processes:
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join()
        assert process.exitcode == 0


def test_two_sync_processes_serialize_complete_stores(segments, tmp_path):
    ctx = mp.get_context("spawn")
    root = tmp_path / "store"
    a_started, a_done, paused, release, b_started, b_done = [
        ctx.Event() for _ in range(6)
    ]
    first = ctx.Process(
        target=_sync_process,
        args=(segments[:3], root, a_started, a_done, paused, release),
    )
    second = ctx.Process(target=_sync_process, args=(segments, root, b_started, b_done))
    first.start()
    processes = [first]
    try:
        assert paused.wait(15)
        second.start()
        processes.append(second)
        assert b_started.wait(15)
        assert not b_done.wait(0.5), "second writer bypassed the exclusive lock"
    finally:
        _finish_processes(processes, release)
    manifest = json.loads((root / "_segments.json").read_text())
    full = tmp_path / "full"
    store.rebuild(segments, full)
    assert manifest == json.loads((full / "_segments.json").read_text())
    for table in store.SCHEMAS:
        assert_frame_equal(
            store.read(table, root), store.read(table, full), check_exact=True
        )
    assert store.sync(segments, root) == "noop"


@pytest.mark.parametrize("mode", ["read", "load_tables"])
def test_reader_waits_for_complete_writer(segments, tmp_path, mode):
    ctx = mp.get_context("spawn")
    root = tmp_path / "store"
    store.rebuild(segments[:3], root)
    a_started, a_done, paused, release, b_started, b_done = [
        ctx.Event() for _ in range(6)
    ]
    results = ctx.Queue()
    writer = ctx.Process(
        target=_sync_process, args=(segments, root, a_started, a_done, paused, release)
    )
    reader = ctx.Process(
        target=_read_process, args=(segments, root, mode, b_started, b_done, results)
    )
    writer.start()
    processes = [writer]
    try:
        assert paused.wait(15)
        reader.start()
        processes.append(reader)
        assert b_started.wait(15)
        assert not b_done.wait(0.5), "reader bypassed an in-progress writer"
        release.set()
        frames = results.get(timeout=20)
    finally:
        _finish_processes(processes, release)
    tables = (
        ("overpasses",) if mode == "read" else ("site_nights", "overpasses", "copies")
    )
    for table, frame in zip(tables, frames):
        assert_frame_equal(expected(segments)[table], frame, check_exact=True)


def test_load_tables_locks_across_three_reads(segments, tmp_path):
    ctx = mp.get_context("spawn")
    root = tmp_path / "store"
    store.rebuild(segments[:3], root)
    a_started, a_done, paused, release, b_started, b_done = [
        ctx.Event() for _ in range(6)
    ]
    results = ctx.Queue()
    reader = ctx.Process(
        target=_read_process,
        args=(
            segments[:3],
            root,
            "load_tables",
            a_started,
            a_done,
            results,
            paused,
            release,
        ),
    )
    writer = ctx.Process(target=_sync_process, args=(segments, root, b_started, b_done))
    reader.start()
    processes = [reader]
    try:
        assert paused.wait(15)
        writer.start()
        processes.append(writer)
        assert b_started.wait(15)
        assert not b_done.wait(0.5), "writer interleaved the three table reads"
        release.set()
        frames = results.get(timeout=20)
    finally:
        _finish_processes(processes, release)
    for table, frame in zip(("site_nights", "overpasses", "copies"), frames):
        assert_frame_equal(expected(segments[:3])[table], frame, check_exact=True)


def test_failed_ingest_invalidates_manifest_and_sync_recovers(
    segments, tmp_path, monkeypatch
):
    root = tmp_path / "store"
    store.rebuild(segments[:3], root)
    atomic = store._Store.atomic

    def fail(self, path, writer):
        if "night=" in path:
            raise RuntimeError("crash")
        return atomic(self, path, writer)

    monkeypatch.setattr(store._Store, "atomic", fail)
    with pytest.raises(RuntimeError, match="crash"):
        store.ingest(segments[-1], root)
    with pytest.raises(FileNotFoundError, match="rebuild"):
        store.read("site_nights", root)
    monkeypatch.setattr(store._Store, "atomic", atomic)
    assert store.sync(segments, root) == "rebuild"
    for table, frame in expected(segments).items():
        assert_frame_equal(frame, store.read(table, root), check_exact=True)


def test_code_change_rebuilds_store(segments, tmp_path, monkeypatch):
    root = tmp_path / "store"
    assert store.sync(segments, root) == "rebuild"
    assert store.sync(segments, root) == "noop"
    monkeypatch.setattr(store, "_code_key", lambda: "changed merge rules")
    assert store.sync(segments, root) == "rebuild"
    assert store.sync(segments, root) == "noop"


def test_code_key_covers_store_and_viewer_loaders(monkeypatch):
    before = store._code_key()
    monkeypatch.setattr(viewer, "load_copies", lambda segments: None)
    assert store._code_key() != before


def test_copy_without_centroid_uses_utc_date(tmp_path):
    segments = [
        write_segment(
            tmp_path,
            1,
            [(1, "2026-08-01", 150.0, "cloudy")],
            [("SNPP", "g", 1, "2026-08-02T03:00:00", 0.0)],
            [
                ("SNPP", "g", "2026-08-02T03:00:00", 0, 1, 2, 150.0),
                ("SNPP", "g", "2026-08-02T03:00:00", 1, 3, 4, np.nan),
            ],
        )
    ]
    root = tmp_path / "store"
    store.rebuild(segments, root)
    assert store.read("copies", root, nights=["2026-08-01"]).group_id.tolist() == []
    assert store.read("copies", root, nights=["2026-08-02"]).group_id.tolist() == [0, 1]
    assert_frame_equal(
        viewer.load_copies(segments), store.read("copies", root), check_exact=True
    )


def test_materialize_removes_crashed_writer_temp_files(segments, tmp_path):
    root = tmp_path / "store"
    store.rebuild(segments, root)
    part = next((root / "overpasses").glob("night=*/part.parquet"))
    leftovers = [
        part.with_name(f"part.parquet.123.{'a' * 32}.tmp"),
        root / f"_segments.json.123.{'b' * 32}.tmp",
    ]
    # Files the store does not own: a pipeline export mid-write under the root
    # and an unrelated temp file next to a night file.
    foreign = [
        root / "v5_live" / "overpasses_x.parquet.1.abc.tmp",
        part.with_name("notes.tmp"),
        root / "other.tmp",
        part.with_name("part.parquet.tmp"),
        root / "_segments.json.tmp",
        part.with_name("part.parquet.123.tmp"),
    ]
    foreign[0].parent.mkdir()
    for path in leftovers + foreign:
        path.write_bytes(b"partial")
    store.rebuild(segments, root)
    assert not any(path.exists() for path in leftovers)
    assert all(path.exists() for path in foreign)
    assert_frame_equal(
        viewer.load_segments(segments)[1], store.read("overpasses", root)
    )


def test_acceptance_convert_keeps_newer_parquet(segments, tmp_path, monkeypatch):
    import check_store_real as check

    folder, tag = segments[0]
    fresh = viewer.load_segments([segments[0]])[1].assign(rh_mw=99.0)
    store.write_export(fresh, folder / f"overpasses_{tag}.parquet", "overpasses")
    monkeypatch.setattr(check, "BASE", tmp_path)
    monkeypatch.setattr(check, "ARTIFACTS", tmp_path / "store_acceptance")
    check.main("convert")
    assert set(store.read_segment(folder, tag, "overpasses").rh_mw) == {99.0}


def test_acceptance_reference_is_pinned_before_this_change():
    import check_store_real as check

    assert len(check.REFERENCE) == 40
    assert "HEAD:" not in Path(check.__file__).read_text()


def test_code_key_describes_loaded_code_not_files_on_disk(tmp_path, monkeypatch):
    """A source edit after import must not be stamped onto a store it did not build."""
    before = store._code_key()
    source = Path(viewer.__file__).read_text()
    edited = tmp_path / "build_viewer.py"
    edited.write_text(source.replace('keep="last"', 'keep="first"'))
    assert edited.read_text() != source
    monkeypatch.setattr(viewer, "__file__", str(edited))
    monkeypatch.setattr(store, "__file__", str(tmp_path / "store.py"))
    (tmp_path / "store.py").write_text("changed on disk")
    import linecache

    linecache.clearcache()
    assert store._code_key() == before


def test_code_key_changes_with_loader_bytecode_and_library_versions(monkeypatch):
    before = store._code_key()

    def load_copies(segments, *, _ordered=False):
        return viewer.load_segments(segments)[1].head(0)

    monkeypatch.setattr(viewer, "load_copies", load_copies)
    changed = store._code_key()
    assert changed != before
    monkeypatch.undo()
    assert store._code_key() == before
    monkeypatch.setattr(pd, "__version__", "0.0.0")
    assert store._code_key() != before


def test_acceptance_artifacts_are_outside_the_store_root():
    import check_store_real as check

    assert check.ROOT not in [check.ARTIFACTS, *check.ARTIFACTS.parents]


def test_code_key_is_stable_across_hash_seeds():
    """Set constants print in hash order; the digest must not depend on it."""
    import subprocess
    import sys

    keys = {
        subprocess.run(
            [sys.executable, "-c", "import store; print(store._code_key())"],
            cwd=Path(store.__file__).parent,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert keys == {store._code_key()}


def test_code_key_covers_loader_defaults(monkeypatch):
    before = store._code_key()
    monkeypatch.setattr(viewer.load_copies, "__kwdefaults__", {"_ordered": True})
    assert store._code_key() != before


def test_code_key_covers_the_store_module_the_loaders_use(monkeypatch):
    before = store._code_key()
    monkeypatch.setattr(viewer.store, "_SOURCE_DIGEST", "edited store cells")
    assert store._code_key() != before


def test_loaders_read_only_globals_the_code_key_accounts_for():
    """The digest does not follow other build_viewer globals; keep it sufficient."""

    def names(code):
        found = set(code.co_names)
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                found |= names(const)
        return found

    module_globals = set(vars(viewer)) - set(vars(__import__("builtins")))
    used = set()
    for loader in (viewer.granule_owners, viewer.load_segments, viewer.load_copies):
        used |= names(loader.__code__) & module_globals
    assert used <= {"pd", "np", "store", "granule_owners"}
