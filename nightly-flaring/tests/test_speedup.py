"""Network-free exactness tests for T0-T7, using real HDF5 range reads."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import requests

import bench_speedup as bench
import granule as g
import nodd
import pipeline as p
import speedup as s


class Response:
    def __init__(self, content=b"", status=206, headers=None):
        self.content, self.status_code = content, status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(str(self.status_code))
            error.response = self
            raise error


class LocalHTTP:
    def __init__(self, files):
        self.files, self.calls = files, []
        self.changed = set()

    def head(self, url, **kwargs):
        key = url.rsplit("/", 1)[-1]
        self.calls.append(("HEAD", key))
        content = self.files[key]
        etag = hashlib.sha256(content).hexdigest()
        return Response(
            headers={
                "Content-Length": str(len(content)),
                "ETag": etag + ("changed" if key in self.changed else ""),
            }
        )

    def get(self, url, headers, **kwargs):
        key = url.rsplit("/", 1)[-1]
        self.calls.append(("GET", key))
        lo, hi = map(int, headers["Range"][6:].split("-"))
        return Response(self.files[key][lo : hi + 1])


def scene(
    tmp_path,
    monkeypatch,
    *,
    mismatch=False,
    terminator=False,
    ties=False,
    cloud=True,
    fills=False,
):
    lat, lon = np.meshgrid(
        np.arange(32, dtype="f4") * 0.006,
        np.arange(96, dtype="f4") * 0.006,
        indexing="ij",
    )
    if terminator:
        lon[:] = np.linspace(-110, 110, lon.shape[1], dtype="f4")
    if ties:
        lat[5, 36], lon[5, 36] = lat[5, 35], lon[5, 35]
    if fills:
        lat[0, :], lon[0, :] = -999.9, -999.9
    files = {}
    for name in ("geo", "cloud", "M10", "M11", "M12", "M13"):
        path = tmp_path / f"{name}.h5"
        with h5py.File(path, "w") as h:
            if name == "geo":
                h.create_dataset(
                    f"{s.GEO}/Latitude", data=lat, compression="gzip", chunks=lat.shape
                )
                h.create_dataset(
                    f"{s.GEO}/Longitude", data=lon, compression="gzip", chunks=lon.shape
                )
                ds = h.create_dataset(
                    "Data_Products/VIIRS-MOD-GEO-TC/VIIRS-MOD-GEO-TC_Gran_0", data=[0]
                )
                ds.attrs["G-Ring_Latitude"] = lat[[0, 0, -1, -1], [0, -1, -1, 0]]
                ds.attrs["G-Ring_Longitude"] = lon[[0, 0, -1, -1], [0, -1, -1, 0]]
                ds.attrs["N_Day_Night_Flag"] = np.bytes_("Night")
            elif name == "cloud":
                h.create_dataset(
                    "Latitude",
                    data=lat + (0.001 if mismatch else 0),
                    compression="gzip",
                    chunks=(8, 12),
                )
                h.create_dataset(
                    "Longitude", data=lon, compression="gzip", chunks=(8, 12)
                )
                cm = np.zeros_like(lat, dtype="u1")
                cm[5, 35] = 3
                h.create_dataset(
                    "CloudMask", data=cm, compression="gzip", chunks=(8, 12)
                )
            else:
                radiance = np.zeros_like(lat)
                radiance[5, 35] = 1
                h.create_dataset(
                    f"All_Data/VIIRS-{name}-SDR_All/Radiance",
                    data=radiance,
                    compression="gzip",
                )
                h.create_dataset(
                    f"All_Data/VIIRS-{name}-SDR_All/QF1_VIIRSMBANDSDR",
                    data=np.zeros(lat.shape, dtype="u1"),
                    compression="gzip",
                )
        files[name] = path.read_bytes()
    session = LocalHTTP(files)
    monkeypatch.setattr(g, "_SESSION", session)
    monkeypatch.setattr(
        g.nf,
        "fit_planck",
        lambda bands, obs, noise, **kwargs: (1800, 0.001, 0, bands, "ok"),
    )
    job = dict(
        satellite="NOAA-20",
        gid="d20260620_t0000000",
        bucket="azure:noaa-20",
        start=datetime(2026, 6, 20),
        keys={k: k for k in files if k != "cloud" or cloud},
        sites=[
            (1, float(lat[5, 35]), float(lon[5, 35])),
            (2, float(lat[20, 70]), float(lon[20, 70])),
        ],
    )
    monkeypatch.setattr(
        g.orbit,
        "predict",
        lambda *a: dict(
            row=np.array([5.0, 20.0]),
            col=np.array([35.0, 70.0]),
            eligible=np.ones(2, bool),
        ),
    )
    return job, session, lat, lon


def tables(result):
    frames = []
    for name, cols in (("records", g.RECORD_COLUMNS), ("copies", g.COPY_COLUMNS)):
        frame = pd.DataFrame(result[name], columns=cols)
        frames.append(
            frame.drop(columns=["bytes_granule"], errors="ignore").to_csv(index=False)
        )
    return frames


@pytest.mark.parametrize(
    "mode", ["t1", "t1_t2", "best", "tier_order", "satellite_first"]
)
def test_modes_match_real_hdf5_range_control(tmp_path, monkeypatch, mode):
    job, session, _, _ = scene(tmp_path, monkeypatch)
    control = g.process_job(job)
    assert (
        control["error"] is None and len(control["records"]) == 2 and control["copies"]
    )
    opts = bench.mode_options(mode, (1,), tmp_path)
    actual = g.process_job(dict(job, speedup=opts))
    assert actual["error"] is None
    assert tables(actual) == tables(control)
    assert actual["stats"]["requests"] > actual["stats"]["heads"] > 0
    # The retained GMTCO header and JRR handle each incur only one HEAD.
    assert session.calls.count(("HEAD", "geo")) == 2  # control + best
    if mode != "t1":
        assert actual["stats"]["geo_mode"] == "windowed"
        assert actual["stats"]["geo_chunks"] > 0
        assert session.calls.count(("HEAD", "cloud")) == 2


@pytest.mark.parametrize(
    "reason", ["terminator", "ties", "missing", "prediction", "historic", "layout"]
)
def test_sparse_uncertainties_fall_back_exactly(tmp_path, monkeypatch, reason):
    job, session, _, _ = scene(
        tmp_path,
        monkeypatch,
        terminator=reason == "terminator",
        ties=reason == "ties",
        cloud=reason != "missing",
    )
    if reason == "prediction":

        def fail(*a):
            raise ValueError("stale TLE")

        monkeypatch.setattr(g.orbit, "predict", fail)
    if reason == "historic":
        job["start"] = datetime(2024, 6, 20)
    if reason == "layout":
        with h5py.File(tmp_path / "cloud.h5", "a") as h:
            del h["Latitude"]
        session.files["cloud"] = (tmp_path / "cloud.h5").read_bytes()
    control = g.process_job(job)
    actual = g.process_job(
        dict(job, speedup=asdict(s.Options(windowed_geo=True, tle_dir=str(tmp_path))))
    )
    assert actual["error"] is None and tables(actual) == tables(control)
    assert actual["stats"]["geo_mode"].startswith("fallback")


def test_sparse_with_fill_and_finite_m10_nearest(tmp_path, monkeypatch):
    _, _, lat, lon = scene(tmp_path, monkeypatch, fills=True)
    rng = np.random.default_rng(3)
    l10 = np.ones_like(lat)
    l10[(rng.random(lat.shape) < 0.3) | ~s.geo_valid(lat, lon)] = np.nan
    sites = np.asarray(
        [
            (i, float(lat[r, c]), float(lon[r, c]))
            for i, (r, c) in enumerate(
                zip(rng.integers(2, 30, 20), rng.integers(2, 94, 20))
            )
        ],
        dtype=object,
    )
    expected = g.site_pixels(lat, lon, l10, sites)
    prediction = dict(row=expected[0], col=expected[1])
    with h5py.File(tmp_path / "cloud.h5", "r") as h:
        sparse = s.SparseGeo(h, lat.shape)
        actual = sparse.nearest(l10, sites, prediction, s.Options())
    assert actual is not None
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


def test_t1_uses_ring_and_never_day_flag(tmp_path, monkeypatch):
    job, session, _, _ = scene(tmp_path, monkeypatch)
    with h5py.File(tmp_path / "geo.h5") as h:
        attrs = s.header(h)
    attrs["N_Day_Night_Flag"] = "Day"
    assert not s.prefilter(job, s.Options(), attrs)[0]
    job["sites"] = [(10, 0, 180)]
    assert s.prefilter(job, s.Options(), attrs) == (True, "site_solar_bound")
    job["sites"] = [(1, 10, 0)]
    assert s.prefilter(job, s.Options(), attrs) == (True, "header_footprint")
    job["sites"] = [(1, 0, -0.005)]
    assert not s.prefilter(job, s.Options(), attrs)[0]
    assert not s.prefilter(job, s.Options(), {})[0]


def test_reduced_mode_only_loses_declared_site_information(tmp_path, monkeypatch):
    job, session, _, _ = scene(tmp_path, monkeypatch)
    control = g.process_job(job)
    actual = g.process_job(
        dict(
            job, speedup=asdict(s.Options(tier2_detections_only=True, tier1_ids=(999,)))
        )
    )
    assert actual["error"] is None
    expected = [dict(r) for r in control["records"] if r["detected"]]
    assert len(actual["records"]) == len(expected) == 1
    for a, e in zip(actual["records"], expected):
        assert a["cloud_mask"] == -1 and a["cloud_clear_neighbors"] == 0
        a = dict(a)
        for key in ("cloud_mask", "cloud_clear_neighbors", "bytes_granule"):
            a.pop(key)
            e.pop(key)
        pd.testing.assert_frame_equal(
            pd.DataFrame([a]), pd.DataFrame([e]), check_dtype=False
        )
    assert tables(actual)[1] == tables(control)[1]
    tier1 = g.process_job(
        dict(job, speedup=asdict(s.Options(tier2_detections_only=True, tier1_ids=(1,))))
    )
    assert tables(tier1) == tables(control)


def test_counting_includes_errors_and_connection_attempts(monkeypatch):
    class Broken:
        def get(self, *a, **kw):
            return Response(b"error response", 503)

        def head(self, *a, **kw):
            raise requests.Timeout()

    counter = nodd.CountingSession(Broken())
    with pytest.raises(requests.HTTPError):
        counter.get("u").raise_for_status()
    with pytest.raises(requests.Timeout):
        counter.head("u")
    assert counter.requests == 2 and counter.heads == 1
    assert counter.bytes_fetched == len(b"error response")
    monkeypatch.setattr(g, "_SESSION", Broken())

    def failed(job):
        g._session().get("u").raise_for_status()

    monkeypatch.setattr(g, "process_granule", failed)
    result = g.process_job({})
    assert result["error"] and result["stats"]["bytes_fetched"] == len(
        b"error response"
    )
    assert result["stats"]["accounting_complete"]


def test_pipeline_sidecars_include_empty_jobs_and_failed_attempts(
    tmp_path, monkeypatch
):
    jobs = [dict(satellite="SNPP", gid=f"g{i}", sites=[]) for i in range(2)]
    calls = {}

    def work(job):
        i = job["gid"]
        calls[i] = calls.get(i, 0) + 1
        return dict(
            records=[],
            copies=[],
            error="retry" if i == "g0" and calls[i] == 1 else None,
            stats=dict(
                bytes_fetched=17,
                requests=3,
                heads=1,
                cpu_s=0.1,
                wall_s=0.2,
                accounting_complete=True,
            ),
        )

    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(g, "process_job", work)
    assert p.run_jobs(jobs, tmp_path, workers=2).empty
    summary = p.traffic_summary(jobs, tmp_path)
    assert summary["bytes_fetched"] == 51 and summary["requests"] == 9
    assert summary["accounted_jobs"] == 2 and summary["accounting_complete"]
    assert all(p.cache_complete(j, tmp_path) for j in jobs)
    assert (
        len(json.loads(p._stats_path(jobs[0], tmp_path).read_text())["attempts"]) == 2
    )
    assert (
        p.traffic_summary([dict(satellite="SNPP", gid="old")], tmp_path)["unknown_jobs"]
        == 1
    )
    with pytest.raises(ValueError, match="differ"):
        p.run_jobs(
            [dict(j, speedup=dict(header_prefilter=True)) for j in jobs], tmp_path
        )


def test_ordering_preserves_scope_and_defaults():
    jobs = [
        dict(
            satellite=sat,
            gid=f"g{i}",
            start=datetime(2026, 9, 20, i),
            sites=[(i, 0, 0)],
        )
        for i, sat in enumerate(("SNPP", "NOAA-20", "NOAA-21"))
    ]
    assert [j["gid"] for j in s.configure(jobs, s.Options())] == ["g0", "g1", "g2"]
    assert [
        j["gid"] for j in s.configure(jobs, s.Options(tier_order=True, tier1_ids=(2,)))
    ] == ["g2", "g0", "g1"]
    assert [
        j["gid"] for j in s.configure(jobs, s.Options(satellite_first="NOAA-20"))
    ] == ["g1", "g0", "g2"]
    assert all("speedup" not in j for j in jobs)
    with pytest.raises(ValueError):
        s.Options(tier_order=True)


def test_csv_comparison_is_exact_except_explicit_columns(tmp_path):
    jobs = [dict(satellite="SNPP", gid="g1", sites=[])]
    cache, reference = tmp_path / "actual", tmp_path / "reference"
    for root in (cache, reference):
        (root / "SNPP").mkdir(parents=True)
        p._cache_path(jobs[0], root).write_text(
            "bytes_granule,dist_m\n1,0.1234567890123\n"
        )
        p._copies_path(jobs[0], root).write_text("rh75_mw\n1.234567890123\n")
    p._cache_path(jobs[0], cache).write_text(
        "bytes_granule,dist_m\n99,0.1234567890123\n"
    )
    result = bench.compare(jobs, cache, reference, {})
    assert result["identical"] and result["changed_columns"] == ["bytes_granule"]
    p._copies_path(jobs[0], cache).write_text("rh75_mw\n1.234567890124\n")
    assert not bench.compare(jobs, cache, reference, {})["identical"]
    p._copies_path(jobs[0], cache).unlink()
    assert not bench.compare(jobs, cache, reference, {})["identical"]


def test_offline_plan_and_protocol_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUTPUT", tmp_path)
    result = bench.main(out_dir=tmp_path)
    assert Path(result["driver"]).exists()
    assert '__name__ == "__main__"' in Path(result["driver"]).read_text()
    schedule = bench.protocol_schedule()
    assert {row[0] for row in schedule} == set(bench.NIGHTS)
    for night in bench.NIGHTS:
        rows = [row for row in schedule if row[0] == night]
        expected = (
            ["control", "best", "control_repeat"]
            if night in bench.REFERENCES
            else ["control", "best"]
        )
        assert [r[3] for r in rows[: len(expected)]] == expected
    assert {r[2] for r in schedule} == {12, 24, 32}
    with pytest.raises(ValueError, match="stay under"):
        bench._output(tmp_path.parent / "elsewhere")


@pytest.mark.parametrize(
    "lat,lon", [(89.9, 179.9), (-89.9, -179.9), (0, 179.999), (0, -179.999)]
)
def test_footprint_contains_dateline_and_poles(lat, lon):
    lat = float(np.clip(lat, -89.8, 89.8))
    attrs = dict(
        zip(
            ("G-Ring_Latitude", "G-Ring_Longitude"),
            (
                [lat - 0.1, lat - 0.1, lat + 0.1, lat + 0.1],
                [
                    (lon - 0.2 + 180) % 360 - 180,
                    (lon + 0.2 + 180) % 360 - 180,
                    (lon + 0.2 + 180) % 360 - 180,
                    (lon - 0.2 + 180) % 360 - 180,
                ],
            ),
        )
    )
    assert not s.footprint_skip(dict(sites=[(1, lat, lon)]), attrs)


def test_reduced_csv_exclusions_are_sufficient(tmp_path, monkeypatch):
    job, _, _, _ = scene(tmp_path, monkeypatch)
    control = g.process_job(job)
    opts = asdict(s.Options(tier2_detections_only=True, tier1_ids=(999,)))
    reduced = g.process_job(dict(job, speedup=opts))
    cache, reference = tmp_path / "actual", tmp_path / "reference"
    for result, root in ((control, reference), (reduced, cache)):
        (root / job["satellite"]).mkdir(parents=True)
        p._write_csv_atomic(
            pd.DataFrame(result["records"], columns=g.RECORD_COLUMNS),
            p._cache_path(job, root),
        )
        p._write_csv_atomic(
            pd.DataFrame(result["copies"], columns=g.COPY_COLUMNS),
            p._copies_path(job, root),
        )
    result = bench.compare([job], cache, reference, opts)
    assert result["identical"] and not result["result_preserving"]
    assert result["intentional_dropped_rows"] == 1
    assert set(result["changed_columns"]) <= {
        "bytes_granule",
        "cloud_mask",
        "cloud_clear_neighbors",
    }


def test_reduced_pipeline_never_builds_normal_site_nights(tmp_path, monkeypatch):
    sites = tmp_path / "sites.csv"
    pd.DataFrame(dict(site_id=[1], lat=[0], lon=[0])).to_csv(sites, index=False)
    monkeypatch.setattr(p, "build_jobs", lambda *a: [])
    monkeypatch.setattr(p, "run_jobs", lambda *a: pd.DataFrame())

    def forbidden(*a):
        raise AssertionError("reduced mode cannot certify clear exposures")

    monkeypatch.setattr(p, "site_nights", forbidden)
    p.main(
        start="2026-09-20",
        end="2026-09-20",
        out_dir=tmp_path,
        sites_csv=sites,
        speedup=dict(tier2_detections_only=True, tier1_ids=(999,)),
    )
    assert not list(tmp_path.glob("site_nights_*.csv"))
    assert list(tmp_path.glob("detections_and_tier1_*.csv"))


def test_harness_runs_counts_compares_and_rejects_wrong_resume(tmp_path, monkeypatch):
    job, _, _, _ = scene(tmp_path, monkeypatch)
    result = g.process_job(job)
    output, reference = tmp_path / "output", tmp_path / "reference"
    cache = reference / "overpasses_v5"
    (cache / job["satellite"]).mkdir(parents=True)
    p._write_csv_atomic(
        pd.DataFrame(result["records"], columns=g.RECORD_COLUMNS),
        p._cache_path(job, cache),
    )
    p._write_csv_atomic(
        pd.DataFrame(result["copies"], columns=g.COPY_COLUMNS),
        p._copies_path(job, cache),
    )
    monkeypatch.setattr(bench, "OUTPUT", output)
    monkeypatch.setattr(bench, "REFERENCES", {"2026-09-20": reference})
    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(g, "process_job", lambda *a: result)
    measured = bench.run_mode("2026-09-20", [job], "control", 2, output, (1,))
    assert measured["correctness"]["raw_identical_files"] == 2
    assert measured["traffic"]["requests"] == result["stats"]["requests"]
    assert measured["noaa20_bytes"] == result["stats"]["bytes_fetched"]
    assert bench.run_mode("2026-09-20", [job], "control", 2, output, (1,)) == measured
    with pytest.raises(ValueError, match="scope/options"):
        bench.run_mode("2026-09-20", [dict(job, sites=[])], "control", 2, output, (1,))
    assert len(bench.report(output)) == 1
    assert not measured["sampled"]
    smoke = bench.run_mode(
        "2026-09-20", [job], "control", 2, output, (1,), label="smoke", max_jobs=1
    )
    assert smoke["sampled"]
    smoke["wall_s"] = 1e7
    (output / "2026-09-20/smoke-w2-b16384/result.json").write_text(json.dumps(smoke))
    bench.report(output)
    expected_hours = 220 * measured["wall_s"] / 2 / 3600
    assert (
        f"Control projection: {expected_hours:.2f} h"
        in (output / "LIVE_REPORT.md").read_text()
    )


def test_spawn_safe_offline_pipeline_runner(tmp_path):
    import os
    import subprocess
    import sys

    driver = tmp_path / "spawn_check.py"
    driver.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "import multiprocessing as mp\n"
        f"sys.path.insert(0, {str(Path(p.__file__).parent)!r})\n"
        "import pipeline\n"
        "def work(job):\n"
        "    return dict(records=[], copies=[], error=None, stats=dict(bytes_fetched=5, requests=2, cpu_s=0.01, accounting_complete=True))\n"
        'if __name__ == "__main__":\n'
        '    mp.set_start_method("spawn", force=True)\n'
        "    pipeline.granule.process_job = work\n"
        '    jobs = [dict(satellite="SNPP", gid=f"g{i}", sites=[]) for i in range(3)]\n'
        '    cache = Path(__file__).parent / "cache"\n'
        "    pipeline.run_jobs(jobs, cache, workers=2)\n"
        '    assert pipeline.traffic_summary(jobs, cache)["bytes_fetched"] == 15\n'
        "    assert all(pipeline.cache_complete(j, cache) for j in jobs)\n"
    )
    subprocess.run(
        [sys.executable, str(driver)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
    )
