"""Cold certificates: orbital bounds, border widening, night mask and fallbacks."""

from dataclasses import asdict
from datetime import datetime, timedelta

import h5py
import numpy as np
import pytest
from sgp4.api import Satrec

import granule as g
import orbit
import speedup as s
from test_speedup import scene, tables

TLE = (
    "1 43013U 17073A   24200.00000000  .00000130  00000-0  61657-4 0  3142",
    "2 43013 098.8090 239.0409 0000998 345.6244 269.0599 14.19540862776996",
)


def test_filename_endpoints_cross_midnight_and_deciseconds():
    job = dict(keys=dict(geo="GMTCO_j01_d20260718_t2359555_e0001200_b1_c0.h5"))
    first, last = orbit.interval(job)
    assert first == datetime(2026, 7, 18, 23, 59, 55, 500000)
    assert last == datetime(2026, 7, 19, 0, 1, 20)
    assert (last - first).total_seconds() == 84.5
    with pytest.raises(ValueError, match="timing"):
        orbit.interval(dict(gid="invalid"))


def test_tle_nearest_epoch_stale_and_missing(tmp_path):
    folder = tmp_path / "noaa-20"
    folder.mkdir()
    for day in (200, 201):
        with h5py.File(folder / f"{day}.h5", "w") as h:
            text = "\n".join(TLE).replace("24200.000", f"24{day:03d}.000")
            h.create_dataset(
                "All_Data/TLE-AUX_All/Dataset_Array_0",
                data=np.frombuffer(text.encode(), dtype="u1"),
            )
    job = dict(satellite="NOAA-20", gid="d20240719_t0000000")
    picked = orbit.tle(job, tmp_path)
    assert picked.epochdays == 201
    job["gid"] = "d20240819_t0000000"
    with pytest.raises(ValueError, match="3 days"):
        orbit.tle(job, tmp_path)
    job["satellite"] = "NOAA-21"
    with pytest.raises(ValueError, match="no local TLE"):
        orbit.tle(job, tmp_path)


def test_orbit_forward_inverse_and_outside_whitelist(monkeypatch):
    satellite = Satrec.twoline2rv(*TLE)
    start = datetime(2024, 7, 18, 0, 0)
    job = dict(satellite="NOAA-20", gid="d20240718_t0000000")
    monkeypatch.setattr(orbit, "tle", lambda *a: satellite)
    r, _ = orbit.state(satellite, start, np.array([42.0]))
    # Geodetic conversion of the sub-satellite point for an independent nadir query.
    point = r[0]
    lon = np.degrees(np.arctan2(point[1], point[0]))
    lat = np.degrees(
        np.arctan2(point[2], np.hypot(point[0], point[1]) * (1 - 0.00669438))
    )
    job["sites"] = [(1, lat, lon), (2, -lat, (lon + 360) % 360 - 180)]
    pred = orbit.predict(job, "unused")
    assert pred["eligible"][0]
    assert abs(pred["seconds"][0] - 42) < 0.2
    assert abs(pred["col"][0] - 1600) < 5
    assert not pred["eligible"][1]


@pytest.mark.parametrize("offset", [32, 45, 63])
def test_sparse_border_widens_once_then_falls_back(tmp_path, offset):
    rr, cc = np.indices((256, 400))
    lat = (rr * 0.006).astype("f4")
    lon = (cc * 0.006).astype("f4")
    path = tmp_path / "jrr.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("Latitude", data=lat, chunks=(32, 40), compression="gzip")
        h.create_dataset("Longitude", data=lon, chunks=(32, 40), compression="gzip")
    row, col = 100 + offset, 200
    sites = np.array([(1, float(lat[row, col]), float(lon[row, col]))], dtype=object)
    with h5py.File(path) as h:
        sparse = s.SparseGeo(h, lat.shape)
        result = sparse.nearest(
            np.ones_like(lat),
            sites,
            dict(row=[100], col=[200]),
            s.Options(row_margin=32, col_margin=32),
        )
        if offset == 63:
            assert result is None and sparse.reason == "border"
        else:
            assert sparse.widened and result is not None
            expected = g.site_pixels(lat, lon, np.ones_like(lat), sites)
            for actual, reference in zip(result, expected):
                np.testing.assert_array_equal(actual, reference)
            sparse.ensure([(row, col)], halo=2)
            assert np.array_equal(
                sparse.lat[row - 2 : row + 3, col - 2 : col + 3],
                lat[row - 2 : row + 3, col - 2 : col + 3],
            )


def test_sparse_rejects_finite_m10_geofill(tmp_path, monkeypatch):
    _, _, lat, _ = scene(tmp_path, monkeypatch, fills=True)
    with h5py.File(tmp_path / "cloud.h5") as h:
        sparse = s.SparseGeo(h, lat.shape)
        sites = np.array([(1, 0.03, 0.21)])
        assert (
            sparse.nearest(
                np.ones_like(lat), sites, dict(row=[5], col=[35]), s.Options()
            )
            is None
        )
        assert sparse.reason == "geo_fill"


def test_orbit_empty_swath_skips_without_opening_files(tmp_path, monkeypatch):
    job, session, _, _ = scene(tmp_path, monkeypatch)
    monkeypatch.setattr(g.orbit, "predict", lambda *a: dict(eligible=np.zeros(2, bool)))
    result = g.process_job(
        dict(job, speedup=asdict(s.Options(orbit_whitelist=True, tle_dir="unused")))
    )
    assert result["error"] is None and result["stats"]["skipped"]
    assert result["stats"]["bytes_fetched"] == 0 and session.calls == []


def test_absent_tle_keeps_control_results(tmp_path, monkeypatch):
    job, _, _, _ = scene(tmp_path, monkeypatch)
    control = g.process_job(job)
    result = g.process_job(
        dict(job, speedup=dict(orbit_whitelist=True, windowed_geo=True))
    )
    assert result["stats"]["orbit"] == "fallback_tle"
    assert result["stats"]["geo_mode"] == "fallback_prediction"
    assert tables(result) == tables(control)


def test_night_certificate_uses_margin_and_rejects_malformed_ring(
    tmp_path, monkeypatch
):
    job, _, _, _ = scene(tmp_path, monkeypatch)
    with h5py.File(tmp_path / "geo.h5") as h:
        attrs = s.header(h)
    assert s.night_certificate(attrs, job["start"])[0]
    assert not s.night_certificate(attrs, job["start"] + timedelta(hours=12))[0]
    assert not s.night_certificate(attrs, job["start"], margin=90)[0]
    attrs["G-Ring_Latitude"] = [float("nan")] * 4
    assert s.night_certificate(attrs, job["start"]) == (False, None)
    assert not s.footprint_skip(job, attrs)


def test_reference_scope_adds_adjacent_cmr_sites_without_reading_results(monkeypatch):
    import bench_speedup as bench
    import pandas as pd

    start = datetime(2026, 9, 19, 23, 59)
    job = dict(
        satellite="NOAA-20",
        gid="d20260919_t2359000",
        start=start,
        sites=[(1, 0.0, 0.0)],
    )
    sites = pd.DataFrame(dict(site_id=[1, 2], lat=[0.0, 1.0], lon=[0.0, 1.0]))
    calls = []

    def cmr(short, day, **kwargs):
        calls.append(day)
        return []

    monkeypatch.setattr(bench.granule_coverage, "cmr_granules", cmr)
    monkeypatch.setattr(
        bench.granule_coverage,
        "site_slices",
        lambda *a: pd.DataFrame(
            dict(
                slice_start=["2026-09-19T23:59:00Z"],
                slice_end=["2026-09-20T00:00:00Z"],
                site_ids=["2"],
            )
        ),
    )
    result = bench.reference_scope([job], "2026-09-20", sites, "unused")
    assert result[0]["sites"] == [(1, 0.0, 0.0), (2, 1.0, 1.0)]
    assert set(calls) == {"2026-09-19", "2026-09-22"}


def test_driver_hard_deadline_terminates_process_group(tmp_path, monkeypatch):
    import json
    import subprocess
    import bench_speedup as bench

    monkeypatch.setattr(bench, "OUTPUT", tmp_path)
    killed = []

    class Child:
        pid = 12345
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("offline fake", timeout)
            return 0

    def start(command, **kwargs):
        assert kwargs["start_new_session"]
        assert Path(command[1]).exists()
        return Child()

    from pathlib import Path

    monkeypatch.setattr(bench.subprocess, "Popen", start)
    monkeypatch.setattr(bench.os, "killpg", lambda pid, sig: killed.append(pid))
    result = bench.run_bounded_protocol(tmp_path, budget_s=0.01)
    assert result["budget_exhausted"] and killed == [12345]
    assert (
        json.loads((tmp_path / "budget_stop.json").read_text())["accounting_complete"]
        is False
    )


def test_prediction_inside_margins_stays_windowed(tmp_path, monkeypatch):
    job, session, _, _ = scene(tmp_path, monkeypatch)
    control = g.process_job(job)
    monkeypatch.setattr(
        g.orbit,
        "predict",
        lambda *a: dict(
            row=np.array([5.0, -1.0]),
            col=np.array([35.0, 70.0]),
            eligible=np.ones(2, bool),
        ),
    )
    result = g.process_job(dict(job, speedup=dict(windowed_geo=True, tle_dir="unused")))
    assert tables(result) == tables(control)
    assert result["stats"]["geo_mode"] == "windowed"


def _with_outside_site(job, lat, lon):
    # About 11 km beyond the last row: no pixel within MAX_SITE_PIXEL_M.
    far = (3, float(lat[-1, 50]) + 0.1, float(lon[-1, 50]))
    return dict(job, sites=[*job["sites"], far])


def test_outside_site_is_certified_miss_or_left_out(tmp_path, monkeypatch):
    job, _, lat, lon = scene(tmp_path, monkeypatch)
    job = _with_outside_site(job, lat, lon)
    control = g.process_job(job)
    for outside_row in (40.0, 500.0):  # within the margin, then far beyond it
        monkeypatch.setattr(
            g.orbit,
            "predict",
            lambda *a, r=outside_row: dict(
                row=np.array([5.0, 20.0, r]),
                col=np.array([35.0, 70.0, 50.0]),
                eligible=np.ones(3, bool),
            ),
        )
        result = g.process_job(
            dict(job, speedup=dict(windowed_geo=True, tle_dir="unused"))
        )
        assert tables(result) == tables(control)
        assert result["stats"]["geo_mode"] == "windowed"
        far = result["stats"]["geo_far_outside"]
        misses = result["stats"]["geo_certified_misses"]
        assert (far, misses) == ((0, 1) if outside_row == 40.0 else (1, 0))


def test_all_sites_far_outside_returns_no_rows(tmp_path, monkeypatch):
    job, _, lat, lon = scene(tmp_path, monkeypatch)
    job = dict(_with_outside_site(job, lat, lon))
    job["sites"] = job["sites"][-1:]
    control = g.process_job(job)
    monkeypatch.setattr(
        g.orbit,
        "predict",
        lambda *a: dict(
            row=np.array([500.0]), col=np.array([50.0]), eligible=np.ones(1, bool)
        ),
    )
    result = g.process_job(dict(job, speedup=dict(windowed_geo=True, tle_dir="unused")))
    assert tables(result) == tables(control) and not result["records"]
    assert result["stats"]["geo_mode"] == "all_sites_excluded"


def test_margins_must_cover_measured_error():
    assert s.margins_cover_error(s.Options())
    assert not s.margins_cover_error(s.Options(row_margin=40))


def test_protocol_stops_at_first_identity_failure(tmp_path, monkeypatch):
    import json
    import bench_speedup as bench

    monkeypatch.setattr(bench, "OUTPUT", tmp_path)
    monkeypatch.setattr(bench, "tier1_ids", lambda *a: (1,))
    monkeypatch.setattr(bench, "_prepare", lambda *a: [])
    monkeypatch.setattr(bench, "report", lambda *a: [])
    called = []

    def failed(night, jobs, mode, *args):
        called.append(mode)
        raise RuntimeError("identity failed")

    monkeypatch.setattr(bench, "run_mode", failed)
    with pytest.raises(RuntimeError, match="identity"):
        bench.main(action="protocol", out_dir=tmp_path)
    assert called == ["control"]
    assert (
        json.loads((tmp_path / "failure_stop.json").read_text())["run"][1] == "control"
    )


def test_loaded_day_pixel_invalidates_night_certificate(tmp_path, monkeypatch):
    job, session, _, _ = scene(tmp_path, monkeypatch)
    control = g.process_job(job)
    with h5py.File(tmp_path / "cloud.h5", "a") as h:
        h["Longitude"][20, 70] = 180.0
    session.files["cloud"] = (tmp_path / "cloud.h5").read_bytes()
    result = g.process_job(dict(job, speedup=dict(windowed_geo=True, tle_dir="unused")))
    assert result["stats"]["geo_mode"] == "fallback_loaded_day_pixel"
    assert tables(result) == tables(control)


def test_antipodal_ring_falls_back():
    attrs = {"G-Ring_Latitude": [0, 0, 0, 0], "G-Ring_Longitude": [0, 90, 180, -90]}
    assert not s.footprint_skip(dict(sites=[(1, 0, 0)]), attrs)
    assert s.night_certificate(attrs, datetime(2026, 1, 1)) == (False, None)


def test_backfill_passes_source_and_speedup(tmp_path, monkeypatch):
    import json

    import backfill

    seen = {}
    monkeypatch.setattr(backfill, "warm_cmr_cache", lambda *a, **k: [])
    monkeypatch.setattr(backfill.pipeline, "main", lambda **k: seen.update(k))
    monkeypatch.setattr(backfill.validate_myc25, "main", lambda **k: None)
    (tmp_path / "run_summary_2026-01-01_2026-01-02.json").write_text(
        json.dumps(dict(expected=1, complete=1))
    )
    backfill.main(
        "2026-01-01",
        "2026-01-02",
        out_dir=tmp_path,
        source="azure",
        speedup=dict(header_prefilter=True),
    )
    assert seen["source"] == "azure"
    assert seen["speedup"] == dict(header_prefilter=True)


def test_bounded_completed_limits_queue_and_keeps_order(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import pipeline

    monkeypatch.setattr(pipeline.granule, "process_job", lambda job: job["i"])

    class CountingPool(ThreadPoolExecutor):
        def __init__(self):
            super().__init__(2)
            self.submitted, self.in_flight, self.peak = [], 0, 0

        def submit(self, fn, job):
            self.submitted.append(job["i"])
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            return super().submit(fn, job)

    jobs = [dict(i=i) for i in range(100)]
    with CountingPool() as pool:
        seen = []
        for fut, job in pipeline._bounded_completed(pool, jobs, 5):
            pool.in_flight -= 1
            assert fut.result() == job["i"]
            seen.append(job["i"])
    assert sorted(seen) == list(range(100))
    assert pool.submitted == list(range(100))
    assert pool.peak <= 5
