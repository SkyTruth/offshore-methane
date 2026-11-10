"""Network-free review-3 counterexamples and v5 output contracts."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import granule as g
import nightfire as nf
import pipeline as p
import validate_myc25 as v
from test_v4 import copies


def test_guard_valid_trial_wins_counterexample():
    fit = nf.fit_planck(
        ["M10", "M7", "M11", "M12"],
        [3.617259, 3.569423, 10.438807, 6.260087],
        [0.001] * 4,
    )
    assert fit[0] == pytest.approx(1696.7, abs=1)
    assert fit[3] == ["M7", "M11", "M12"]
    assert fit[4] == "eliminated:M10"


@pytest.mark.parametrize("bands", [["M10", "M7", "M8"], ["M7", "M8"], ["M11", "M8"]])
def test_two_swir_required(bands):
    obs = [1e-4 * nf.planck(nf.BAND_UM[b], 1800) for b in bands]
    diagnostic = {}
    fit = nf.fit_planck(bands, obs, [0.001] * len(bands), diagnostics=diagnostic)
    assert fit[4].startswith("pinned:") or fit[4] == "unfitted"
    if fit[4].startswith("pinned:"):
        assert diagnostic["t_fit_rejected"] == pytest.approx(1800, abs=1)
        assert diagnostic["band_rule_changed"]


@pytest.mark.parametrize("saddle,count", [(4, 2), (6, 1), (5, 2)])
def test_saddle_prominence(saddle, count):
    arr = np.array([[10, 8, 7, saddle, 7, 8, 11]], float)
    clusters = g.candidate_clusters(arr > 0, arr, np.array([0]), np.array([3]))
    assert len(clusters) == count
    if count == 2:
        assert clusters[1]["prominence_ratio"] == pytest.approx(saddle / 10)


def test_same_scan_bridge_stays_separate():
    cs = copies([0, 1600, 800])
    cs[1]["peak_row"] = 5
    cs[2]["peak_row"] = 20
    for c in cs:
        c["rh_mw"] = 10
    groups = g.merge_scan_copies(cs, cKDTree(g._unit_xyz([0], [0])))
    assert sorted(x["n_clusters"] for x in groups) == [1, 2]
    assert cs[0]["group_id"] != cs[1]["group_id"]


@pytest.mark.parametrize(
    "dcol,ratio,distance,count", [(2, 1, 800, 2), (1, 3, 800, 2), (1, 2.9, 1400, 1)]
)
def test_copy_pair_guards(dcol, ratio, distance, count):
    cs = copies([0, distance])
    cs[1]["peak_col"] += dcol
    cs[1]["rh_mw"] = ratio
    groups = g.merge_scan_copies(cs, cKDTree(g._unit_xyz([0], [0])))
    assert len(groups) == count


def test_centroid_assignment_mean_and_max():
    cs = copies([0, 1400])
    cs[0]["rh_mw"], cs[1]["rh_mw"] = 10, 12
    tree = cKDTree(g._unit_xyz([0, 0], [700 / 111195, 1400 / 111195]))
    groups = g.merge_scan_copies(cs, tree)
    assert groups[0]["site"] == 0
    assert groups[0]["rh_mw"] == 11
    assert cs[0]["group_max_rh_mw"] == 12
    assert groups[0]["peak_row"] == cs[1]["peak_row"]


def test_cache_needs_both_tables(tmp_path):
    job = dict(satellite="SNPP", gid="example")
    path = p._cache_path(job, tmp_path)
    path.parent.mkdir()
    path.write_text(",".join(g.RECORD_COLUMNS))
    assert not p.cache_complete(job, tmp_path)
    p._copies_path(job, tmp_path).write_text(",".join(g.COPY_COLUMNS))
    assert p.cache_complete(job, tmp_path)
    assert {
        "t_fit_rejected",
        "prominence_ratio",
        "centroid_lat",
        "pair_drow",
        "assigned_site",
    } <= set(g.COPY_COLUMNS)


def test_myc_radius_filters_group_centroid_and_cloud_definitions():
    nights = pd.DataFrame(dict(site_id=[1], lat=[0], lon=[0], night=["2026-10-29"]))
    over = pd.DataFrame(
        dict(
            site_id=[1],
            satellite=["N20"],
            granule=["g"],
            time_utc=["2026-10-30T01:00:00"],
            dist_m=[0],
            sza=[150],
            cloud_mask=[3],
            cloud_clear_neighbors=[0],
            detected=[True],
        )
    )
    cs = pd.DataFrame(
        dict(
            satellite=["N20"],
            granule=["g"],
            time_utc=over.time_utc,
            centroid_lat=[0],
            centroid_lon=[0.03],
        )
    )
    assert v.like_for_like_frequency(nights, over, 0, 0, cs) == (0, 0)
    cs.centroid_lon = 0.01
    assert v.like_for_like_frequency(nights, over, 0, 0, cs) == (1, 1)
    assert v.like_for_like_frequency(nights, over, 0, 0, cs, sensitivity=False) == (
        0,
        0,
    )
    over.cloud_clear_neighbors = 6
    assert v.like_for_like_frequency(nights, over, 0, 0, cs, sensitivity=False) == (
        1,
        1,
    )


def test_rh75_excludes_remote_and_other_scan_pixels():
    from test_v4 import pixel

    lat, lon = np.meshgrid(np.arange(32) * 0.007, np.arange(8) * 0.007, indexing="ij")
    clusters = [dict(peak=(15, 3), pixels=[(15, 3), (15, 4), (15, 5), (16, 3)])]
    pixels = {rc: pixel(10) for rc in clusters[0]["pixels"]}
    sink = []
    result = g.summarize_clusters(
        clusters, pixels, lat, lon, np.array([[1, lat[15, 3], lon[15, 3]]]), sink
    )
    assert result[0]["rh_mw"] == 20
    assert sink[0]["n_rh75"] == 2


def test_copy_output_schema_integration(monkeypatch):
    # Exercise the no-candidate return contract as well as the cached schema.
    from datetime import datetime

    lat, lon = np.meshgrid(np.arange(16) * 0.007, np.arange(10) * 0.007, indexing="ij")
    monkeypatch.setattr(g, "_read", lambda *a: ([lat, lon], 0))
    monkeypatch.setattr(
        g,
        "_band",
        lambda *a: (np.zeros(lat.shape), np.zeros(lat.shape, dtype=np.uint8), 0),
    )
    monkeypatch.setattr(
        nf, "solar_zenith_deg", lambda lat, lon, t: np.full(lat.shape, 150)
    )
    job = dict(
        satellite="SNPP",
        gid="test",
        bucket="test",
        start=datetime(2026, 10, 30),
        keys=dict(geo="geo", M10="m10"),
        sites=[(1, lat[5, 5], lon[5, 5])],
    )
    records, cs = g.process_granule(job)
    assert len(records) == 1 and not cs
    assert not records[0]["detected"]
    assert list(pd.DataFrame(cs, columns=g.COPY_COLUMNS)) == g.COPY_COLUMNS


def test_good_fit_failing_guard_pins_without_elimination():
    bands = ["M7", "M8", "M10", "M11", "M12", "M13"]
    obs = [1e-5 * nf.planck(nf.BAND_UM[b], 1100) for b in bands]
    t, _, chi, used, flag = nf.fit_planck(bands, obs, [0.01] * 6)
    assert chi <= nf.FIT_CHI2_MAX
    assert flag == "pinned:low_t" and used == ["M10", "T1810"] and t == 1810


def test_atomic_cache_write_and_manifest(tmp_path):
    job = dict(satellite="SNPP", gid="g1", sites=[(1, 0, 0)])
    cache = tmp_path / "cache"
    assert not p.cache_complete(job, cache)
    (cache / "SNPP").mkdir(parents=True)
    p._write_csv_atomic(
        pd.DataFrame(columns=g.COPY_COLUMNS), p._copies_path(job, cache)
    )
    assert not p.cache_complete(job, cache)  # site file still missing
    p._write_csv_atomic(pd.DataFrame({"a": [1]}), p._cache_path(job, cache))
    assert p.cache_complete(job, cache)
    assert not list((cache / "SNPP").glob("*.tmp"))
    other = dict(satellite="SNPP", gid="g2", sites=[])
    summary = p.write_manifest([job, other], cache, tmp_path / "jobs.csv")
    assert summary["expected"] == 2 and summary["complete"] == 1
    assert summary["missing"] == [["SNPP", "g2"]]


def test_suppressed_maxima_exported():
    arr = np.array([[0, 10, 9, 8, 9.5, 0]], dtype=float)
    clusters = g.candidate_clusters(arr > 0, arr, np.array([0]), np.array([2]))
    assert len(clusters) == 1
    c = clusters[0]
    assert c["n_suppressed_maxima"] == 1
    assert c["suppressed_max_m10"] == 9.5
    assert c["suppressed_saddle_ratio"] == pytest.approx(8 / 9.5)


def test_get_with_retries(monkeypatch):
    import requests

    import nodd

    monkeypatch.setattr(nodd.time, "sleep", lambda s: None)

    class Session:
        def __init__(self, outcomes):
            self.outcomes, self.calls = list(outcomes), 0

        def get(self, url, **kwargs):
            self.calls += 1
            out = self.outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            resp = requests.Response()
            resp.status_code = out
            return resp

    s = Session([requests.ReadTimeout(), 503, 200])
    assert nodd.get_with_retries(s, "u").status_code == 200 and s.calls == 3
    s = Session([404])
    with pytest.raises(requests.HTTPError):
        nodd.get_with_retries(s, "u")
    assert s.calls == 1
    s = Session([requests.ConnectionError()] * 4)
    with pytest.raises(requests.ConnectionError):
        nodd.get_with_retries(s, "u")


def test_cmr_cache(monkeypatch, tmp_path):
    from datetime import date, timedelta

    import granule_coverage as gc

    calls = []

    class Resp:
        def json(self):
            return {"feed": {"entry": [{"id": len(calls)}]}}

    def fake_get(session, url, **kwargs):
        calls.append(kwargs["params"]["day_night_flag"])
        return Resp()

    monkeypatch.setattr(gc.nodd, "get_with_retries", fake_get)
    old = (date.today() - timedelta(days=10)).isoformat()
    first = gc.cmr_granules("VJ203MOD", old, tmp_path)
    assert gc.cmr_granules("VJ203MOD", old, tmp_path) == first and len(calls) == 2
    recent = date.today().isoformat()
    gc.cmr_granules("VJ203MOD", recent, tmp_path)
    gc.cmr_granules("VJ203MOD", recent, tmp_path)
    assert len(calls) == 6 and not (tmp_path / "VJ203MOD" / f"{recent}.json").exists()


def test_build_jobs_oldest_first(monkeypatch):
    from datetime import date, datetime, timedelta

    import pipeline as p

    sites = pd.DataFrame(dict(site_id=[1], lat=[0.0], lon=[0.0]))
    base = datetime(2026, 11, 3, 1)
    offsets = {"SNPP": 30, "NOAA-20": 20, "NOAA-21": 10}

    def slices(sites_, granules):
        return pd.DataFrame(
            dict(
                slice_start=[pd.Timestamp("2026-11-03T00:00Z")],
                slice_end=[pd.Timestamp("2026-11-04T23:59Z")],
                site_ids=["1"],
            )
        )

    def index(sat, day, session, source="aws"):
        out = {}
        for k in (0, 60):
            start = base + timedelta(minutes=offsets[sat] + k)
            gid = start.strftime("d%Y%m%d_t%H%M%S0")
            out[gid] = dict(
                start=start,
                end=start + timedelta(seconds=86),
                **{r: "x" for r in p.REQUIRED},
            )
        return out

    monkeypatch.setattr(p.granule_coverage, "cmr_granules", lambda *a: [{}])
    monkeypatch.setattr(p.granule_coverage, "site_slices", slices)
    monkeypatch.setattr(p.nodd, "granule_index", index)
    jobs = p.build_jobs(sites, date(2026, 11, 3), date(2026, 11, 3))
    starts = [j["start"] for j in jobs]
    assert starts == sorted(starts) and jobs[0]["satellite"] == "NOAA-21"
