"""Synthetic, network-free regressions for cluster and nightly processing."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import granule as g
import pipeline as p


def test_border_window_clipping():
    hot = np.zeros((16, 8), bool)
    hot[0, 0] = hot[3, 3] = hot[15, 7] = True
    clusters = g.candidate_clusters(hot, hot.astype(float), [1], [1])
    assert {rc for c in clusters for rc in c["pixels"]} == {(0, 0), (3, 3)}


def test_selected_cluster_includes_pixels_beyond_search_window():
    hot = np.zeros((16, 12), bool)
    hot[5, 3:10] = True
    l10 = np.zeros(hot.shape)
    l10[5, 3:10] = np.arange(1, 8)
    clusters = g.candidate_clusters(hot, l10, np.array([5]), np.array([1]))
    assert len(clusters) == 1
    assert clusters[0]["peak"] == (5, 9)
    assert len(clusters[0]["pixels"]) == 7


def test_site_pixel_skips_fill_radiance():
    lat = np.zeros((3, 3))
    lon = np.tile(np.arange(3) * 0.001, (3, 1))
    l10 = np.ones((3, 3))
    l10[:, 0] = np.nan
    rows, cols, dist = g.site_pixels(lat, lon, l10, np.array([[1, 0, 0]]))
    assert cols.tolist() == [1]
    assert dist[0] == pytest.approx(111.2, abs=0.1)
    assert g.site_pixels(lat, lon, l10 * np.nan, np.array([[1, 0, 0]])) is None


def test_eight_connectivity_stops_at_scan_boundary():
    hot = np.zeros((32, 8), bool)
    hot[14, 2] = hot[15, 3] = hot[16, 4] = True
    clusters = g.candidate_clusters(hot, hot.astype(float), [15], [3])
    assert sorted(len(c["pixels"]) for c in clusters) == [1, 2]


def test_glow_rejected_by_nonhot_local_max_and_scan_clipping():
    hot = np.zeros((32, 8), bool)
    hot[15, 3] = True
    l10 = hot.astype(float)
    l10[14, 5] = 2  # brighter, despite not being in the hot mask
    assert g.candidate_clusters(hot, l10, [15], [3]) == []
    l10[14, 5] = 0
    l10[16, 3] = 10  # next scan must not veto this peak
    assert len(g.candidate_clusters(hot, l10, [15], [3])) == 1


def fitted(rh, temp=1800):
    return dict(rh_mw=rh, temp_k=temp, esf=0.001, area_m2=1000, bands=["M10", "M11"])


def test_cluster_nearest_site_rh75_and_bowtie_max():
    clusters = [
        dict(peak=(1, 2), pixels=[(1, 2), (1, 3), (1, 4)]),
        dict(peak=(17, 2), pixels=[(17, 2)]),
    ]
    pixels = {
        (1, 2): fitted(8, 1900),
        (1, 3): fitted(10),
        (1, 4): fitted(7.49),
        (17, 2): fitted(12, 2000),
    }
    sites = np.array([[1, 0, 0], [2, 0, 0.01]])  # both within 2 km
    lat, lon = np.zeros((32, 8)), np.full((32, 8), 0.001)
    assigned = g.summarize_clusters(clusters, pixels, lat, lon, sites)
    assert [c["site"] for c in assigned] == [0, 0]
    assert assigned[0]["rh_mw"] == pytest.approx(25.49)  # peak-anchored threshold
    summary = g.site_cluster_summary(assigned, 0)
    assert summary["n_clusters"] == 2
    assert summary["rh_mw"] == pytest.approx(37.49)  # sum distinct sources
    assert summary["temp_k"] == 1900  # winning cluster's M10 peak
    assert not g.site_cluster_summary(assigned, 1)["detected"]
    pixels[(1, 2)] = {}  # fitted wings cannot rescue an unconfirmed peak
    assert len(g.summarize_clusters(clusters, pixels, lat, lon, sites)) == 1
    lon[:] = 1
    assert not g.summarize_clusters(clusters, pixels, lat, lon, sites)


def observations():
    return pd.DataFrame(
        dict(
            site_id=[1] * 5,
            satellite=["SNPP"] * 5,
            time_utc=[
                "2026-10-30T01:00:00",
                "2026-10-30T01:00:02",
                "2026-10-30T01:01:00",
                "2026-10-30T01:02:00",
                "2026-10-30T01:03:00",
            ],
            dist_m=[200, 100, 100, 100, 100],
            sza=[150] * 5,
            cloud_mask=[3, 3, 3, 3, 0],
            cloud_clear_neighbors=[0, 0, 5, 6, 0],
            detected=[True] * 4 + [False],
            rh_mw=[100, 8, 4, 6, 0],
            temp_k=[1800] * 4 + [np.nan],
        )
    )


def test_dedupe_smallest_distance_and_satellite_site_separation():
    over = observations()
    out = p.dedupe_overpasses(over.sample(frac=1, random_state=7))
    assert len(out) == 4
    assert out.iloc[0].rh_mw == 8
    extra = over.iloc[[0]].copy()
    extra["satellite"] = "N20"
    assert len(p.dedupe_overpasses(pd.concat([over, extra]))) == 5
    extra["satellite"] = "SNPP"
    extra["site_id"] = 2
    assert len(p.dedupe_overpasses(pd.concat([over, extra]))) == 5
    over.loc[1, "time_utc"] = "2026-10-30T01:00:10"
    assert len(p.dedupe_overpasses(over)) == 5  # strictly less than ten seconds


def test_strict_vs_detclear_cloud_aggregation():
    sites = pd.DataFrame(dict(site_id=[1], lat=[0], lon=[0]))
    night = date(2026, 10, 29)
    out = p.site_nights(observations(), sites, night, night).iloc[0]
    assert out.n_overpasses == 4 and out.n_clear == 2 and out.n_detected == 1
    assert out.status == "flaring" and out.rh_mw == 3 and out.rh_max_mw == 6
    assert out.n_clear_detclear == 4 and out.rh_mw_detclear == 4.5
    assert out.bcm_per_year_detclear == pytest.approx(4.5 * p.BCM_PER_MW_YEAR)
    out = p.site_nights(observations().iloc[:3], sites, night, night).iloc[0]
    assert out.status == "flaring_cloud" and out.n_detected == 0
    assert out.n_detected_cloud == 2 and np.isnan(out.rh_mw)  # rows 1-2 after dedupe
    assert out.rh_mw_detclear == 6


def test_process_granule_cluster_integration(monkeypatch):
    lat, lon = np.meshgrid(np.arange(32) * 0.005, np.arange(12) * 0.005, indexing="ij")
    bands = {b: np.zeros(lat.shape) for b in ["M10", "M11", "M12", "M13"]}
    for b in bands:
        bands[b][5, 5] = 1
        bands[b][5, 6] = 0.9
    flags = {b: np.zeros(lat.shape, dtype=np.uint8) for b in bands}
    monkeypatch.setattr(g, "_read", lambda *a: ([lat, lon], 0))
    monkeypatch.setattr(g, "_band", lambda url, b: (bands[b], flags[b], 0))
    monkeypatch.setattr(
        g.nf, "solar_zenith_deg", lambda lat, lon, t: np.full(lat.shape, 150)
    )
    monkeypatch.setattr(
        g.nf,
        "zone_statistics",
        lambda arr, mask: (np.full(arr.shape[1], 0.1), np.ones(arr.shape[1])),
    )
    monkeypatch.setattr(g.nf, "local_background", lambda *a: (0, 0))
    monkeypatch.setattr(
        g.nf, "fit_planck", lambda bands, obs, noise: (1800, 0.001, 0, bands, "ok")
    )
    job = dict(
        satellite="SNPP",
        bucket="test",
        gid="test",
        start=datetime(2026, 10, 30),
        keys=dict(geo="geo", **{b: b for b in bands}),
        sites=[(1, lat[5, 5], lon[5, 5]), (2, lat[5, 7], lon[5, 7])],
    )
    records = g.process_granule(job)
    assert len(records) == 2
    assert set(records[0]) == set(g.RECORD_COLUMNS)
    assert records[0]["detected"] and records[0]["n_clusters"] == 1
    assert records[0]["n_fitted"] == 2
    assert not records[1]["detected"]
    # Raw saturated M10 must be removed; M11-M13 still confirm and fit.
    bands["M10"][5, 5] = g.nf.SATURATION["M10"]
    records = g.process_granule(job)
    assert records[0]["detected"] and "M10" not in records[0]["bands"]
    # Partial saturation flags remove a band even well below its full-pixel limit.
    bands["M10"][5, 5] = 1
    flags["M11"][5, 5] = 4
    records = g.process_granule(job)
    assert "M11" not in records[0]["bands"]
    assert "M10" in records[0]["bands"]
    # Optional M7/M8 contribute only when hot, and their noise reaches the fit.
    for name in ("M7", "M8"):
        bands[name] = np.zeros(lat.shape)
        flags[name] = np.zeros(lat.shape, dtype=np.uint8)
        job["keys"][name] = name
    bands["M7"][5, 5] = 1
    records = g.process_granule(job)
    assert "M7" in records[0]["bands"] and "M8" not in records[0]["bands"]
    # No candidates means no stage-2 reads (including optional bands).
    reads = []
    for arr in bands.values():
        arr[:] = 0

    def track_band(url, name):
        reads.append(name)
        return bands[name], flags[name], 0

    monkeypatch.setattr(g, "_band", track_band)
    g.process_granule(job)
    assert reads == ["M10"]


def test_main_creates_independent_output_paths(tmp_path, monkeypatch):
    sites = pd.DataFrame(dict(site_id=[1], lat=[0], lon=[0]))
    source = tmp_path / "sites.csv"
    sites.to_csv(source, index=False)
    output = tmp_path / "new" / "dev"
    seen = []
    monkeypatch.setattr(p, "build_jobs", lambda *a: ["synthetic_job"])

    def run_jobs(jobs, cache_dir, workers):
        seen.append((jobs, cache_dir, workers))
        return observations()

    monkeypatch.setattr(p, "run_jobs", run_jobs)
    result = p.main(
        start="2026-10-29",
        end="2026-10-29",
        workers=12,
        sites_csv=source,
        out_dir=output,
    )
    assert seen == [(["synthetic_job"], output / "overpasses_v4", 12)]
    assert (output / "overpasses_2026-10-29_2026-10-29.csv").exists()
    assert (output / "site_nights_2026-10-29_2026-10-29.csv").exists()
    assert result.iloc[0].rh_mw == 3
