"""Network-free review-2 regressions, including the observed site-3802 spectrum."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import granule as g
import nightfire as nf
import pipeline as p
import validate_myc25 as v


def test_observed_3802_peak():
    # NOAA-20 d20261010_t2330408 (21,3125); review3-gpt61/peak.txt.
    bands = ["M10", "M7", "M8", "M11", "M12", "M13"]
    obs = [
        9.605476723518223,
        0.12164172530174255,
        4.8704103087075055,
        7.821446444024332,
        2.8952914235266864,
        1.5903347364643163,
    ]
    noise = [
        0.008896060746724067,
        0.007632865683929609,
        0.014036094260076327,
        0.005066581405699472,
        0.002404313521851856,
        0.003405299007027461,
    ]
    t, esf, chi, used, flag = nf.fit_planck(bands, obs, noise)
    assert {"M10", "M11"} & set(used)
    assert t == pytest.approx(1699, abs=30)
    # Footprint inferred from the raw-geolocation review's v3 RH, T and ESF.
    footprint = 3126.8685175813453 / nf.radiant_heat_mw(
        916.1618169511964, 0.03793270655238951, 1
    )
    assert nf.radiant_heat_mw(t, esf, footprint) == pytest.approx(124, rel=0.15)
    assert flag == "eliminated:M7" and chi < 9


@pytest.mark.parametrize(
    "bands,temp,reason",
    [
        (["M10", "M11"], 2800, "two_band_range"),
        (["M10", "M11", "M12"], 1100, "low_t"),
        (["M7", "M8"], 1800, None),
    ],
)
def test_temperature_and_anchor_guards(bands, temp, reason):
    obs = [1e-4 * nf.planck(nf.BAND_UM[b], temp) for b in bands]
    fit = nf.fit_planck(bands, obs, [0.00001] * len(bands))
    if reason:
        assert fit[4] == "pinned:" + reason and fit[0] == 1810
    else:
        assert fit[4] == "unfitted"


@pytest.mark.parametrize("n_bands", [3, 4, 6])
def test_elimination_stops_at_three_and_keeps_anchor(monkeypatch, n_bands):
    lengths = []

    def minimize(fun, x, **kwargs):
        variables = dict(
            zip(fun.__code__.co_freevars, (c.cell_contents for c in fun.__closure__))
        )
        n = len(variables["values"])
        lengths.append(n)
        return SimpleNamespace(
            x=[1800, np.log(1e-4)], fun=1000 * (n - 2) ** 2, success=True
        )

    monkeypatch.setattr(nf, "minimize", minimize)
    bands = ["M10", "M7", "M8", "M11", "M12", "M13"][:n_bands]
    fit = nf.fit_planck(bands, [1] * n_bands, [0.01] * n_bands)
    assert min(lengths) == 3 and fit[4] == "pinned:chi2"


def pixel(rh):
    return dict(
        rh_mw=rh,
        temp_k=1800,
        esf=0.001,
        area_m2=1e6,
        bands=["M10", "M11"],
        fit_flag="ok",
    )


def test_peak_rh_threshold_and_cap():
    lat, lon = np.meshgrid(np.arange(16) * 0.007, np.arange(8) * 0.007, indexing="ij")
    clusters = [dict(peak=(4, 3), pixels=[(4, 3), (4, 4), (4, 5), (4, 6)])]
    pixels = {(4, 3): pixel(10), (4, 4): pixel(100), (4, 5): pixel(8), (4, 6): {}}
    sources = g.summarize_clusters(
        clusters, pixels, lat, lon, np.array([[1, lat[4, 3], lon[4, 3]]])
    )
    assert sources[0]["rh_mw"] == 18
    assert sources[0]["rh_peak_mw"] == 10 and sources[0]["n_rh75"] == 2
    assert sources[0]["n_rh_cap_excluded"] == 1


def test_component_split():
    arr = np.array([[0, 10, 3, 2, 1, 2, 3, 11, 0]], dtype=float)
    clusters = g.candidate_clusters(arr > 0, arr, np.array([0]), np.array([4]))
    assert {c["peak"] for c in clusters} == {(0, 1), (0, 7)}
    assert sum(len(c["pixels"]) for c in clusters) == 7


def copies(distances):
    return [
        dict(
            peak=pixel(i + 1),
            peak_row=i * 16 + 4,
            peak_col=3,
            peak_lat=0,
            peak_lon=d / 111195,
            footprint=1e6,
            rh_mw=i + 1,
            rh_peak_mw=i + 1,
            n_rh75=1,
            n_hot=1,
            n_fitted=1,
            n_rh_cap_excluded=0,
            n_components_split=0,
        )
        for i, d in enumerate(distances)
    ]


@pytest.mark.parametrize(
    "distances,count", [([0, 999], 1), ([0, 1001], 2), ([0, 800, 1600], 1)]
)
def test_bowtie_one_pixel_and_chains(distances, count):
    tree = cKDTree(g._unit_xyz([0], [0]))
    sources = g.merge_scan_copies(copies(distances), tree)
    assert len(sources) == count
    assert sum(s["n_clusters"] for s in sources) == len(distances)
    summary = g.site_cluster_summary(sources, 0)
    assert summary["n_sources"] == count
    assert summary["rh_mw"] == (3 if len(distances) == 3 or count == 2 else 2)
    assert summary["peak_row"] == (len(distances) - 1) * 16 + 4


def test_dedupe_prefers_detection():
    frame = pd.DataFrame(
        dict(
            site_id=[1, 1],
            satellite=["N20"] * 2,
            time_utc=["2026-10-30T01:00:00", "2026-10-30T01:00:01"],
            dist_m=[0, 100],
            detected=[False, True],
        )
    )
    assert p.dedupe_overpasses(frame).detected.tolist() == [True]


def test_like_for_like_counts_pass_once_and_includes_cloud_detection():
    nights = pd.DataFrame(
        dict(site_id=[1, 2], lat=[0, 0], lon=[0, 0.005], night=["2026-10-29"] * 2)
    )
    over = pd.DataFrame(
        dict(
            site_id=[1, 2, 1],
            satellite=["N20"] * 3,
            time_utc=[
                "2026-10-30T01:00:00",
                "2026-10-30T01:00:01",
                "2026-10-30T02:00:00",
            ],
            dist_m=[0] * 3,
            sza=[150] * 3,
            cloud_mask=[0, 3, 0],
            cloud_clear_neighbors=[0] * 3,
            detected=[False, True, False],
        )
    )
    assert v.like_for_like_frequency(nights, over, 0, 0) == (1, 2)


def test_empty_satellite_coverage_is_skipped(monkeypatch):
    from datetime import date

    monkeypatch.setattr(p.granule_coverage, "cmr_granules", lambda *args: [])
    monkeypatch.setattr(p.granule_coverage, "site_slices", lambda *args: pd.DataFrame())
    sites = pd.DataFrame(dict(site_id=[1], lat=[0], lon=[0]))
    assert p.build_jobs(sites, date(2026, 11, 3), date(2026, 11, 3)) == []
