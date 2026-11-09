"""Network-free fit, quality-flag and validation regressions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import granule as g
import nightfire as nf
import nodd
import validate_myc25 as v

BANDS = ["M7", "M8", "M10", "M11", "M12", "M13"]


def spectrum(temp=1800, esf=1e-5):
    return np.array([esf * nf.planck(nf.BAND_UM[b], temp) for b in BANDS])


def test_weighting_reduces_influence_of_noisy_band():
    obs = spectrum()
    obs[-1] *= 1.5
    weighted = nf.fit_planck(BANDS, obs, [0.01] * 5 + [100])
    unweighted = nf.fit_planck(BANDS, obs, [1] * 6)
    assert abs(weighted[0] - 1800) < abs(unweighted[0] - 1800)
    assert weighted[4] == "ok"


def test_normalised_negative_residual_elimination():
    obs = spectrum(esf=1e-3)
    obs[3] *= 0.3  # far outside the 10% model error
    result = nf.fit_planck(BANDS, obs, [0.01] * 6)
    assert result[4] == "eliminated:M11"
    assert result[0] == pytest.approx(1800, rel=0.01)


@pytest.mark.parametrize("temp,esf", [(5000, 1e-5), (400, 1e-5), (1800, 2)])
def test_physical_rejection_and_pin(temp, esf):
    result = nf.fit_planck(BANDS, spectrum(temp, esf), [0.01] * 6)
    assert result[4].startswith("pinned:") or result[4] == "unfitted"
    if result[4].startswith("pinned:"):
        assert result[0] == 1810 and 0 < result[1] <= 1


def test_nonconvergence_pins_only_usable_m10(monkeypatch):
    monkeypatch.setattr(
        nf,
        "minimize",
        lambda *a, **k: SimpleNamespace(x=[1800, np.log(1e-5)], fun=0, success=False),
    )
    assert nf.fit_planck(BANDS, spectrum(), [0.01] * 6)[4] == "pinned:nonconvergence"
    assert nf.fit_planck(["M12", "M13"], [0.1, 0.1], [0.01] * 2)[4] == "unfitted"


def test_high_final_chi_square_rejected(monkeypatch):
    monkeypatch.setattr(
        nf,
        "minimize",
        lambda *a, **k: SimpleNamespace(x=[1800, np.log(1e-5)], fun=100, success=True),
    )
    assert nf.fit_planck(BANDS, spectrum(), [0.01] * 6)[4] == "pinned:chi2"


def test_zone_noise_uses_same_clipped_sample():
    arr = np.tile(np.arange(640, dtype=float), (2, 1))
    threshold, noise = nf.zone_statistics(arr, np.ones_like(arr, dtype=bool))
    assert threshold[0] == pytest.approx(arr.mean() + 4 * arr.std())
    assert noise[0] == pytest.approx(arr.std())
    assert np.array_equal(
        threshold, nf.zone_thresholds(arr, np.ones_like(arr, dtype=bool))
    )


def test_qf1_saturation_bits_include_partial():
    flags = np.array([0, 1, 2, 4, 8, 12, 16, 64], dtype=np.uint8)
    assert (flags & g.QF1_SATURATION_MASK != 0).tolist() == [False] * 3 + [True] * 3 + [
        False
    ] * 2
    assert nodd.PRODUCTS["M7"] == "VIIRS-M7-SDR"
    assert nodd.PRODUCTS["M8"] == "VIIRS-M8-SDR"


def inputs():
    nights = pd.DataFrame(
        dict(
            site_id=[1, 1, 2],
            lat=[0, 0, 0],
            lon=[0, 0, 0.005],
            night=["2026-10-29", "2026-10-30", "2026-10-29"],
        )
    )
    over = pd.DataFrame(
        dict(
            site_id=[1] * 6,
            satellite=["N20"] * 6,
            time_utc=[
                "2026-10-30T01:00:00",
                "2026-10-30T01:00:01",
                "2026-10-30T02:00:00",
                "2026-10-31T01:00:00",
                "2026-10-31T02:00:00",
                "2026-10-31T03:00:00",
            ],
            dist_m=[0, 10, 0, 0, 0, 0],
            sza=[150] * 6,
            cloud_mask=[0, 0, 3, 0, 0, 0],
            cloud_clear_neighbors=[0] * 6,
            detected=[True, True, True, False, False, False],
            rh_mw=[8, 999, 999, 0, 0, 0],
        )
    )
    return nights, over


def test_pooled_mean_dedupes_and_excludes_cloudy_detection():
    nights, over = inputs()
    result = v.pooled_observations(nights, over).set_index("site_id").loc[1]
    assert result.clear_overpasses == 4 and result.detections == 1
    assert result.rh_mw == 2  # mean of nightly means would be 4
    assert result.bcm_per_year == pytest.approx(0.0232)
    assert result.det_freq == 0.25
    assert result.nights_flaring == 1 and result.nights_clear == 2


def test_one_to_one_matching_nearest_pairs_first():
    nights, over = inputs()
    myc = pd.DataFrame(
        dict(
            flare_id=[11, 12],
            lat=[0, 0],
            lon=[0.001, 0.02],
            regime=["persistent"] * 2,
            bcm=[1, 2],
            upstream_class=["oil"] * 2,
            country_iso=["XX"] * 2,
            n_det=[5, 6],
            n_clear=[10, 10],
        )
    )
    result = v.match_sites(nights, myc, 2000, over)
    assert result.myc_flare_id.tolist() == [11, 12]
    assert result.myc_flare_id.is_unique
    # The second site is 1.67 km from the remaining MYC site.
    assert result.myc_dist_m.iloc[1] > 1000
    summary = v.validation_summary(result)
    assert summary["det_freq"] == dict(ours=0.25, myc=0.5, n=1)
    assert summary["regions"] == [["Gulf of Guinea", pytest.approx(0.0232), 1.0, 1]]


@pytest.mark.parametrize("band,dtype", [("M7", np.float32), ("M8", np.uint16)])
def test_band_reads_radiance_and_qf1(tmp_path, monkeypatch, band, dtype):
    import h5py

    path = tmp_path / "band.h5"
    with h5py.File(path, "w") as h:
        group = h.create_group(f"All_Data/VIIRS-{band}-SDR_All")
        group["Radiance"] = np.array([[10, 20]], dtype=dtype)
        group["RadianceFactors"] = [0.5, 1]
        group["QF1_VIIRSMBANDSDR"] = np.array([[0, 4]], dtype=np.uint8)

    def local_file(*args):
        f = path.open("rb")
        f.bytes_fetched = 0
        return f

    monkeypatch.setattr(nodd, "HTTPRangeFile", local_file)
    radiance, flags, _ = g._band("synthetic", band)
    assert radiance.tolist() == ([[10, 20]] if band == "M7" else [[6, 11]])
    assert flags.tolist() == [[0, 4]]


def test_viewer_validation_keeps_contract_and_uses_pooled_values(tmp_path, monkeypatch):
    from datetime import date

    import build_viewer as viewer
    import pipeline

    nights, over = inputs()
    over["satellite"] = "NOAA-20"
    over["granule"] = "synthetic"
    over["bytes_granule"] = 0
    over["temp_k"] = 1800
    sites = nights[["site_id", "lat", "lon"]].drop_duplicates()
    sites["n_structures"] = 1
    sites["structure_ids"] = "test"
    nights = pipeline.site_nights(over, sites, date(2026, 10, 29), date(2026, 10, 30))
    tag = "2026-10-29_2026-10-30"
    nights.to_csv(tmp_path / f"site_nights_{tag}.csv", index=False)
    over.to_csv(tmp_path / f"overpasses_{tag}.csv", index=False)
    sites.to_csv(tmp_path / "gfw_oil_sites.csv", index=False)
    myc = pd.DataFrame(
        dict(
            flare_id=["m1"],
            lat=[0],
            lon=[0.001],
            regime=["persistent"],
            bcm=[0.0232],
            upstream_class=["oil"],
            country_iso=["XX"],
            n_det=[5],
            n_clear=[10],
        )
    )
    monkeypatch.setattr(viewer.validate_myc25, "load_myc", lambda year: myc)
    nights, over = viewer.load_segments([(tmp_path, tag)])
    data, passes = viewer.build_data(nights, over, tmp_path)
    meta = data["meta"]["validation"]
    assert set(passes) <= {int(s) for s in nights["site_id"]}
    assert {
        "matched",
        "observed",
        "by_regime",
        "log_r",
        "n_both",
        "unmatched_flaring",
        "match_m",
        "det_freq",
        "regions",
    } <= meta.keys()
    assert meta["match_m"] == 2000 and meta["median_ratio"] == pytest.approx(1)
    assert meta["det_freq"] == dict(ours=0.25, myc=0.5, n=1)


def test_model_error_keeps_bright_bands_within_ten_percent():
    obs = spectrum(esf=1e-3)
    obs[4] *= 1.06  # 6% calibration-scale mismatch in M12 on a bright pixel
    obs[1] *= 0.95
    result = nf.fit_planck(BANDS, obs, [0.001] * 6)
    assert result[4] == "ok"
    assert result[3] == BANDS
    assert result[0] == pytest.approx(1800, rel=0.03)
