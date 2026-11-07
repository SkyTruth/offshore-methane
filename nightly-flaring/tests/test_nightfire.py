from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import nightfire as nf


def test_planck_peak_follows_wien():
    wl = np.linspace(0.5, 5.0, 4501)
    peak = wl[np.argmax(nf.planck(wl, 1810.0))]
    assert peak == pytest.approx(2897.77 / 1810.0, abs=0.01)


@pytest.mark.parametrize("temp", [1300.0, 1800.0, 2400.0])
def test_fit_planck_recovers_temperature_and_esf(temp):
    bands = ["M10", "M11", "M12", "M13"]
    esf = 2e-5
    obs = [esf * float(nf.planck(nf.BAND_UM[b], temp)) for b in bands]
    t, e, ssr, used = nf.fit_planck(bands, obs)
    assert t == pytest.approx(temp, rel=0.01)
    assert e == pytest.approx(esf, rel=0.05)
    assert used == bands
    assert ssr < 1e-6


def test_fit_planck_drops_band_with_negative_outlier():
    bands = ["M10", "M11", "M12", "M13"]
    esf, temp = 1e-3, 1700.0
    obs = [esf * float(nf.planck(nf.BAND_UM[b], temp)) for b in bands]
    obs[1] -= 10.0  # M11 far below the curve
    t, _, _, used = nf.fit_planck(bands, obs)
    assert "M11" not in used
    assert t == pytest.approx(temp, rel=0.02)


def test_zone_thresholds_ignore_hot_outliers():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=(768, 3200))
    values[100, 100] = 500.0
    thr = nf.zone_thresholds(values, np.ones_like(values, dtype=bool))
    assert thr.shape == (3200,)
    assert np.all((thr > 3.8) & (thr < 4.2))


def test_sdr_radiance_fills_become_nan():
    raw = np.array([[10, 65533]], dtype=np.uint16)
    out = nf.sdr_radiance(raw, np.array([0.5, 1.0]))
    assert out[0, 0] == 6.0 and np.isnan(out[0, 1])
    flt = nf.sdr_radiance(np.array([[0.2, -999.3]], dtype=np.float32), None)
    assert flt[0, 0] == pytest.approx(0.2) and np.isnan(flt[0, 1])


def test_solar_zenith_noon_and_midnight_at_equinox():
    noon = nf.solar_zenith_deg(0.0, 0.0, datetime(2026, 3, 20, 12, 7))
    midnight = nf.solar_zenith_deg(0.0, 0.0, datetime(2026, 3, 20, 0, 7))
    assert noon < 2.0
    assert midnight > 178.0


def test_pixel_area_from_regular_grid():
    step = 750.0 / 111_195.0  # degrees per 750 m at the equator
    lat, lon = np.meshgrid(np.arange(16) * step, np.arange(10) * step, indexing="ij")
    area = nf.pixel_area_m2(lat, lon, 5, 5)
    assert area == pytest.approx(750.0**2, rel=0.01)


def test_radiant_heat_units():
    assert nf.radiant_heat_mw(1000.0, 1.0, 1.0) == pytest.approx(0.0567, rel=1e-3)


def test_site_nights_mean_counts_clear_non_detections_as_zero():
    pytest.importorskip("h5py")
    import pipeline

    sites = pd.DataFrame({"site_id": [1, 2], "lat": [0.0, 0.0], "lon": [0.0, 0.0]})
    over = pd.DataFrame(
        {
            "site_id": [1, 1, 1, 2],
            "time_utc": [
                "2026-11-02T01:30:00",
                "2026-11-02T02:20:00",
                "2026-11-02T03:10:00",
                "2026-11-02T01:30:00",
            ],
            "sza": [150.0, 150.0, 150.0, 150.0],
            "cloud_mask": [0, 1, 3, 3],
            "detected": [True, False, False, False],
            "rh_mw": [6.0, 0.0, 0.0, 0.0],
            "temp_k": [1800.0, np.nan, np.nan, np.nan],
        }
    )
    out = pipeline.site_nights(over, sites, date(2026, 11, 1), date(2026, 11, 1))
    s1 = out.set_index("site_id").loc[1]
    assert s1["status"] == "flaring"
    assert s1["n_clear"] == 2 and s1["n_overpasses"] == 3
    assert s1["rh_mw"] == pytest.approx(3.0)
    assert s1["bcm_per_year"] == pytest.approx(3.0 * pipeline.BCM_PER_MW_YEAR)
    assert out.set_index("site_id").loc[2, "status"] == "cloudy"
