"""VIIRS Nightfire physics, simplified from Elvidge et al. 2013 and 2019.

Pure numpy/scipy; no I/O. Radiances are W m-2 sr-1 um-1 throughout.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.optimize import minimize

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W m-2 K-4
H, C, KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
EARTH_RADIUS_M = 6_371_000.0

# Band centres (um) and full-pixel saturation radiances (Elvidge 2019 Table 1).
BAND_UM = {"M7": 0.865, "M8": 1.24, "M10": 1.61, "M11": 2.25, "M12": 3.70, "M13": 4.05}
SATURATION = {
    "M7": 349.0,
    "M8": 197.88,
    "M10": 85.44,
    "M11": 38.16,
    "M12": 4.41,
    "M13": 404.3,
}

UINT16_FILL_MIN = 65528  # SDR uint16 values >= this are fill codes
FLOAT_FILL_MAX = -900.0  # SDR float fills are -999.x

# M-band aggregation zones by column: 1:1, 2:1, 3:1 (nadir), 2:1, 1:1.
ZONES = ((0, 640), (640, 1008), (1008, 2192), (2192, 2560), (2560, 3200))

M_SCAN_LINES = 16  # detectors per scan
FLARE_T_DEFAULT = 1810.0  # K; M10 peak-emission temperature (Elvidge 2013 s2.5)


def planck(wl_um: np.ndarray | float, temp_k: np.ndarray | float) -> np.ndarray:
    """Spectral radiance (W m-2 sr-1 um-1) at band centre ``wl_um``."""
    wl = np.asarray(wl_um, dtype=float) * 1e-6
    t = np.asarray(temp_k, dtype=float)
    return 2 * H * C**2 / wl**5 / np.expm1(H * C / (wl * KB * t)) * 1e-6


def sdr_radiance(raw: np.ndarray, factors: np.ndarray | None) -> np.ndarray:
    """Scaled radiance with fills as NaN (uint16 + scale/offset, or float)."""
    if raw.dtype.kind == "f":
        out = raw.astype(np.float64)
        out[out < FLOAT_FILL_MAX] = np.nan
        return out
    out = raw.astype(np.float64) * factors[0] + factors[1]
    out[raw >= UINT16_FILL_MIN] = np.nan
    return out


def zone_statistics(
    values: np.ndarray, usable: np.ndarray, n_sigma: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column threshold: zone mean + n_sigma * SD over usable pixels.

    Bright outliers are excluded from the background by one 4-sigma clip, which
    stands in for the 2013 paper's fixed DN > 100 screen.
    """
    thr = np.full(values.shape[1], np.inf)
    noise = np.full(values.shape[1], np.nan)
    for c0, c1 in ZONES:
        v = values[:, c0:c1][usable[:, c0:c1]]
        if v.size < 100:
            continue
        mu, sd = v.mean(), v.std()
        v = v[v <= mu + 4 * sd]
        thr[c0:c1] = v.mean() + n_sigma * v.std()
        noise[c0:c1] = v.std()
    return thr, noise


def zone_thresholds(
    values: np.ndarray, usable: np.ndarray, n_sigma: float = 4.0
) -> np.ndarray:
    """Thresholds from the same clipped statistics used for fit noise."""
    return zone_statistics(values, usable, n_sigma)[0]


def local_background(
    values: np.ndarray, exclude: np.ndarray, row: int, col: int
) -> tuple[float, float]:
    """Mean and SD of a 10x10 window (100x100 if < 50 pixels), minus ``exclude``."""
    for half in (5, 50):
        r0, r1 = max(row - half, 0), min(row + half, values.shape[0])
        c0, c1 = max(col - half, 0), min(col + half, values.shape[1])
        win = values[r0:r1, c0:c1]
        bg = win[~exclude[r0:r1, c0:c1] & np.isfinite(win)]
        if bg.size >= 50:
            return float(bg.mean()), float(bg.std())
    return np.nan, np.nan


# Local quality choices: chi-square per max(n-2, 1) <= 9 (3-sigma RMS).
# Two-band fits have no residual degrees of freedom; use denominator one.
FIT_CHI2_MAX = 9.0
FIT_T_MIN, FIT_T_MAX = 1200.0, 3500.0
TWO_BAND_T_MIN, TWO_BAND_T_MAX = 1200.0, 2500.0
NOISE_FLOOR = 1e-6  # radiance units; guards uniform synthetic/background scenes
# Relative model error added in quadrature to each band's noise: band-centre
# Planck (no spectral response), no atmospheric correction and calibration
# (~2-5%). Without it bright flares fail chi-square on model error alone and
# lose good bands to elimination (100% of fits above 20 MW on 2026-10-29).
MODEL_REL_ERROR = 0.10


def _fit_planck(
    bands: list[str],
    radiance: list[float],
    noise: list[float] | None = None,
    t0: float = 1000.0,
    diagnostics: dict | None = None,
    require_swir: bool = True,
) -> tuple[float, float, float, list[str], str]:
    """Noise-weighted fit with guarded leave-one-out elimination or M10 pin.

    Return T, ESF, reduced chi-square, used bands, fit flag. Rejected fits
    without usable M10 return NaNs and 'unfitted'. The pin is a constrained
    estimate rather than an accepted spectral fit.
    """
    original = dict(zip(bands, radiance))
    bands, obs = list(bands), np.asarray(radiance, dtype=float)
    sd = np.asarray(noise if noise is not None else np.ones(len(obs)), dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sd) & (sd >= 0)
    bands = [b for b, good in zip(bands, valid) if good]
    obs, sd = obs[valid], np.maximum(sd[valid], NOISE_FLOOR)
    sd = np.hypot(sd, MODEL_REL_ERROR * np.abs(obs))
    usable_m10 = "M10" in bands and original["M10"] > 0
    dropped = []
    chi = np.inf
    reason = "insufficient_bands" if len(bands) < 2 else "no_m10_m11"

    def refit(names, values, errors):
        wl = np.array([BAND_UM[b] for b in names])

        def objective(p):
            t = np.clip(p[0], 300.0, 5000.0)
            model = np.exp(np.clip(p[1], -100, 10)) * planck(wl, t)
            return float(np.sum(((model - values) / errors) ** 2))

        esf0 = max(values[0], 1e-9) / planck(wl[0], t0)
        res = minimize(
            objective,
            [t0, np.log(esf0)],
            method="Nelder-Mead",
            options={"xatol": 0.001, "fatol": 1e-10, "maxiter": 2000},
        )
        return res, float(res.fun) / max(len(names) - 2, 1)

    def guard(names, res):
        t = float(res.x[0])
        esf = float(np.exp(np.clip(res.x[1], -100, 10)))
        if not res.success:
            return "nonconvergence"
        if not np.isfinite(t) or not 0 < esf <= 1 or t > FIT_T_MAX:
            return "bounds"
        if t < FIT_T_MIN:
            return "low_t"
        if not {"M10", "M11"}.intersection(names):
            return "no_m10_m11"
        if require_swir and len({"M10", "M11", "M12", "M13"}.intersection(names)) < 2:
            return "band_set"
        if len(names) == 2 and not TWO_BAND_T_MIN <= t <= TWO_BAND_T_MAX:
            return "two_band_range"
        return ""

    if len(bands) >= 2:
        res, chi = refit(bands, obs, sd)
        while True:
            reason = guard(bands, res)
            if not reason and np.isfinite(chi) and chi <= FIT_CHI2_MAX:
                return (
                    float(res.x[0]),
                    float(np.exp(res.x[1])),
                    chi,
                    bands,
                    "eliminated:" + "+".join(dropped) if dropped else "ok",
                )
            if not reason:
                reason = "chi2"
            elif np.isfinite(chi) and chi <= FIT_CHI2_MAX:
                # A good fit that fails a physical guard is pinned, not
                # searched for a band subset that passes (D43).
                break
            if len(bands) <= 3:
                break
            trials = []
            for i in range(len(bands)):
                remaining = bands[:i] + bands[i + 1 :]
                fit, score = refit(remaining, np.delete(obs, i), np.delete(sd, i))
                trials.append(
                    (
                        bool(guard(remaining, fit)),
                        score if np.isfinite(score) else np.inf,
                        i,
                        fit,
                    )
                )
            _, chi, i, res = min(trials, key=lambda trial: trial[:3])
            dropped.append(bands.pop(i))
            obs, sd = np.delete(obs, i), np.delete(sd, i)
        if diagnostics is not None:
            diagnostics["t_fit_rejected"] = float(res.x[0])

    if usable_m10:
        esf = original["M10"] / float(planck(BAND_UM["M10"], FLARE_T_DEFAULT))
        if 0 < esf <= 1:
            return FLARE_T_DEFAULT, esf, chi, ["M10", "T1810"], "pinned:" + reason
    return np.nan, np.nan, chi, [], "unfitted"


def fit_planck(
    bands: list[str],
    radiance: list[float],
    noise: list[float] | None = None,
    t0: float = 1000.0,
    diagnostics: dict | None = None,
) -> tuple[float, float, float, list[str], str]:
    """Guarded spectral fit; optionally audit the v5 band-set rule's effect."""
    fit = _fit_planck(bands, radiance, noise, t0, diagnostics)
    if diagnostics is not None:
        relaxed = _fit_planck(bands, radiance, noise, t0, require_swir=False)
        diagnostics["band_rule_changed"] = fit[3:] != relaxed[3:]
    return fit


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def pixel_area_m2(lat: np.ndarray, lon: np.ndarray, row: int, col: int) -> float:
    """Pixel footprint from neighbour spacing (along-scan x along-track).

    Along-track neighbours are taken within the same 16-line scan, so the
    bow-tie overlap between scans does not shrink the estimate.
    """
    nrow, ncol = lat.shape
    c0, c1 = max(col - 1, 0), min(col + 1, ncol - 1)
    scan0 = row - row % M_SCAN_LINES
    r0 = max(row - 1, scan0)
    r1 = min(row + 1, scan0 + M_SCAN_LINES - 1, nrow - 1)
    ds = haversine_m(lat[row, c0], lon[row, c0], lat[row, c1], lon[row, c1])
    dt = haversine_m(lat[r0, col], lon[r0, col], lat[r1, col], lon[r1, col])
    return float(ds / max(c1 - c0, 1) * dt / max(r1 - r0, 1))


def radiant_heat_mw(temp_k: float, esf: float, area_m2: float) -> float:
    """Radiant heat of the sub-pixel source: ESF * sigma * T^4 * pixel area."""
    return esf * SIGMA * temp_k**4 * area_m2 / 1e6


def solar_zenith_deg(lat, lon, when: datetime) -> np.ndarray:
    """Solar zenith angle (degrees), NOAA general solar position equations."""
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    tst = hour * 60 + eqtime + 4 * np.asarray(lon, dtype=float)
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(np.asarray(lat, dtype=float))
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(cosz, -1, 1)))
