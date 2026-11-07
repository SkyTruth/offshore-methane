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
BAND_UM = {"M10": 1.61, "M11": 2.25, "M12": 3.70, "M13": 4.05}
SATURATION = {"M10": 85.44, "M11": 38.16, "M12": 4.41, "M13": 404.3}

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


def zone_thresholds(
    values: np.ndarray, usable: np.ndarray, n_sigma: float = 4.0
) -> np.ndarray:
    """Per-column threshold: zone mean + n_sigma * SD over usable pixels.

    Bright outliers are excluded from the background by one 4-sigma clip, which
    stands in for the 2013 paper's fixed DN > 100 screen.
    """
    thr = np.full(values.shape[1], np.inf)
    for c0, c1 in ZONES:
        v = values[:, c0:c1][usable[:, c0:c1]]
        if v.size < 100:
            continue
        mu, sd = v.mean(), v.std()
        v = v[v <= mu + 4 * sd]
        thr[c0:c1] = v.mean() + n_sigma * v.std()
    return thr


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


def fit_planck(
    bands: list[str], radiance: list[float], t0: float = 1000.0
) -> tuple[float, float, float, list[str]]:
    """Single-curve Planck fit for temperature and emission scaling factor.

    Nelder-Mead on (T, log ESF), minimising the sum of squared radiance
    residuals. While SSR > 2 and more than two bands remain, the band with the
    most negative residual is dropped and the fit repeated (Elvidge 2019).
    Returns (T, ESF, SSR, bands used).
    """
    bands, obs = list(bands), np.asarray(radiance, dtype=float)
    while True:
        wl = np.array([BAND_UM[b] for b in bands])

        def ssr(p, wl=wl, obs=obs):
            t = np.clip(p[0], 300.0, 5000.0)
            return float(np.sum((np.exp(p[1]) * planck(wl, t) - obs) ** 2))

        i10 = bands.index("M10") if "M10" in bands else 0
        esf0 = max(obs[i10], 1e-9) / planck(wl[i10], t0)
        res = minimize(
            ssr,
            x0=[t0, np.log(esf0)],
            method="Nelder-Mead",
            options={"xatol": 0.1, "fatol": 1e-12, "maxiter": 2000},
        )
        t, esf = float(np.clip(res.x[0], 300.0, 5000.0)), float(np.exp(res.x[1]))
        if res.fun <= 2.0 or len(bands) <= 2:
            return t, esf, float(res.fun), bands
        resid = obs - esf * planck(wl, t)
        drop = int(np.argmin(resid))
        bands.pop(drop)
        obs = np.delete(obs, drop)


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
