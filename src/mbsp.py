"""Minimal MBSP retrieval algorithm for Sentinel-2.

This module implements a simplified version of the MultiBand Single Pass
(MBSP) methane retrieval as described in Varon et al. (2021). It is not a
full reproduction of the paper but captures key steps often omitted in
quick-look notebooks.
"""

from __future__ import annotations

import numpy as np
import ee
from scipy import ndimage

try:
    ee.Initialize()
except Exception:
    pass


def mask_s2(img: ee.Image) -> ee.Image:
    """Mask clouds, cirrus, water and snow using QA60 and SCL bands."""
    qa = img.select("QA60")
    scl = img.select("SCL")
    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
        .And(scl.remap([3, 8, 9, 10, 11], [0, 0, 0, 0, 0]).eq(0))
    )
    return img.updateMask(mask)


def load_l1c_collection(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cloud: int = 20,
    region_km: int = 2,
) -> ee.ImageCollection:
    """Load masked Sentinel-2 L1C images clipped to a ROI."""
    point = ee.Geometry.Point(lon, lat)
    roi = point.buffer(region_km * 500).bounds()
    col = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud))
        .map(mask_s2)
        .map(lambda img: img.clip(roi))
    )
    return col.sort("system:time_start")


def _fit_c(r11: np.ndarray, r12: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        r11 = r11[mask]
        r12 = r12[mask]
    return float(np.nansum(r11 * r12) / np.nansum(r11 ** 2))


def mbsp_arrays(b11: np.ndarray, b12: np.ndarray) -> tuple[np.ndarray, float]:
    """Return MBSP fractional absorption and regression constant."""
    r11 = b11.astype(np.float32) * 1e-4
    r12 = b12.astype(np.float32) * 1e-4

    c = _fit_c(r11, r12)
    mbsp = (r12 - c * r11) / r11
    background = np.abs(mbsp - np.nanmedian(mbsp)) < 3 * np.nanstd(mbsp)
    if not np.any(background):
        background = np.ones_like(mbsp, dtype=bool)
    c = _fit_c(r11, r12, background)
    mbsp = (r12 - c * r11) / r11
    return mbsp, c


def mbsp_to_column(mbsp: np.ndarray) -> np.ndarray:
    """Convert MBSP fractional absorption to methane column using a LUT."""
    delta_x = np.linspace(0, 2000, 41)  # ppb km
    lut = -np.expm1(-0.003 * delta_x)  # simple approximate curve
    return np.interp(mbsp, lut, delta_x)


def detrend_and_smooth(column: np.ndarray) -> np.ndarray:
    trend = ndimage.gaussian_filter(column, sigma=15, mode="nearest")
    detrended = column - trend
    return ndimage.median_filter(detrended, size=3)


def plume_mask(column: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    thresh = np.nanpercentile(column, percentile)
    return column > thresh


def ime_and_rate(column: np.ndarray, mask: np.ndarray, wind_speed: float) -> float:
    ime = np.nansum(column[mask]) * (20 * 20) * 16 / 1000  # kg
    length = np.sqrt(np.count_nonzero(mask)) * 20
    u_eff = 0.33 * wind_speed + 0.45
    q = ime * u_eff / length  # kg / s
    return q * 3.6  # t h^-1
