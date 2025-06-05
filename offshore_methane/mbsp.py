"""Utilities for MultiBand Single Pass methane retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MBSPConstants:
    """Absorption coefficients for Sentinel-2 MBSP retrieval."""

    k11: float
    k12: float


S2_CONSTANTS = {
    "A": MBSPConstants(k11=0.009258572808558494, k12=0.05481104252792486),
    "B": MBSPConstants(k11=0.007711602805452748, k12=0.04210953353251079),
}


def compute_scaling_coefficient(b11: np.ndarray, b12: np.ndarray) -> float:
    """Return regression slope between band 11 and band 12.

    Parameters
    ----------
    b11, b12 : np.ndarray
        Arrays of the same shape containing reflectances.

    Returns
    -------
    float
        Best fit slope ``c`` such that ``b12 ~ c * b11``.
    """
    mask = np.isfinite(b11) & np.isfinite(b12)
    if not np.any(mask):
        raise ValueError("No valid pixels to compute slope")
    s12 = np.sum(b11[mask] * b12[mask])
    s11 = np.sum(b11[mask] ** 2)
    if s11 == 0:
        raise ValueError("Sum of squares is zero")
    return s12 / s11


def mbsp_fractional_absorption(b11: np.ndarray, b12: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute fractional absorption field for MBSP retrieval.

    Parameters
    ----------
    b11, b12 : np.ndarray
        Arrays of band 11 and band 12 reflectances.

    Returns
    -------
    Tuple[float, np.ndarray]
        The regression coefficient ``c`` and the fractional field ``R``.
    """
    c = compute_scaling_coefficient(b11, b12)
    r = (c * b12 - b11) / b11
    return c, r


def invert_mbsp(r: np.ndarray, const: MBSPConstants, tol: float = 1e-6) -> np.ndarray:
    """Invert MBSP fractional absorption to methane column enhancement.

    Solves ``r = exp(-k12 * delta) - exp(-k11 * delta)`` for ``delta``.

    Parameters
    ----------
    r : np.ndarray
        Fractional absorption field.
    const : MBSPConstants
        Absorption coefficients for the sensor.
    tol : float, optional
        Root solver tolerance.

    Returns
    -------
    np.ndarray
        Array of methane column enhancements ``delta``.
    """

    def solve(value: float) -> float:
        lo = 0.0
        hi = 5.0
        # Bracket root
        for _ in range(100):
            f_hi = np.exp(-const.k12 * hi) - np.exp(-const.k11 * hi) - value
            f_lo = np.exp(-const.k12 * lo) - np.exp(-const.k11 * lo) - value
            if f_lo * f_hi < 0:
                break
            hi *= 2
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = np.exp(-const.k12 * mid) - np.exp(-const.k11 * mid) - value
            if abs(f_mid) < tol:
                return mid
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return mid

    vectorized = np.vectorize(solve, otypes=[float])
    return vectorized(r)
