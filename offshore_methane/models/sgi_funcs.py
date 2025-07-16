"""Pre-baked SGI ≈ f(SGA) model - generated automatically.

No SciPy / sklearn runtime dependency.  Provides:

• sgi_hat(alpha) / sgi_std(alpha)     - NumPy-friendly
• sgi_hat_img(img) / sgi_std_img(img) - ee.Image-friendly (if `earthengine-api` is available)

Mean model:
    SGÎ(alpha) = c0 + c1·alpha + c2·alpha² + c3·alpha³

St.dev. model (cubic fit to the spline you trained):
    σ(alpha)   = s0 + s1·alpha + s2·alpha² + s3·alpha³
"""

# ----------- NumPy helpers -------------------------------------------------
import numpy as np

# ---- cubic-polynomial coeffs (mean) ----
_C0 = -0.4311114865932371
_C1 = 0.03386401318818421
_C2 = -0.002193887448915637
_C3 = 2.813232215127007e-05

# ---- cubic-polynomial coeffs (σ) ----
_S0 = 0.08081091517891231
_S1 = -0.005428377169097809
_S2 = 0.00024364624931538838
_S3 = -3.3266593929279482e-06


def _poly_mean(a):
    return _C0 + _C1 * a + _C2 * a**2 + _C3 * a**3


def _poly_std(a):
    return _S0 + _S1 * a + _S2 * a**2 + _S3 * a**3


def sgi_hat(alpha):
    """Return expected SGI for glint-alpha *alpha* (NumPy scalar/array)."""
    a = np.asarray(alpha, dtype=float)
    return _poly_mean(a)


def sgi_std(alpha):
    """Return 1-σ SGI for glint-alpha *alpha* (NumPy scalar/array)."""
    a = np.asarray(alpha, dtype=float)
    return np.clip(_poly_std(a), 1e-4, None)


# ----------- Optional Earth-Engine helpers --------------------------------
def sgi_hat_img(alpha_img):
    """Pixel-wise SGI mean for an ee.Image *alpha_img* band."""
    return (
        alpha_img.pow(3)
        .multiply(_C3)
        .add(alpha_img.pow(2).multiply(_C2))
        .add(alpha_img.multiply(_C1))
        .add(_C0)
    )


def sgi_std_img(alpha_img):
    """Pixel-wise SGI σ for an ee.Image *alpha_img* band."""
    std = (
        alpha_img.pow(3)
        .multiply(_S3)
        .add(alpha_img.pow(2).multiply(_S2))
        .add(alpha_img.multiply(_S1))
        .add(_S0)
    )
    return std.max(1e-4)
