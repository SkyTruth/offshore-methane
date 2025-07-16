# %%
#!/usr/bin/env python3
"""
Fit SGI = f(glint_alpha) and σ(glint_alpha) without statsmodels.

Usage:
    python sgi_regression_no_statsmodels.py data.csv
"""

import argparse
import pathlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# --------------------------------------------------------------------------- #
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV with columns: sgi, glint_alpha [, sid]")
    p.add_argument("--alpha-col", default="glint_alpha")
    p.add_argument("--sgi-col", default="sgi")
    p.add_argument("--out", default="sgi_fit.png")
    return p.parse_args(argv)


def bin_stats(x, y, bins=80, min_pts=40):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    centers, sigmas = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= min_pts:
            centers.append((lo + hi) / 2)
            sigmas.append(np.std(y[mask], ddof=1))
    return np.asarray(centers), np.asarray(sigmas)


# --------------------------------------------------------------------------- #
def main(csv_path="../data/sgi_vs_sga_pixels.csv"):
    df = pd.read_csv(csv_path)
    x_orig = df["glint_alpha"].values.astype(float).reshape(-1, 1)
    y_orig = df["sgi"].values.astype(float)

    # ---------- SKIP RANGE 27° - 33° --------------------------------------- #
    skip_lo, skip_hi = 24.0, 37.0
    keep = ~((x_orig[:, 0] >= skip_lo) & (x_orig[:, 0] <= skip_hi))
    x, y = x_orig[keep], y_orig[keep]
    # ---------------------------------------------------------------------- #

    # --- robust polynomial (degree-3) -------------------------------------- #
    base_model = make_pipeline(
        PolynomialFeatures(3, include_bias=False), LinearRegression()
    )
    ransac = RANSACRegressor(
        base_model, min_samples=0.5, residual_threshold=0.08, max_trials=100
    )
    ransac.fit(x, y)
    inlier_mask = ransac.inlier_mask_
    x_in, y_in = x[inlier_mask].ravel(), y[inlier_mask]

    # expose mean function
    def sgi_hat(alpha):
        alpha = np.asarray(alpha).reshape(-1, 1)
        return ransac.predict(alpha)

    # --- local σ(alpha) -------------------------------------------------------- #
    centers, sig = bin_stats(x_in, y_in)
    std_spline = UnivariateSpline(centers, sig, k=3, s=len(sig) * 0.05)

    def sgi_std(alpha):
        return np.clip(std_spline(np.asarray(alpha)), 1e-4, None)

    # --- diagnostic figure ------------------------------------------------- #
    grid = np.linspace(x.min(), x.max(), 400)
    mean = sgi_hat(grid)
    std = sgi_std(grid)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_orig, y_orig, s=5, alpha=0.08, c="k")
    plt.plot(grid, mean, lw=2, label="mean (cubic poly, RANSAC)")
    plt.fill_between(grid, mean - 3 * std, mean + 3 * std, alpha=0.25, label="±3 σ")
    plt.xlabel("glint_alpha")
    plt.ylabel("sgi")
    plt.title("SGI vs glint_alpha (robust cubic fit, ±3σ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../data/images/sgi_vs_sga_pixels.png", dpi=240)
    print("saved → ../data/images/sgi_vs_sga_pixels.png")

    # quick demo
    for a in [12, 20, 28, 34]:
        print(f"alpha={a:>5.1f}  SGI≈{sgi_hat([a])[0]:>7.4f}  σ≈{sgi_std([a])[0]:.4f}")

    # ---- 2.1  cubic coefficients (degree 3, no bias term) --------------------
    linreg = ransac.estimator_.named_steps["linearregression"]
    c0 = float(linreg.intercept_)
    c1, c2, c3 = map(float, linreg.coef_)  # x, x², x³ terms

    # ---- 2.2  tabulate the ±σ curve -----------------------------------------
    grid = np.linspace(x.min(), x.max(), 401)  # SAME grid you plotted
    sigma = std_spline(grid)
    s0, s1, s2, s3 = np.polyfit(grid, sigma, 3)[::-1]  # low-order → high-order

    # ---- 2.3  write the standalone module ------------------------------------
    module_code = f"""
    \"\"\"Pre-baked SGI ≈ f(SGA) model - generated automatically.

    No SciPy / sklearn runtime dependency.  Provides:

    • sgi_hat(alpha) / sgi_std(alpha)     - NumPy-friendly
    • sgi_hat_img(img) / sgi_std_img(img) - ee.Image-friendly (if `earthengine-api` is available)

    Mean model:
        SGÎ(alpha) = c0 + c1·alpha + c2·alpha² + c3·alpha³

    St.dev. model (cubic fit to the spline you trained):
        σ(alpha)   = s0 + s1·alpha + s2·alpha² + s3·alpha³
    \"\"\"

    # ----------- NumPy helpers -------------------------------------------------
    import numpy as np

    # ---- cubic-polynomial coeffs (mean) ----
    _C0 = {c0}
    _C1 = {c1}
    _C2 = {c2}
    _C3 = {c3}

    # ---- cubic-polynomial coeffs (σ) ----
    _S0 = {s0}
    _S1 = {s1}
    _S2 = {s2}
    _S3 = {s3}

    def _poly_mean(a):
        return _C0 + _C1*a + _C2*a**2 + _C3*a**3

    def _poly_std(a):
        return _S0 + _S1*a + _S2*a**2 + _S3*a**3

    def sgi_hat(alpha):
        \"\"\"Return expected SGI for glint-alpha *alpha* (NumPy scalar/array).\"\"\"
        a = np.asarray(alpha, dtype=float)
        return _poly_mean(a)

    def sgi_std(alpha):
        \"\"\"Return 1-σ SGI for glint-alpha *alpha* (NumPy scalar/array).\"\"\"
        a = np.asarray(alpha, dtype=float)
        return np.clip(_poly_std(a), 1e-4, None)


    # ----------- Optional Earth-Engine helpers --------------------------------
    def sgi_hat_img(alpha_img):
        \"\"\"Pixel-wise SGI mean for an ee.Image *alpha_img* band.\"\"\"
        return (alpha_img.pow(3).multiply(_C3)
                .add(alpha_img.pow(2).multiply(_C2))
                .add(alpha_img.multiply(_C1))
                .add(_C0))

    def sgi_std_img(alpha_img):
        \"\"\"Pixel-wise SGI σ for an ee.Image *alpha_img* band.\"\"\"
        std = (alpha_img.pow(3).multiply(_S3)
            .add(alpha_img.pow(2).multiply(_S2))
            .add(alpha_img.multiply(_S1))
            .add(_S0))
        return std.max(1e-4)
    """

    out_path = pathlib.Path("../offshore_methane/models/sgi_funcs.py")
    out_path.write_text(textwrap.dedent(module_code))
    print(f"Wrote hard‑coded functions → {out_path.resolve()}")


if __name__ == "__main__":
    main()

# %%
