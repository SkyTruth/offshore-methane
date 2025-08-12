# %%
#!/usr/bin/env python3
"""
Fit SGI = f(SGA) and sigma(SGA) with an arbitrary-degree polynomial (2-4+).

Usage
-----
python sgi_regression_poly.py ../data/18M_dotplot.csv           # default deg=3
python sgi_regression_poly.py ../data/18M_dotplot.csv --deg 4
"""

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
def bin_stats(x, y, bins=80, min_pts=40):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    ctrs, sigs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_pts:
            ctrs.append((lo + hi) / 2)
            sigs.append(np.std(y[m], ddof=1))
    return np.asarray(ctrs), np.asarray(sigs)


def rolling_sigma(x, y, n=5000, step=1000, core_pct=0.25):
    """sigma(α) on equal-count windows, returns centers & sigmas."""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    centers, sigs = [], []
    lo_q, hi_q = (0.5 - core_pct) * 100, (0.5 + core_pct) * 100

    for start in range(0, len(x_s) - n + 1, step):
        xs = x_s[start : start + n]
        ys = y_s[start : start + n]
        q_lo, q_hi = np.percentile(ys, [lo_q, hi_q])
        ys_core = ys[(ys >= q_lo) & (ys <= q_hi)]
        centers.append(xs.mean())
        sigs.append(np.std(ys_core, ddof=1))
    return np.asarray(centers), np.asarray(sigs)


def densest_core_mask(x, y, *, bins=80, core_pct=0.20, min_pts=40):
    """
    Return a Boolean mask selecting only the densest (±core_pct) region in
    each SGA bin.  core_pct=0.20  -> keep the central 1-2·core_pct = 40 % of
    points around the median (i.e. between the 40th and 60th percentiles).

    Parameters
    ----------
    x, y : 1-D ndarrays (same length)
    bins : int        - how many bins along x
    core_pct : float  - e.g. 0.20 keeps 40 % of points
    min_pts : int     - skip bins with fewer points

    Returns
    -------
    mask : Boolean ndarray, same length as x
    """
    mask = np.zeros_like(y, dtype=bool)
    edges = np.linspace(x.min(), x.max(), bins + 1)
    lo_q = (0.5 - core_pct) * 100  # e.g. 30 for 0.20
    hi_q = (0.5 + core_pct) * 100  # e.g. 70
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < min_pts:
            continue
        y_bin = y[sel]
        q_lo, q_hi = np.percentile(y_bin, [lo_q, hi_q])
        mask |= sel & (y >= q_lo) & (y <= q_hi)
    return mask


# --------------------------------------------------------------------------- #
def main(csv_path="../data/18M_dotplot.csv", degree=4, core_mean=0.25, core_std=0.5):
    df = pd.read_csv(csv_path).sample(frac=0.02, random_state=42)
    x0 = df["SGA"].to_numpy(float).reshape(-1, 1)
    y0 = df["SGI"].to_numpy(float)

    skip_lo, skip_hi = 45.0, 100.0
    keep_range = ~((x0[:, 0] >= skip_lo) & (x0[:, 0] <= skip_hi))

    # densest 40 % around the median in every SGA bin
    mean_core = densest_core_mask(x0[:, 0], y0, core_pct=core_mean, bins=80)

    x_mean = x0[keep_range & mean_core]
    y_mean = y0[keep_range & mean_core]

    base = make_pipeline(
        PolynomialFeatures(degree, include_bias=False), LinearRegression()
    )
    ransac = RANSACRegressor(
        base,
        min_samples=0.05,
        residual_threshold=0.1,  # may tighten now
        max_trials=100,
    )
    ransac.fit(x_mean, y_mean)

    sgi_hat = lambda a: ransac.predict(np.asarray(a).reshape(-1, 1))  # noqa: E731

    # ── 4️⃣ looser core for STD  (wider) ────────────────────────────────────────
    std_core = densest_core_mask(x0[:, 0], y0, core_pct=core_std, bins=80)

    # We keep sigma only on points in the looser core *and* outside the skip range:
    x_std = x0[keep_range & std_core].ravel()
    y_std = y0[keep_range & std_core]

    # centers, sig = bin_stats(x_std, y_std)  # local sigma over wider set
    centers, sig = rolling_sigma(x_std, y_std, n=5000, step=100, core_pct=0.2)

    std_spline = UnivariateSpline(
        centers, sig, k=3, s=len(sig) * 0.05, ext=3
    )  # ← NEW: clamp to boundary value
    sgi_std = lambda a: np.clip(std_spline(np.asarray(a)), 1e-4, None)  # noqa: E731
    grid = np.linspace(x_mean.min(), x_mean.max(), 400)
    mean, std = sgi_hat(grid), sgi_std(grid)

    plt.figure(figsize=(8, 6))
    plt.scatter(x0, y0, s=5, alpha=0.08, c="k")
    plt.scatter(x0[mean_core], y0[mean_core], s=5, alpha=0.08, c="r")
    plt.plot(grid, mean, lw=2, label="mean (deg=%d)" % degree)
    plt.fill_between(grid, mean - 0 * std, mean + 4 * std, alpha=0.25, label="±3 sigma")
    # add many values to the vertical axis
    plt.xlabel("SGA")
    plt.ylabel("SGI")
    plt.yticks(np.arange(-0.7, 0.1, 0.03))
    plt.xticks(np.arange(10, 50, 5))
    plt.ylim(-0.7, 0.1)
    plt.xlim(10, 50)
    plt.title(f"SGI vs SGA - polynomial deg={degree}")
    plt.legend()
    plt.tight_layout()
    out_png = f"../data/images/dotplot_18M_deg{degree}.png"
    plt.savefig(out_png, dpi=240)
    print("saved →", out_png)

    for a in [12, 20, 28, 34]:
        print(
            f"alpha={a:>5.1f}  SGI≈{sgi_hat([a])[0]:>7.4f}  sigma≈{sgi_std([a])[0]:.4f}"
        )

    # ---------------- export hard-coded module -----------------------------
    linreg = ransac.estimator_.named_steps["linearregression"]
    coeffs_mean = [float(linreg.intercept_)] + list(map(float, linreg.coef_))
    grid_f = np.linspace(x_std.min(), x_std.max(), 401)
    sigma_f = std_spline(grid_f)
    coeffs_std = list(np.polyfit(grid_f, sigma_f, degree))[::-1]  # low→high

    # helper to emit _C0, _C1 … lines
    def coeff_lines(name, arr):
        return "\n".join(f"_{name}{i} = {v}" for i, v in enumerate(arr))

    # build polynomial evaluation string for EE (explicit powers)
    def ee_poly(img_name, arr, prefix):
        """Build a chain like  a.pow(n).multiply(_Pn).add(…).add(_P0)  with no
        stray parens.  Works for any polynomial degree ≥ 0."""
        if len(arr) == 1:  # degree‑0 edge case
            return f"_{prefix}0"
        terms = [
            f"{img_name}.pow({i}).multiply(_{prefix}{i})"
            for i in range(len(arr) - 1, 0, -1)
        ]
        expr = terms[0] + "".join(f".add({t})" for t in terms[1:])
        return expr + f".add(_{prefix}0)"

    module_code = f"""
\"\"\"Pre-baked SGI≈f(SGA) model (degree={degree}) - generated automatically.\"\"\"

import numpy as np
{coeff_lines("C", coeffs_mean)}

{coeff_lines("S", coeffs_std)}

def _poly(arr, coeffs):
    return sum(c * arr ** i for i, c in enumerate(coeffs))

def sgi_hat(alpha):
    a = np.asarray(alpha, float)
    return _poly(a, [{", ".join(map(str, coeffs_mean))}])

def sgi_std(alpha):
    a = np.asarray(alpha, float)
    return np.clip(_poly(a, [{", ".join(map(str, coeffs_std))}]), 1e-4, None)

# -------- Earth-Engine helpers -----------
def sgi_hat_img(alpha_img):
    return {ee_poly("alpha_img", coeffs_mean, "C")}

def sgi_std_img(alpha_img):
    std = {ee_poly("alpha_img", coeffs_std, "S")}
    return std.max(1e-4)
"""
    out_path = pathlib.Path("../offshore_methane/models/sgi_funcs.py")
    out_path.write_text(textwrap.dedent(module_code))
    print("Wrote hard-coded functions →", out_path.resolve())


if __name__ == "__main__":
    main()

# %%
