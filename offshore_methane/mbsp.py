# mbsp.py
"""
Both the complex (SGI + coarse SGA) and simple (Varon 21 fractional-slope)
MBSP implementations, unchanged from your monolith.
"""

import math
from typing import Tuple

import ee


# ------------------------------------------------------------------
def mbsp_complex_ee(
    image: ee.Image,
    sga_coarse: ee.Image,
    centre: ee.Geometry,
    local_sga_range: tuple[float, float],
) -> ee.Image:
    """
    Complex MBSP (Varon 2021, §3.2) with scene-specific regression +
    coarse-grid SGA upscale to 20 m.
    """
    b12 = image.select("B12").divide(10000)
    b11 = image.select("B11").divide(10000)

    b02 = (
        image.select("B2")
        .divide(10000)
        .resample("bilinear")
        .reproject(crs=b11.projection())
    )
    b03 = (
        image.select("B3")
        .divide(10000)
        .resample("bilinear")
        .reproject(crs=b11.projection())
    )
    b04 = (
        image.select("B4")
        .divide(10000)
        .resample("bilinear")
        .reproject(crs=b11.projection())
    )
    b_vis = b02.add(b03).add(b04).divide(3).rename("Bvis_20m")

    # -- SGI ---------------------------------------------------------
    denom = b12.add(b_vis).max(1e-4)
    sgi = b12.subtract(b_vis).divide(denom).clamp(-1, 1)

    mask_geom = centre.buffer(50_000)  # @Ethan is this right? Improved mask?

    # @Ethan Think through the following code and make sure it's correct.
    reg1_img = ee.Image.cat(ee.Image.constant(1), sgi, b11.rename("y"))
    lr1 = reg1_img.reduceRegion(
        ee.Reducer.linearRegression(numX=2, numY=1),
        geometry=mask_geom,
        scale=20,
        bestEffort=True,
    )
    coef1 = ee.Array(lr1.get("coefficients"))
    a0, a1 = coef1.get([0, 0]), coef1.get([1, 0])
    fit = ee.Image.constant(a0).add(ee.Image.constant(a1).multiply(sgi))
    b11_flat = b11.subtract(fit)

    ratio = b12.divide(b11.add(1e-6))
    b12_flat = b12.subtract(fit.multiply(ratio))

    x1 = b12_flat
    x2 = b12_flat.multiply(sgi)
    reg2_img = ee.Image.cat(x1, x2, b11_flat.rename("y"))
    lr2 = reg2_img.reduceRegion(
        ee.Reducer.linearRegression(numX=2, numY=1),
        geometry=mask_geom,
        scale=20,
        bestEffort=True,
    )
    coef2 = ee.Array(lr2.get("coefficients"))
    c0, c1 = coef2.get([0, 0]), coef2.get([1, 0])
    c = ee.Image.constant(c0).add(ee.Image.constant(c1).multiply(sgi))

    R = c.multiply(b12_flat).subtract(b11_flat).divide(b11_flat.add(1e-6))

    sga_hi = sga_coarse.resample("bilinear").reproject(crs=b11.projection())
    R = R.updateMask(
        sga_hi.gt(local_sga_range[0]).And(sga_hi.lt(local_sga_range[1]))
    ).rename("MBSP")
    return ee.Image(R).copyProperties(image, ["system:index", "system:time_start"])


# ------------------------------------------------------------------
def mbsp_simple_ee(image: ee.Image, mask_c: ee.Image, mask_mbsp: ee.Image) -> ee.Image:
    """
    Fractional-slope MBSP (single zero-intercept regression).
    """

    # First calculate the C factor
    C_masked_img = image.updateMask(mask_c)
    num = (
        C_masked_img.select("B11")
        .multiply(C_masked_img.select("B12"))
        .reduceRegion(ee.Reducer.sum(), C_masked_img.geometry(), 20, bestEffort=True)
    )

    den = (
        C_masked_img.select("B12")
        .pow(2)
        .reduceRegion(ee.Reducer.sum(), C_masked_img.geometry(), 20, bestEffort=True)
    )
    C_factor = ee.Number(num.get("B11")).divide(ee.Number(den.get("B12")))

    # Then calculate the MBSP
    MBSP_masked_img = image.updateMask(mask_mbsp)

    MBSP_masked_result = (
        MBSP_masked_img.select("B12")
        .multiply(C_factor)
        .subtract(MBSP_masked_img.select("B11"))
        .divide(MBSP_masked_img.select("B11"))
        .rename("MBSP")
        .set({"C_factor": C_factor})
    )

    return MBSP_masked_result.copyProperties(
        image, ["system:index", "system:time_start"]
    )


# ------------------------------------------------------------------
def mbsp_sgx(
    image: ee.Image,
    mask_mbsp: ee.Image,
    sga_limits: Tuple[float, float],
) -> ee.Image:
    """
    MBSP variant that estimates a **local, glint-aware scaling factor**
    α(x,y)  =  a  +  b · SGI(x,y)

    Steps
    -----
    1.  Compute per-pixel ratio  r = B12 / B11.
    2.  Fit     r = a + b·SGI   (linear)   across all pixels that satisfy:
          • mask_mbsp == 1
          • sga_limits[0] ≤ SGA ≤ sga_limits[1]
    3.  α(x,y) = a + b·SGI(x,y)
    4.  MBSP   = (B12 - α·B11) / (α·B11)

    Notes
    -----
    • Requires two extra bands already present in *image*:
        - SGA : pixel-wise sun-glint angle (deg)
        - SGI : sun-glint index   (dimensionless)
    • Uses **single** global a,b (scene-level) but lets α vary via SGI(x,y).
      Keeping a,b global stabilises the regression and avoids EE neighbourhood
      reducers, which struggle with variable-size windows.
    """
    # ── 1.  Apply quality mask & SGA window ───────────────────────────
    sga_ok = (
        image.select("SGA")
        .gte(sga_limits[0])
        .And(image.select("SGA").lte(sga_limits[1]))
    )
    valid = mask_mbsp.And(sga_ok)

    img = image.updateMask(valid)

    # ── 2.  r = B12 / B11 ;  fit r ~ (1, SGI)  ────────────────────────
    ratio = img.select("B12").divide(img.select("B11").add(1e-6)).rename("ratio")
    reg_img = ee.Image.cat(ee.Image.constant(1), img.select("SGI"), ratio)

    lr = reg_img.reduceRegion(
        ee.Reducer.linearRegression(numX=2, numY=1),
        geometry=img.geometry(),
        scale=20,
        bestEffort=True,
    )

    coef = ee.Array(lr.get("coefficients"))
    a = coef.get([0, 0])
    b = coef.get([1, 0])

    alpha = ee.Image.constant(a).add(ee.Image.constant(b).multiply(img.select("SGI")))

    # ── 3.  MBSP calculation ──────────────────────────────────────────
    mbsp = (
        img.select("B12")
        .subtract(alpha.multiply(img.select("B11")))
        .divide(alpha.multiply(img.select("B11")).add(1e-6))
        .rename("MBSP")
        .set({"a_coef": a, "b_coef": b})
    )

    return mbsp.copyProperties(image, ["system:index", "system:time_start"])


def _fresnel_ratio(theta_deg: ee.Image, n: float = 1.333) -> ee.Image:
    """
    Per-pixel Fresnel reflectance of an air-water interface.
    Returns a two-band image (F11, F12).  Uses the same refractive
    index for both MSI SWIR bands because the wavelength dependence
    of n is < 0.1 % between 1.6 µm and 2.2 µm.
    """
    theta = theta_deg.multiply(math.pi / 180)  # → radians
    cos_t = theta.cos()  # Image
    sin_t2 = theta.sin().pow(2)  # Image

    def _band(refr_idx: float) -> ee.Image:
        n2 = refr_idx**2  # scalar
        n2_img = ee.Image.constant(n2)  # → Image

        root = n2_img.subtract(sin_t2).sqrt()  # √(n² - sin²θ)

        # s-polarisation
        rs = cos_t.subtract(root).divide(cos_t.add(root)).pow(2)

        # p-polarisation
        rp = (
            n2_img.multiply(cos_t)
            .subtract(root)
            .divide(n2_img.multiply(cos_t).add(root))
            .pow(2)
        )

        return rs.add(rp).divide(2)

    f11 = _band(n).rename("F11")
    f12 = _band(n).rename("F12")  # same n; negligible dispersion

    return ee.Image.cat(f11, f12)
