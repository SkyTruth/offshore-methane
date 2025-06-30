# mbsp.py
"""
Both the complex (SGI + coarse SGA) and simple (Varon 21 fractional-slope)
MBSP implementations, unchanged from your monolith.
"""

import ee

from offshore_methane.masking import (
    DEFAULT_MASK_PARAMS,
    build_mask_for_C,
    build_mask_for_MBSP,
)


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
def mbsp_simple_ee(image: ee.Image, centre: ee.Geometry) -> ee.Image:
    """
    Fractional-slope MBSP (single zero-intercept regression).
    """

    # First calculate the C factor
    C_masked_img = image.updateMask(
        build_mask_for_C(image, centre, DEFAULT_MASK_PARAMS)
    )
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
    MBSP_masked_img = image.updateMask(
        build_mask_for_MBSP(image, centre, DEFAULT_MASK_PARAMS)
    )
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
