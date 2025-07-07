# mbsp.py
"""
Both the complex (SGI + coarse SGA) and simple (Varon 21 fractional-slope)
MBSP implementations, unchanged from your monolith.
"""

import ee


# ------------------------------------------------------------------
def mbsp_complex(
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
def mbsp_simple(image: ee.Image, mask_c: ee.Image, mask_mbsp: ee.Image) -> ee.Image:
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
def mbsp_sgx(image: ee.Image, mask_c: ee.Image, mask_mbsp: ee.Image) -> ee.Image:
    """
    MBSP variant that estimates a **local, glint-aware scaling factor**

        α(x,y) = a + b·SGI(x,y)

    Pixels **in mask_c** use that per-pixel α.
    Pixels **only in mask_mbsp** fall back to the C-mask-average ratio.
    """
    # 1. Restrict regression to high-quality C-mask pixels
    C_masked_img = image.updateMask(mask_c)

    # 2. Compute r = B12 / B11 and fit r ~ (1, SGI) on C-mask
    ratio = (
        C_masked_img.select("B12")
        .divide(C_masked_img.select("B11").add(1e-6))
        .rename("ratio")
    )
    reg_img = ee.Image.cat(ee.Image.constant(1), C_masked_img.select("SGI"), ratio)
    lr = reg_img.reduceRegion(
        ee.Reducer.linearRegression(numX=2, numY=1),
        geometry=C_masked_img.geometry(),
        scale=20,
        bestEffort=True,
    )
    coef = ee.Array(lr.get("coefficients"))
    a = coef.get([0, 0])
    b = coef.get([1, 0])

    # 3. SGI-based α across the whole scene (no mask applied here)
    alpha_full = ee.Image.constant(a).add(
        ee.Image.constant(b).multiply(image.select("SGI"))
    )

    # 4. C-mask-average ratio to use where SGI is unavailable (outside mask_c)
    ratio_mean = ee.Number(
        ratio.reduceRegion(
            ee.Reducer.mean(),
            geometry=C_masked_img.geometry(),
            scale=20,
            bestEffort=True,
        ).get("ratio")
    )
    alpha = alpha_full.where(mask_c.eq(0), ee.Image.constant(ratio_mean))

    # 5. MBSP
    MBSP_masked_img = image.updateMask(mask_mbsp)
    mbsp = (
        MBSP_masked_img.select("B12")
        .subtract(alpha.multiply(MBSP_masked_img.select("B11")))
        .divide(alpha.multiply(MBSP_masked_img.select("B11")).add(1e-6))
        .rename("MBSP")
        .set({"a_coef": a, "b_coef": b, "mean_ratio": ratio_mean})
    )

    return mbsp.copyProperties(image, ["system:index", "system:time_start"])


# ------------------------------------------------------------------
def snr_boost(
    image: ee.Image,
    mbsp_img: ee.Image,
    local_sga_range: tuple[float, float],
    window: int = 3,
    alpha: float = 1000.0,
    percentile_norm: float = 98.0,
) -> ee.Image:
    """
    Estimate per-pixel SNR quality for MBSP methane detection and weight the MBSP field.

    Parameters
    ----------
    image : ee.Image
        Sentinel-2 granule containing TOA bands (must include B2, B3, B4, B8, B11, B12)
        and a per-pixel sun-glint-angle band (default name 'SGA').
    mbsp_img : ee.Image
        Image that already contains a 'MBSP' band (fractional methane absorption).
    window : int, optional
        Size (in *pixels*) of the square neighbourhood for uniformity adjustment.
        Default = 3 (i.e. 3 x 3 window).
    alpha : float, optional
        Maximum fractional boost applied to water pixels under full glint.
        alpha = 1 ⇒ a 100 % boost (factor 2) when glint weight = 1.
    sga_cutoff : float, optional
        SGA (degrees) above which the geometric glint weight becomes 0.
    percentile_norm : float, optional
        High-percentile used to rescale quality raster to 0-1.

    Returns
    -------
    ee.Image
        Two-band image:
          • 'SNR_Q'   - quality raster (0-1)
          • 'MBSP_snr'- MBSP values weighted by SNR_Q
    """

    # ------------------------------------------------------------------
    # 1. Basic reflectance signal (RS) - average of SWIR bands
    B11 = image.select("B11")
    B12 = image.select("B12")
    rs = B11.add(B12).multiply(0.5)  # mean SWIR reflectance

    # ------------------------------------------------------------------
    # 3. Sun-Glint Index (SGI) using B8 and a visible composite B_vis
    #    If the image already has 'B_vis', use it; else make it from B2,B3,B4
    b_vis = image.select(["B2", "B3", "B4"]).reduce(ee.Reducer.mean()).rename("B_vis")
    nir = image.select("B8")
    sgi = nir.subtract(b_vis).divide(nir.add(b_vis).add(1e-6)).rename("SGI")

    # Empirical SGI weight in [0,1]   (map SGI = −1 → 0  ;  SGI = +1 → 1)
    w_sgi = sgi.add(1).multiply(0.5).clamp(0, 1)

    # ------------------------------------------------------------------
    # 4. Geometric sun-glint weight from SGA
    sga = image.select("SGA")
    w_sga = sga.multiply(-1.0 / local_sga_range[1]).add(1).clamp(0, 1)  # linear 0-1

    # Combined glint weight - max of the two factors
    w_glint = w_sgi.max(w_sga)

    # ------------------------------------------------------------------
    # 5. Boost reflectance signal for water pixels under glint
    rs_boosted = rs.multiply(ee.Image(1).add(w_glint.multiply(alpha)))  # (1 + α W) * RS

    # ------------------------------------------------------------------
    # 6. Uniformity (texture) adjustment - coefficient of variation over window
    # Build kernel in *pixels* (Sentinel-2 SWIR is 20 m/pixel)
    half = window // 2
    # ker = ee.Kernel.square(half, "pixels", normalize=False)
    ker = ee.Kernel.circle(half, "pixels", normalize=False)

    local_mean = rs_boosted.reduceNeighborhood(ee.Reducer.mean(), kernel=ker)
    local_std = rs_boosted.reduceNeighborhood(ee.Reducer.stdDev(), kernel=ker)
    cv = local_std.divide(local_mean.add(1e-6))  # coefficient of variation
    w_tex = cv.add(1).pow(-1)  # = 1 / (1 + cv)  in [0,1]

    # ------------------------------------------------------------------
    # 7. Raw quality raster (before global scaling)
    q_raw = rs_boosted.multiply(w_tex)

    # ------------------------------------------------------------------
    # 8. Scale quality raster to 0-1 using high percentile of the clear scene
    #    (avoid bright clouds dominating the scale)
    q_percentile = (
        q_raw.reduceRegion(
            ee.Reducer.percentile([percentile_norm]),
            geometry=image.geometry(),
            scale=20,
            bestEffort=True,
            maxPixels=1e8,
        )
        .values()
        .get(0)
    )
    norm = ee.Number(q_percentile).max(1e-3)  # guard against 0
    snr_q = q_raw.divide(norm).clamp(0, 1).rename("SNR_Q")

    # ------------------------------------------------------------------
    # 9. Apply quality raster to MBSP field
    mbsp_weighted = ee.Image(mbsp_img).select("MBSP").multiply(snr_q).rename("MBSP")

    # ------------------------------------------------------------------
    # 10. Return two-band result
    # out = snr_q.addBands(mbsp_weighted)
    out = mbsp_weighted
    return out.copyProperties(image, ["system:index", "system:time_start"])
