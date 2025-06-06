# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: methane
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MultiBand Single Pass (MBSP) Demonstration
#
# This notebook demonstrates a basic implementation of the MBSP algorithm
# described in *Varon et al. (2021)* for detecting large methane plumes using
# Sentinel‑-2 imagery. The MBSP method compares band‑11 and band‑12
# reflectances for a single scene to retrieve methane column enhancements.
# We keep core routines in `src/mbsp.py` for reuse, while image selection
# and experimentation remain in the notebook.

# %% [markdown]
# ## MBSP versus MBMP
#
# MBSP uses two spectral bands from a single acquisition. A scaling coefficient
# is fitted between bands 11 and 12 over the scene and the fractional absorption
# is computed as:
#
# \begin{align}
# R_{MBSP} = \frac{c R_{12} - R_{11}}{R_{11}}
# \end{align}
#
# where $c$ is the slope from fitting $R_{12}$ to $R_{11}$. This approach relies
# on the surface behaving similarly in both bands.
#
# The MultiBand MultiPass (MBMP) technique performs the MBSP retrieval on two
# different dates and subtracts them to reduce artifacts. MBMP generally provides
# better precision when a good plume‑free reference image is available.

# %%
import datetime as dt

import ee
import geemap
import xarray as xr

# Authenticate with Earth Engine
ee.Authenticate()
ee.Initialize()


# %% [markdown]
# ## Helper Functions

# %%
def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using the QA60 band."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    masked = image.updateMask(mask).divide(10000)
    return masked.copyProperties(image, image.propertyNames())  # <-- keep metadata!


# %%
def mbsp_fractional_image(image: ee.Image, region: ee.Geometry) -> ee.Image:
    """
    Compute the MBSP (Multi-Band Single-Pass) fractional methane signal
    for a Sentinel-2 scene following Varon et al. 2021 (Atmos. Meas. Tech.).

    MBSP definition
    ----------------
        R_MBSP = ( c · R12  –  R11 ) / R11

    where
        R11, R12 : TOA reflectances for Sentinel-2 bands 11 (1610 nm) and 12 (2190 nm)
        c        : scene-wide slope from a zero-intercept linear regression of
                   R11 on R12 (c = Σ R11·R12 / Σ R12²)

    R_MBSP isolates the methane-induced absorption in band 12 by:
      1) Bringing band 12 onto the radiometric scale of band 11 (multiplying by c)
      2) Differencing and normalising by R11.

    A *negative* R_MBSP indicates that band 12 is darker than expected and is
    therefore consistent with CH₄ absorption.

    Parameters
    ----------
    image  : ee.Image
        Sentinel-2 L1C/L2A image containing bands ‘B11’ and ‘B12’.
    region : ee.Geometry
        Area (ideally plume-free) used to derive the scene-wide slope *c*.

    Returns
    -------
    ee.Image
        Single-band image named ‘R’ holding the pixel-wise MBSP signal.
        The fitted slope *c* is attached as image property ‘slope’.
    """
    # --- 1. Numerator and denominator for the regression slope -----------------
    # c = Σ( R11 * R12 ) / Σ( R12² )
    num_img = image.select("B11").multiply(image.select("B12"))  # R11·R12
    den_img = image.select("B12").multiply(image.select("B12"))  # R12²

    num_sum = num_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=20, bestEffort=True
    )
    den_sum = den_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=20, bestEffort=True
    )

    # Convert to ee.Number for further arithmetic
    slope = ee.Number(num_sum.get("B11")).divide(ee.Number(den_sum.get("B12")))

    # --- 2. Per-pixel MBSP field ----------------------------------------------
    mbsp = (
        image.select("B12")
        .multiply(slope)  # c · R12
        .subtract(image.select("B11"))
        .divide(image.select("B11"))
        .rename("R")
    )

    # Embed the slope for traceability
    return mbsp.set({"slope": slope})



# %% [markdown]
# ## Image Selection

# %%
# Location and date range of interest
lat, lon = 31.6585, 5.9053  # Hassi Messaoud example
start = dt.date(2019, 10, 1)
end = dt.date(2019, 10, 15)

point = ee.Geometry.Point(lon, lat)
collection = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterDate(str(start), str(end))
    .filterBounds(point)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .sort("system:time_start")
    .map(mask_s2_clouds)
)

images = collection.toList(collection.size())
count = images.size().getInfo()
print(f"Found {count} images")

# %%
if count:
    region = point.buffer(1000).bounds()
    m = geemap.Map(center=(lat, lon), zoom=12)
    for i in range(count):
        img = ee.Image(images.get(i))
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        r_img = mbsp_fractional_image(img, region)
        m.addLayer(
            r_img,
            {"min": -0.05, "max": 0.05, "palette": ["blue", "white", "red"]},
            f"{date} fractional",
            True,
        )
        rgb = img.select(["B4", "B3", "B2"])
        m.addLayer(rgb, {"min": 0, "max": 0.3}, f"{date} RGB", False)

    styled_pt = ee.FeatureCollection([point]).style(
        **{
            "color": "green",  # outline & fill
            "fillColor": None,  # no fill
            "pointSize": 8,  # pixel radius of the dot
        }
    )
    m.addLayer(styled_pt, {}, "centre pt", True)

# %%
m

# %% [markdown]
# The map above contains fractional signal layers for each scene interleaved
# with true-color imagery. In the second half we apply the per-pixel water-vapour correction and robust MBSP retrieval described earlier to get CH₄ enhancement maps from a single image.


# %%

# lookup helper --------------------------------------------------------------

def k_lookup(lut_ds, band, gas, sza, vza, pres):
    """Return scalar k_{j,g} from a local xarray LUT."""
    return float(
        lut_ds.k_prime.sel(band=band, gas=gas)
        .interp(sza=sza, vza=vza, pres=pres)
        .values
    )

# load absorption-coefficient tables (once per session)
lutA = xr.open_dataset("../data/lookup/k_S2A_v1.nc")
lutB = xr.open_dataset("../data/lookup/k_S2B_v1.nc")

# ---------------------------------------------------------------------------
# cloud masking (reuse earlier helper)
# ---------------------------------------------------------------------------

def _mask_clouds(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).neq(0)
    cirrus = qa.bitwiseAnd(1 << 11).neq(0)
    mask = cloud.Or(cirrus)
    return img.updateMask(mask.Not()).divide(10000)

# ---------------------------------------------------------------------------
# per-pixel water vapour correction
# ---------------------------------------------------------------------------

def _correct_water(img: ee.Image, region: ee.Geometry, k9, k11, k12) -> ee.Image:
    # Bring band 9 to the 20 m grid used by bands 11/12
    b9 = (
        img.select("B9")
        .resample("bicubic")
        .reproject(img.select("B11").projection())
    )
    ratio = b9.divide(img.select("B8A"))
    delta_tau = ratio.log().multiply(-1)
    W = delta_tau.divide(k9)
    tau11 = W.multiply(k11)
    tau12 = W.multiply(k12)
    m11 = tau11.reduceRegion(ee.Reducer.mean(), region, 20, bestEffort=True).values().get(0)
    m12 = tau12.reduceRegion(ee.Reducer.mean(), region, 20, bestEffort=True).values().get(0)
    # remove scene-mean water vapour while keeping absolute scale
    r11 = img.select("B11").multiply(tau11.subtract(m11).exp())
    r12 = img.select("B12").multiply(tau12.subtract(m12).exp())
    return r11.rename("B11c").addBands(r12.rename("B12c"))

# ---------------------------------------------------------------------------
# robust slope estimation (two iterations)
# ---------------------------------------------------------------------------

def _slope(b11: ee.Image, b12: ee.Image, region: ee.Geometry) -> ee.Number:
    num = b11.multiply(b12).reduceRegion(ee.Reducer.sum(), region, 20, bestEffort=True).values().get(0)
    den = b12.multiply(b12).reduceRegion(ee.Reducer.sum(), region, 20, bestEffort=True).values().get(0)
    return ee.Number(num).divide(ee.Number(den))


def _robust_slope(b11: ee.Image, b12: ee.Image, region: ee.Geometry) -> ee.Number:
    c1 = _slope(b11, b12, region)
    resid = b12.multiply(c1).subtract(b11).divide(b11)
    mask = resid.gt(-0.01)
    c2 = _slope(b11.updateMask(mask), b12.updateMask(mask), region)
    return c2

# ---------------------------------------------------------------------------
# main retrieval
# ---------------------------------------------------------------------------

def ch4_mbsp_single(image: ee.Image, region: ee.Geometry) -> ee.Image:
    img = _mask_clouds(image)
    sat = image.get("SPACECRAFT_NAME").getInfo()
    sza = image.get("MEAN_SOLAR_ZENITH_ANGLE").getInfo()
    vza = 0.0
    pres = 1013.0
    lut = lutA if "2A" in sat else lutB
    k9 = k_lookup(lut, "B9", "H2O", sza, vza, pres)
    k11 = k_lookup(lut, "B11", "H2O", sza, vza, pres)
    k12 = k_lookup(lut, "B12", "H2O", sza, vza, pres)
    k11_ch4 = ee.Number(k_lookup(lut, "B11", "CH4", sza, vza, pres))
    k12_ch4 = ee.Number(k_lookup(lut, "B12", "CH4", sza, vza, pres))

    dry = _correct_water(img, region, k9, k11, k12)
    b11c = dry.select("B11c")
    b12c = dry.select("B12c")

    c = _robust_slope(b11c, b12c, region)
    r_frac = b12c.multiply(c).subtract(b11c).divide(b11c).rename("R")
    denom = k12_ch4.subtract(c.multiply(k11_ch4))
    dch4 = r_frac.multiply(-1).divide(denom).rename("dCH4")
    return ee.Image.cat([r_frac, dch4]).set({"slope": c})


def debug_ch4_mbsp(image: ee.Image, region: ee.Geometry) -> ee.Image:
    """Run the single-scene MBSP retrieval with verbose diagnostics."""
    img = _mask_clouds(image)
    sat = image.get("SPACECRAFT_NAME").getInfo()
    sza = image.get("MEAN_SOLAR_ZENITH_ANGLE").getInfo()
    vza = 0.0
    pres = 1013.0
    lut = lutA if "2A" in sat else lutB
    k9 = k_lookup(lut, "B9", "H2O", sza, vza, pres)
    k11 = k_lookup(lut, "B11", "H2O", sza, vza, pres)
    k12 = k_lookup(lut, "B12", "H2O", sza, vza, pres)
    print("k9, k11, k12 =", k9, k11, k12)
    dry = _correct_water(img, region, k9, k11, k12)

    b11c = dry.select("B11c")
    b12c = dry.select("B12c")
    mean_tau11 = b11c.subtract(img.select("B11")).reduceRegion(
        ee.Reducer.mean(), region, 20, bestEffort=True
    ).values().get(0)
    mean_tau12 = b12c.subtract(img.select("B12")).reduceRegion(
        ee.Reducer.mean(), region, 20, bestEffort=True
    ).values().get(0)
    print("mean delta B11, B12 =", ee.Number(mean_tau11).getInfo(), ee.Number(mean_tau12).getInfo())

    c = _robust_slope(b11c, b12c, region)
    print("robust slope =", c.getInfo())
    r_frac = b12c.multiply(c).subtract(b11c).divide(b11c).rename("R")

    # also compute uncorrected fractional signal for comparison
    c0 = _robust_slope(img.select("B11"), img.select("B12"), region)
    r_frac0 = (
        img.select("B12").multiply(c0).subtract(img.select("B11")).divide(img.select("B11"))
    )
    diff = (
        r_frac.subtract(r_frac0)
        .reduceRegion(ee.Reducer.mean(), region, 20, bestEffort=True)
        .values()
        .get(0)
    )
    print("mean fractional difference =", ee.Number(diff).getInfo())

    return r_frac



# %%
if count:
    region = point.buffer(1000).bounds()
    m = geemap.Map(center=(lat, lon), zoom=12)
    for i in range(count):
        img = ee.Image(images.get(i))
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        # print debugging output
        debug_ch4_mbsp(img, region)
        r_img = ch4_mbsp_single(img, region).select("R")
        m.addLayer(
            r_img,
            {"min": -0.05, "max": 0.05, "palette": ["blue", "white", "red"]},
            f"{date} fractional",
            True,
        )
        rgb = img.select(["B4", "B3", "B2"])
        m.addLayer(rgb, {"min": 0, "max": 0.3}, f"{date} RGB", False)
    styled_pt = ee.FeatureCollection([point]).style(color="green", fillColor=None, pointSize=8)
    m.addLayer(styled_pt, {}, "centre pt", True)


# %%
m
