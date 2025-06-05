# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
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
import numpy as np
import matplotlib.pyplot as plt

from offshore_methane import mbsp

# %% [markdown]
# ## Helper Functions

# %%

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using the QA60 band."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000)


# %% [markdown]
# ## Image Selection

# %%
# Location and date range of interest
lat, lon = 31.6585, 5.9053  # Hassi Messaoud example
start = dt.date(2019, 10, 1)
end = dt.date(2019, 10, 31)

point = ee.Geometry.Point(lon, lat)
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate(str(start), str(end))
    .filterBounds(point)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_s2_clouds)
)

images = collection.toList(collection.size())
count = images.size().getInfo()
print(f"Found {count} images")

# %% [markdown]
# ## MBSP Retrieval for One Image

# %%
if count:
    image = ee.Image(images.get(0))
    region = point.buffer(1000).bounds()  # 2 km region
    bands = ["B11", "B12"]
    arr = np.asarray(geemap.ee_to_numpy(image, region=region, bands=bands, scale=20))
    b11 = arr[:, :, 0]
    b12 = arr[:, :, 1]
    c, r = mbsp.mbsp_fractional_absorption(b11, b12)
    delta = mbsp.invert_mbsp(r, mbsp.S2_CONSTANTS["A"])
    print("MBSP slope:", c)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("MBSP Fractional Signal")
    plt.imshow(r, cmap="RdBu", vmin=-0.05, vmax=0.05)
    plt.colorbar(label="R")
    plt.subplot(1, 2, 2)
    plt.title("Retrieved \u0394CH$_4$")
    plt.imshow(delta, cmap="inferno", vmin=0, vmax=1.0)
    plt.colorbar(label="mol m$^{-2}$")
    plt.tight_layout()

# %% [markdown]
# The example above selects the first image in the stack and computes methane
# enhancements using the MBSP equations. Further analysis could loop over all
# images in `collection` to build a time series of emissions.

