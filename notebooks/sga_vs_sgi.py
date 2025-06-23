# Imports and init
# %%
import os
import sys
from offshore_methane.sga import compute_sga_coarse
from pathlib import Path

import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
from tqdm import tqdm

module_path = os.path.abspath(os.path.join("..", "offshore_methane"))
if module_path not in sys.path:
    sys.path.append(module_path)
from ee_utils import add_sgi, add_sgi_b3, add_sga_ok, _cloud_color_mask
import ee
import geemap

ee.Authenticate()
ee.Initialize()

# Step 3: Define functions used for creating plume maps.


def sample_glint_vs_sgi(
    system_index: str,
    glint_alpha_path: str,
    n_points: int = 500,
    aoi: ee.Geometry = None,
    use_single_band: str = None,
):
    """
    Samples 'sgi' from Earth Engine and 'glint_alpha' from a local GeoTIFF, then plots them.

    Args:
        system_index (str): Sentinel-2 system:index string.
        glint_alpha_path (str): Path to local GeoTIFF with glint_alpha values.
        n_points (int): Number of random points to sample.
        aoi (ee.Geometry, optional): Area of interest to restrict sampling. Defaults to the image footprint.

    Returns:
        GeoDataFrame with columns: lon, lat, sgi, glint_alpha
    """
    # ---- 1. Load Sentinel-2 image and add 'sgi' band ----
    s2_image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", system_index))
        .first()
    )
    if s2_image is None:
        raise ValueError(f"No image found for system:index = {system_index}")

    if use_single_band is not None:
        image_with_sgi = s2_image.select(use_single_band)
    else:
        use_single_band = "sgi"
        image_with_sgi = add_sgi(s2_image).select("sgi")

    # ---- 2. Use provided AOI or fallback to image geometry ----
    sample_region = aoi or s2_image.geometry()

    # ---- 3. Generate random points in AOI ----
    points_fc = ee.FeatureCollection.randomPoints(
        sample_region, points=n_points, seed=42
    )

    # ---- 4. Sample 'sgi' values at points ----
    sampled = image_with_sgi.sampleRegions(
        collection=points_fc, scale=1, geometries=True
    )
    sampled_info = sampled.getInfo()

    data = []
    for feat in sampled_info["features"]:
        coords = feat["geometry"]["coordinates"]
        sgi_val = feat["properties"].get(use_single_band)
        if sgi_val is not None:
            data.append({"lon": coords[0], "lat": coords[1], use_single_band: sgi_val})

    df = pd.DataFrame(data)

    # ---- 5. Sample 'glint_alpha' from local GeoTIFF ----
    with rasterio.open(glint_alpha_path) as src:
        crs = src.crs
        df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(crs)

        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        gdf["glint_alpha"] = [
            val[0] if val is not None else np.nan for val in src.sample(coords)
        ]

    # ---- 6. Filter and plot ----
    gdf = gdf.dropna(subset=["glint_alpha", use_single_band])

    plt.scatter(gdf["glint_alpha"], gdf[use_single_band], alpha=0.5)
    plt.xlabel("glint_alpha")
    plt.ylabel(use_single_band)
    plt.title(f"Glint Alpha vs {use_single_band} for {system_index}")
    plt.grid(True)
    plt.show()

    return gdf


def linear_fit(img):
    """
    Perform a linear regression of B11 vs. B12 in the AOI to estimate coefficient c_fit.
    Sets the coefficient and coefficient list as properties on the image.
    """
    # Prepare the regression: stack B12 (x) and B11 (y).
    x_var = img.select("B12")
    y_var = img.select("B11")
    stacked = ee.Image.cat([x_var, y_var])

    # Run a linear regression reducer with numX=1, numY=1.
    fit_dict = stacked.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=1, numY=1),
        geometry=aoi,
        scale=20,
        maxPixels=1e10,
        bestEffort=True,
    )
    # The result is a dictionary containing an “array” of coefficients.
    coeff_array = ee.Array(fit_dict.get("coefficients"))
    coeff_list = coeff_array.toList().get(0)
    c0 = ee.List(coeff_list).get(0)

    return img.set({"c_fit": c0, "coef_list": coeff_list})


def MBSP(img):
    """
    Calculate the MBSP ratio (IR_comp) per pixel:
        IR_comp = ((c * B12) - B11) / B11
    where c is the regression coefficient stored in the image properties.
    """
    c = ee.Number(img.get("c_fit"))
    formula = (
        ee.Image(img.select("B12"))
        .multiply(c)
        .subtract(img.select("B11"))
        .divide(img.select("B11"))
    ).rename("IR_comp")
    return img.addBands(formula)


def mask_land(image):
    """
    Use JAXA/GCOM-C LST as a land mask. We take the mean over Jan 2024,
    then mask out any pixel where LST_AVE is non-zero (i.e., keep non-land).
    """
    # Filter the land surface temperature collection to January 2024.
    lst_collection = ee.ImageCollection("JAXA/GCOM-C/L3/LAND/LST/V3").filterDate(
        "2024-01-01", "2024-02-01"
    )
    lst_mosaic = lst_collection.mean()
    # Any pixel with a non-zero LST_AVE is considered land.
    land_masker = lst_mosaic.select("LST_AVE").reduce(ee.Reducer.anyNonZero())

    # Keep only pixels where land_masker == 0 (i.e., water or non-land).
    return image.updateMask(land_masker.unmask(0).eq(0))


# Step 1: Set universal values here.
rgb_vis = {
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 2500,
}

# Step 2: Define study area.

# Sample plume locations, provided by UNEP IMEO.
plume_locs = ee.FeatureCollection(
    "projects/ee-brendan-skytruth/assets/unepIMEO_plumes_sentinel2"
)

# Convert the collection to a list so we can index into it.
p_size = plume_locs.size()
plume_list = plume_locs.toList(plume_locs.size())

# Brendan's assessments of plumes reviewed in this UNEP dataset:
# 5, 10, 13, 18, 20, 21 are good.
# 7 is really good.
# 12, 28, 31, 33,  is dirty, but boost up palette and draw geometry around sus area.
# 17, 19, 27, 34 might be good? They're ambiguous to me.
index = 0
glint_alphas = []
sgi_means = []
sgi_means_b3 = []

sgi_pixels = []
sga_pixels = []
sid_pixels = []
# %%
for index in tqdm(range(p_size.getInfo())):
    # for index in range(41):
    platform = ee.Feature(plume_list.get(index))
    aoi = platform.geometry().buffer(10000).bounds()
    plume_date = ee.Date(platform.get("start_date"))
    print("Plume start date:", plume_date.getInfo())
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(aoi)
        # .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
        .filterDate(plume_date.format("YYYY-MM-dd"), plume_date.advance(2, "day"))
    )
    img = ee.Image(s2_collection.first())
    # img = linear_fit(img)
    # img = MBSP(img)
    img = img.clip(aoi)

    radius_of_interest = 5000
    img = add_sga_ok(img)
    glint_alpha = img.get("SGA_DEG").getInfo()
    print("glint_alpha:", glint_alpha)
    img = add_sgi(img)
    img = add_sgi_b3(img)

    mean_dict = img.select("sgi").reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=platform.buffer(radius_of_interest).geometry(),
        bestEffort=True,  # optionally optimize for large geometries
    )

    sgi_mean = mean_dict.get("sgi").getInfo()

    mean_dict_b3 = img.select("sgi_b3").reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=platform.buffer(radius_of_interest).geometry(),
        bestEffort=True,  # optionally optimize for large geometries
    )

    sgi_mean_b3 = mean_dict_b3.get("sgi_b3").getInfo()

    print("SGI mean:", sgi_mean)
    print("SGI b3 mean:", sgi_mean_b3)
    glint_alphas.append(glint_alpha)
    sgi_means.append(sgi_mean)
    sgi_means_b3.append(sgi_mean_b3)
    sid = img.get("system:index").getInfo()
    print(sid)

    prefixed = f"SGA_{sid}"
    tif_path = Path("../data") / "sga" / f"{prefixed}.tif"
    # glint_vs_sgi_gdf = sample_glint_vs_sgi(
    #     sid, tif_path, n_points=500, aoi=platform.buffer(radius_of_interest).geometry()
    # )

    if not tif_path.is_file():
        print(f"  ↻ computing *coarse* SGA grid for {sid}")
        compute_sga_coarse(sid, tif_path)

    glint_vs_sgi_gdf = sample_glint_vs_sgi(
        sid, tif_path, n_points=5000, use_single_band="B12"
    )
    # sgi_pixels.extend(list(glint_vs_sgi_gdf["sgi"].values))
    sgi_pixels.extend(list(glint_vs_sgi_gdf["B12"].values))
    sga_pixels.extend(list(glint_vs_sgi_gdf["glint_alpha"].values))
    sid_pixels.extend([sid] * len(glint_vs_sgi_gdf))

    # %%
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 6))
    # plt.scatter(glint_alphas, sgi_means, label="Sun Glint Index", color="blue", alpha=0.7)
    # plt.scatter(
    #     glint_alphas,
    #     sgi_means_b3,
    #     label="Sun Glint Index (GREEN)",
    #     color="green",
    #     alpha=0.7,
    # )
    # plt.xlabel("Sun Glint Alpha")
    # plt.ylabel("SGI Mean")
    # plt.title("SGI Mean vs Sun Glint Alpha")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # %%

    # %%
# index = 0
# # index = 28
# platform = ee.Feature(plume_list.get(index))
# aoi = platform.geometry().buffer(10000).bounds()
# plume_date = ee.Date(platform.get("start_date"))
# print("Plume start date:", plume_date.getInfo())
# s2_collection = (
#     ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
#     .filterBounds(aoi)
#     # .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
#     .filterDate(plume_date.format("YYYY-MM-dd"), plume_date.advance(2, "day"))
# )
# sgi = add_sgi(s2_collection.first())
# Map = geemap.Map()
# Map.addLayer(s2_collection.first(), rgb_vis, "Sentinel-2 True-Colour", True)
# Map.addLayer(sgi.select("sgi"), {}, "actual sgi")
# Map.centerObject(platform, 9)
# Map

# %%


import matplotlib.pyplot as plt

# Assign a unique color to each sid using a colormap
unique_sids = list(set(sid_pixels))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sids)))
sid_to_color = {sid: colors[i] for i, sid in enumerate(unique_sids)}

plt.figure(figsize=(8, 6))
for sid in unique_sids:
    idx = [i for i, s in enumerate(sid_pixels) if s == sid]
    plt.scatter(
        [sga_pixels[i] for i in idx],
        [sgi_pixels[i] for i in idx],
        color=sid_to_color[sid],
        alpha=0.008,
        label=str(sid),
    )

plt.xlabel("glint_alpha")
plt.ylabel("sgi")
plt.title("Glint Alpha vs SGI (colored by sid)")
# plt.legend(title="sid", loc="upper right")
plt.tight_layout()
plt.show()

# %%
# # Save sgi_pixels, sga_pixels, sid_pixels as a DataFrame and export to CSV
# df = pd.DataFrame({
#     "sgi": sgi_pixels,
#     "glint_alpha": sga_pixels,
#     "sid": sid_pixels
# })
# csv_path = Path("../data") / "sgi_vs_sga_pixels.csv"
# df.to_csv(csv_path, index=False)
# print(f"Saved DataFrame to {csv_path}")
# %%

sga_pixels_arr = np.array(sga_pixels)
sgi_pixels_arr = np.array(sgi_pixels)

# coeffs = np.polyfit(sga_pixels_arr, sgi_pixels_arr, deg=1)  # degree 1 = linear
# m, b = coeffs
# sgi_pixels_fit = m * sga_pixels_arr + b

poly_coeffs = np.polyfit(sga_pixels_arr, sgi_pixels_arr, deg=5)

# # Turn into function
p = np.poly1d(poly_coeffs)
sgi_pixels_fit = p(sga_pixels_arr)

plt.scatter(sga_pixels, sgi_pixels, alpha=0.005)
# plt.scatter(np.linspace(10, 45), m * np.linspace(10, 45) + b)
plt.scatter(np.linspace(10, 45), p(np.linspace(10, 45)))
## %%
system_index = "20240913T162829_20240913T163739_T15QWB"
s2_image = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filter(ee.Filter.eq("system:index", system_index))
    .first()
)
