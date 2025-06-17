# Imports and init
# %%
import os
import sys

module_path = os.path.abspath(os.path.join("..", "offshore_methane"))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils import calculateSunglint_alpha
import ee
import geemap

ee.Authenticate()
ee.Initialize()

# Step 3: Define functions used for creating plume maps.


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
index = 20
mbsp_visRange = 0.3

# Step 2: Define study area.

# Sample plume locations, provided by UNEP IMEO.
plume_locs = ee.FeatureCollection(
    "projects/ee-brendan-skytruth/assets/unepIMEO_plumes_sentinel2"
)

# Convert the collection to a list so we can index into it.
plume_list = plume_locs.toList(plume_locs.size())

# Brendan's assessments of plumes reviewed in this UNEP dataset:
# 5, 10, 13, 18, 20, 21 are good.
# 7 is really good.
# 12, 28, 31, 33,  is dirty, but boost up palette and draw geometry around sus area.
# 17, 19, 27, 34 might be good? They're ambiguous to me.
platform = ee.Feature(plume_list.get(index))

# Define the area of interest (AOI) by buffering the platform location by 10 km,
# then taking the bounding box.
aoi = platform.geometry().buffer(10000).bounds()

# Create a Map centered on the platform (zoom level 15).
Map = geemap.Map()

Map.centerObject(aoi)


rgb_vis = {
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 2500,
}

# Visual parameters for the MBSP ratio (IR_comp).
ir_vis = {
    "bands": ["IR_comp"],
    "min": mbsp_visRange * -1,
    "max": mbsp_visRange,
    "palette": ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"],
}

# Visual parameters for the enhancement (omega) band.
enhancement_vis = {
    "bands": ["omega"],
    "min": 0.000000001,
    "max": 0.000002,
    "palette": ["#4B2991", "#952EA0", "#D44292", "#F66D7A", "#F6A97A"],
}


# Step 4: Image Collection setup.

# Extract the plume’s “start_date” property (as an ee.Date).
plume_date = ee.Date(platform.get("start_date"))

# Print plume_date for debugging (in a Python context, we can get it with getInfo())
print("Plume start date:", plume_date.getInfo())

# Build the Sentinel-2 Harmonized ImageCollection over the AOI,
# for a 2-day window starting at plume_date, with cloud cover < 90%.
s2_collection = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterBounds(aoi)
    # .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
    .filterDate(plume_date.format("YYYY-MM-dd"), plume_date.advance(2, "day"))
)

# For demonstration, select the first (least cloudy) image in the filtered collection.
img = ee.Image(s2_collection.first())

# Apply regression to estimate “c_fit,” then compute MBSP ratio, then clip to AOI.
img = linear_fit(img)
img = MBSP(img)
img = img.clip(aoi)


# Step 5: Map display.

# Add true-color (RGB) layer.
Map.addLayer(img, rgb_vis, "Sentinel-2 True-Colour", True)

# Add MBSP ratio (IR_comp) layer.
Map.addLayer(img.select("IR_comp"), ir_vis, "MBSP Ratio (IR_comp)", True)

# Add individual SWIR bands for reference (B12 and B11).
Map.addLayer(
    img.select("B12").multiply(ee.Image.constant(img.get("c_fit"))),
    {"bands": ["B12"], "min": 0, "max": 3000},
    "SWIR-2 (B12)",
    False,
)
Map.addLayer(
    img.select("B11"),
    {"bands": ["B11"], "min": 0, "max": 3000},
    "SWIR-1 (B11)",
    False,
)


# Step 6: Legend for MBSP raster.

# geemap has a built-in function to add a colorbar legend.
# We create a JavaScript-like thumbnail image for the palette, then add labels.


def make_colorbar(palette, vmin, vmax):
    """
    Create a colorbar image from 0–1, applying the given palette.
    Returns an ee.Image suitable for a UI thumbnail.
    """
    color_bar = ee.Image.pixelLonLat().select("latitude")
    return color_bar.visualize(min=0, max=1, palette=palette, forceRgbOutput=True)


# Create a small colorbar thumbnail (1 px high by 100 px wide).
colorbar_img = make_colorbar(ir_vis["palette"], ir_vis["min"], ir_vis["max"])
colorbar_params = {
    "min": 0,
    "max": 1,
    "palette": ir_vis["palette"],
}
colorbar_thumbnail = (
    ee.Image.pixelLonLat().select("latitude").visualize(**colorbar_params)
)

# Add the colorbar thumbnail as an overlay
# Map.add_ee_layer(
#     colorbar_thumbnail,
#     {"opacity": 0},
#     "MBSP Colorbar",
# )

# Add legend labels manually using geemap’s built-in add_legend function.
# Note: geemap’s add_legend expects a dictionary of value: color strings.
# We’ll interpolate three labels: min, midpoint, max.
Map.add_legend(
    title="MBSP Ratio",
    legend_dict={
        f"{ir_vis['min']}": ir_vis["palette"][0],
        f"{(ir_vis['max'] + ir_vis['min']) / 2:.3f}": ir_vis["palette"][2],
        f"{ir_vis['max']}": ir_vis["palette"][-1],
    },
    position="bottomright",
)

# %%
img = calculateSunglint_alpha(img)
print("glint_alpha:", img.get("glint_alpha").getInfo())

# %%
# Add the platform point to the Map for reference.
Map.addLayer(
    ee.FeatureCollection([platform]),
    {"color": "green"},
    "Platform (Emission Source)",
)
Map.addLayerControl()  # Add a layer control panel
Map  # Display the interactive map (in Jupyter or colab)

# %%
