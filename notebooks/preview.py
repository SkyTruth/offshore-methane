# %%
import ee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ee
import geemap
import ipywidgets as widgets
from IPython.display import display
import sys
import os
import math


mbsp_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "offshore_methane"
)
if mbsp_path not in sys.path:
    sys.path.append(mbsp_path)

from mbsp import mbsp_simple_ee
from masking import build_mask_for_C, build_mask_for_MBSP

ee.Authenticate()
ee.Initialize()


def mbsp_decay(x: float) -> float:
    """
    Exponential decay function passing through (12, 0.2) and (22, 0.05).

    Args:
        x (float): Input value.

    Returns:
        float: Output of the decay function.
    """
    A = 1.055
    k = -0.1386294361
    return A * math.exp(k * x)


def calculate_sunglint_alpha(image: ee.Image):
    pi = 3.141592653589793

    # --- Solar geometry (radians) ---
    theta_naught = (
        ee.Number(image.get("MEAN_SOLAR_ZENITH_ANGLE")).multiply(pi).divide(180)
    )
    phi_naught = (
        ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE")).multiply(pi).divide(180)
    )

    # --- View geometry for bands B11 & B12 (radians) ---
    b11_zenith = (
        ee.Number(image.get("MEAN_INCIDENCE_ZENITH_ANGLE_B11")).multiply(pi).divide(180)
    )
    b11_azimuth = (
        ee.Number(image.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B11"))
        .multiply(pi)
        .divide(180)
    )
    b12_zenith = (
        ee.Number(image.get("MEAN_INCIDENCE_ZENITH_ANGLE_B12")).multiply(pi).divide(180)
    )
    b12_azimuth = (
        ee.Number(image.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B12"))
        .multiply(pi)
        .divide(180)
    )

    # --- Mean view zenith (θ) and azimuth (φ) ---
    theta = b11_zenith.add(b12_zenith).divide(2)
    phi = b11_azimuth.add(b12_azimuth).divide(2)

    # --- Intermediate terms (Hedley et al.) ---
    A = theta_naught.add(theta).cos().add(theta_naught.subtract(theta).cos())
    B = theta_naught.add(theta).cos().subtract(theta_naught.subtract(theta).cos())
    C = phi_naught.subtract(phi).cos()

    arg = A.add(B.multiply(C)).divide(2)
    alpha = arg.acos()  # radians
    alpha_deg = alpha.multiply(180 / pi)

    return image.set("glint_alpha", alpha_deg)


def evaluate_offshore_site(image, lat, lon, buffer_m=200):
    # Define point and buffer
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m)

    # Select relevant bands from S2
    b3 = image.select("B3")  # Green
    b8 = image.select("B8")  # NIR
    b11 = image.select("B11")  # SWIR1
    b12 = image.select("B12")  # SWIR2
    b8a = image.select("B8A")  # Narrow NIR

    # --- Cloudiness (using S2 Cloud Probability dataset) ---
    system_index = image.get("system:index")

    cloud_img = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.equals("system:index", system_index))
        .first()
    )

    # Cloud probability band is "probability" (0–100)
    cloud_prob = cloud_img.select("probability")
    cloud_fraction = cloud_prob.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e8
    ).get("probability")

    # --- Structure Presence (via NDWI min) ---
    ndwi = b3.subtract(b8).divide(b3.add(b8))
    ndwi_stats = ndwi.reduceRegion(
        reducer=ee.Reducer.min(), geometry=region, scale=20, maxPixels=1e8
    )
    min_ndwi = ndwi_stats.get("B3")

    structure_present = ee.Algorithms.If(
        ee.Number(min_ndwi).lt(0),
        1,  # structure
        0,  # only ocean
    )

    # --- Flaring Detection (max difference B12 - B11) ---
    # flare_diff = b12.subtract(b11).rename("flare_diff")
    delta_sw = b12.subtract(b11)
    tai = delta_sw.divide(b8a).rename("TAI")

    # Get max value and pixel location
    flare_reduce = tai.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
        bestEffort=True,
    )
    max_flare_diff = flare_reduce.get("TAI")

    # Get coordinates of max flare pixel
    flare_mask = tai.eq(ee.Number(max_flare_diff))
    flare_coords = (
        flare_mask.reduceToVectors(
            scale=20,
            geometryType="centroid",
            maxPixels=1e8,
            bestEffort=True,
        )
        .filterBounds(region)
        .geometry()
        .coordinates()
    )
    flare_coords = ee.List(flare_coords)

    flare_present = ee.Algorithms.If(
        ee.Number(max_flare_diff).gt(0.15),  # threshold lowered
        1,
        0,
    )

    # Return as dictionary
    result = ee.Dictionary(
        {
            "system_index": system_index,
            "lat": lat,
            "lon": lon,
            "cloud_fraction": cloud_fraction,
            "min_ndwi": min_ndwi,
            "structure_present": structure_present,
            "max_TAI": max_flare_diff,
            "flare_present": flare_present,
            "flare_latlon": flare_coords,
        }
    )

    return result


def evaluateScene(image, region):
    # Define point and buffer
    point = region.centroid()
    # region = point.buffer(buffer_m)

    # Select relevant bands from S2
    b3 = image.select("B3")  # Green
    b8 = image.select("B8")  # NIR
    b11 = image.select("B11")  # SWIR1
    b12 = image.select("B12")  # SWIR2
    b8a = image.select("B8A")  # Narrow NIR

    # --- Cloudiness (using S2 Cloud Probability dataset) ---
    system_index = image.get("system:index")

    cloud_img = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.equals("system:index", system_index))
        .first()
    )

    # Cloud probability band is "probability" (0–100)
    cloud_prob = cloud_img.select("probability")
    cloud_fraction = cloud_prob.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e8
    ).get("probability")

    # --- Structure Presence (via NDWI min) ---
    ndwi = b3.subtract(b8).divide(b3.add(b8))
    ndwi_stats = ndwi.reduceRegion(
        reducer=ee.Reducer.min(), geometry=region, scale=20, maxPixels=1e8
    )
    min_ndwi = ndwi_stats.get("B3")

    structure_present = ee.Algorithms.If(
        ee.Number(min_ndwi).lt(0),
        1,  # structure
        0,  # only ocean
    )

    # --- Flaring Detection (max difference B12 - B11) ---
    # flare_diff = b12.subtract(b11).rename("flare_diff")
    delta_sw = b12.subtract(b11)
    tai = delta_sw.divide(b8a).rename("TAI")

    # Get max value and pixel location
    flare_reduce = tai.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
        bestEffort=True,
    )
    max_flare_diff = flare_reduce.get("TAI")

    # Get coordinates of max flare pixel
    flare_mask = tai.eq(ee.Number(max_flare_diff))
    flare_coords = (
        flare_mask.reduceToVectors(
            scale=20,
            geometryType="centroid",
            maxPixels=1e8,
            bestEffort=True,
        )
        .filterBounds(region)
        .geometry()
        .coordinates()
    )
    flare_coords = ee.List(flare_coords)

    flare_present = ee.Algorithms.If(
        ee.Number(max_flare_diff).gt(0.15),  # threshold lowered
        1,
        0,
    )

    # flare_present = ee.Number(0)

    # Return as dictionary
    result = ee.Feature(point).set(
        {
            "system_index": system_index,
            "cloud_fraction": cloud_fraction,
            "min_ndwi": min_ndwi,
            "structure_present": structure_present,
            "max_TAI": max_flare_diff,
            "flare_present": flare_present,
            "flare_latlon": flare_coords,
        }
    )

    return result


def get_image(sid):
    col = ee.ImageCollection("COPERNICUS/S2")
    return col.filter(ee.Filter.eq("system:index", sid)).first()


def show_granule_viewer(
    sid_data,
    b11_b12_max=[1000, 750],
    zoom=12,
    mbsp_min_max=[-0.2, 0.2],
    extra_gdf=None,
):
    index = 0
    m = geemap.Map(center=(sid_data[0]["lat"], sid_data[0]["lon"]), zoom=zoom)
    out = widgets.Output()

    # --- add a slider for B12 max ---
    b12_max_slider = widgets.FloatSlider(
        value=b11_b12_max[1],
        min=100,
        max=5000,
        step=50,
        description="B12 max:",
        continuous_update=False,
    )

    def update_map(idx):
        for lyr_name in ["B11", "B12", "RGB", "MBSP", "Flaring"]:
            if lyr_name in m.ee_layers:
                try:
                    m.remove_layer(m.ee_layers[lyr_name]["ee_layer"])
                except Exception:
                    pass

        sid = sid_data[idx]["SID"]
        lat, lon = sid_data[idx]["lat"], sid_data[idx]["lon"]
        img = get_image(sid)

        # Use evaluateScene instead of evaluate_offshore_site
        region = ee.Geometry.Point([lon, lat]).buffer(200)
        img_flaring_data = evaluateScene(img, region).getInfo()

        vis_params_b11 = {"bands": ["B11"], "min": 0, "max": b11_b12_max[0]}
        vis_params_b12 = {"bands": ["B12"], "min": 0, "max": b12_max_slider.value}
        vis_params_rgb = {
            "bands": ["B4", "B3", "B2"],
            "min": 0,
            "max": 3000,
            "gamma": [1.4, 1.4, 1.4],
        }

        vis_params_mbsp = {
            "bands": ["MBSP"],
            "min": mbsp_min_max[0],
            "max": mbsp_min_max[1],
            "palette": [
                "000000",
                "0000FF",
                "00FFFF",
                "00FF00",
                "FFFF00",
                "FF0000",
                "FFFFFF",
            ],
        }

        # --- flaring & mbsp logic as before ---
        b8a = img.select("B8A")
        b11 = img.select("B11")
        b12 = img.select("B12")
        delta_sw = b12.subtract(b11)
        tai = delta_sw.divide(b8a)
        flare_mask = tai.gte(0.15)
        flare_mask_layer = img.select("B12").updateMask(flare_mask)

        b_vis = img.select("B2").add(img.select("B3")).add(img.select("B4")).divide(3)
        img = img.addBands(b_vis.rename("B_vis"))
        sgi = img.normalizedDifference(["B8A", "B_vis"])
        img = img.addBands(sgi.rename("SGI"))

        img = calculate_sunglint_alpha(img)
        img = img.addBands(
            ee.Image.constant(ee.Number(img.get("glint_alpha"))).rename("SGA")
        )

        mask_c = ee.Image(ee.Number(1))
        mask_mbsp = ee.Image(ee.Number(1))
        mbsp = ee.Image(mbsp_simple_ee(img, mask_c, mask_mbsp))

        if img is not None:
            m.center = (lat, lon, zoom)
            m.addLayer(
                tai, {"min": 0, "max": 1, "palette": ["black", "white"]}, "TAI", False
            )
            m.addLayer(mbsp, vis_params_mbsp, "MBSP")
            m.addLayer(img.select("B12"), vis_params_b12, "B12")
            m.addLayer(img.select("B11"), vis_params_b11, "B11")
            m.addLayer(img.select(["B4", "B3", "B2"]), vis_params_rgb, "RGB", False)
            m.addLayer(
                flare_mask_layer,
                {"palette": ["yellow"], "min": 1500, "max": 3000},
                "Flaring",
                False,
            )
            m.addLayer(
                ee.Geometry.Point(lon, lat).buffer(200),
                {"color": "red"},
                "Point of Interest",
                False,
            )

            if img_flaring_data["properties"]["flare_present"]:
                flare_coords = img_flaring_data["properties"]["flare_latlon"]
                coords = ee.List(flare_coords).getInfo()
                m.addLayer(
                    ee.Geometry.MultiPoint(coords),
                    {"color": "orange"},
                    "Detected Flare",
                    True,
                )
            if extra_gdf is not None:
                ee_gdf = geemap.gdf_to_ee(extra_gdf)
                m.addLayer(ee_gdf, {}, "Extra GDF", False)

        with out:
            out.clear_output()
            print(
                f"Showing {sid} ({idx + 1}/{len(sid_data)} "
                f"c: {round(mbsp.get('C_factor').getInfo(), 3)} "
                f"glint_alpha: {round(img.get('glint_alpha').getInfo(), 3)})"
            )
            props = img_flaring_data["properties"]
            print(
                f"Cloud fraction: {round(props['cloud_fraction'], 3)}"
                f" | Min NDWI: {round(props['min_ndwi'], 3)}"
                f" | Structure present: {props['structure_present']}"
                f" | Max TAI: {round(props['max_TAI'], 1)}"
                f" | Flaring present: {props['flare_present']}"
            )

    # --- navigation buttons ---
    next_btn = widgets.Button(description="Next Granule")
    prev_btn = widgets.Button(description="Previous Granule")
    update_btn = widgets.Button(description="Refresh")

    def on_next(b):
        nonlocal index
        index = (index + 1) % len(sid_data)
        update_map(index)

    def on_prev(b):
        nonlocal index
        index = (index - 1) % len(sid_data)
        update_map(index)

    def on_refresh(b):
        nonlocal index
        update_map(index)

    next_btn.on_click(on_next)
    prev_btn.on_click(on_prev)
    update_btn.on_click(on_refresh)
    # b12_max_slider.observe(lambda change: update_map(index), names="value")

    update_map(index)
    controls = widgets.HBox([prev_btn, next_btn, b12_max_slider, update_btn])
    display(controls, out, m)
    # return mbsp


# mbsp = show_granule_viewer(sid_data)
sid_data = [
    {
        "SID": "20170705T164319_20170705T165225_T15RXL",
        "lon": -90.968,
        "lat": 27.292,
    },
    {
        "SID": "20230716T162841_20230716T164533_T15QWB",
        "lon": -92.2361,
        "lat": 19.56586,
    },
    {
        "SID": "20240421T162841_20240421T164310_T15QWB",
        "lon": -92.23655,
        "lat": 19.56582,
    },
    {
        "SID": "20220707T032531_20220707T033631_T48NTP",
        "lon": 102.986983,
        "lat": 7.592794,
    },
]


# sid_data = [
#     {
#         "SID": "20241228T033049_20241228T034259_T47NRJ",
#         "lon": 102.986983,
#         "lat": 7.592794,
#     }
# ]
# %%
b11 = 3000
sid = "20230607T130251_20230607T130249_T23JQM"
lon = 102.61423806737008
lat = 7.494175116033479
# mbsp = show_granule_viewer(
#     sid_data,
#     b11_b12_max=[b11, b11],
#     zoom=16,
#     mbsp_min_max=[-0.1, 0.1],
# )

# %%
