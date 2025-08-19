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


mbsp_path = r"C:\Users\ebeva\SkyTruth\git\offshore-methane\offshore_methane"
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

    def get_image(sid):
        col = ee.ImageCollection("COPERNICUS/S2")
        return col.filter(ee.Filter.eq("system:index", sid)).first()

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
        flare_threshold = 1500
        flare_mask = (
            tai.gte(0.45)
            .And(delta_sw.gte(b11.subtract(b8a)))
            .And(b12.gte(flare_threshold))
        )
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

        if extra_gdf is not None:
            ee_gdf = geemap.gdf_to_ee(extra_gdf)
            m.addLayer(ee_gdf, {}, "Extra GDF", False)

        if img is not None:
            m.center = (lat, lon, zoom)
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
                ee.Geometry.Point(lon, lat), {"color": "red"}, "Point of Interest"
            )

        with out:
            out.clear_output()
            print(
                f"Showing {sid} ({idx + 1}/{len(sid_data)} "
                f"c: {round(mbsp.get('C_factor').getInfo(), 3)} "
                f"glint_alpha: {round(img.get('glint_alpha').getInfo(), 3)})"
            )
        return mbsp

    # --- navigation buttons ---
    next_btn = widgets.Button(description="Next Granule")
    prev_btn = widgets.Button(description="Previous Granule")

    def on_next(b):
        nonlocal index
        index = (index + 1) % len(sid_data)
        update_map(index)

    def on_prev(b):
        nonlocal index
        index = (index - 1) % len(sid_data)
        update_map(index)

    next_btn.on_click(on_next)
    prev_btn.on_click(on_prev)
    b12_max_slider.observe(lambda change: update_map(index), names="value")

    mbsp = update_map(index)
    controls = widgets.HBox([prev_btn, next_btn, b12_max_slider])
    display(controls, out, m)
    return mbsp


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
# %%
b11 = 3000

mbsp = show_granule_viewer(
    sid_data,
    b11_b12_max=[b11, b11],
    zoom=16,
    mbsp_min_max=[-0.1, 0.1],
)

# %%
