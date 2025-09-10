# %%
import os
import sys

import ee
import geemap
import ipywidgets as widgets
from flaring_evaluation import evaluateScene
from IPython.display import display
from tqdm import tqdm

mbsp_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "offshore_methane"
)
if mbsp_path not in sys.path:
    sys.path.append(mbsp_path)

from mbsp import mbsp_simple_ee  # noqa

ee.Authenticate()
ee.Initialize()


def add_glint_alpha(sid_data, collection="COPERNICUS/S2_HARMONIZED"):
    """
    Adds glint_alpha metadata to each SID in sid_data.

    Parameters
    ----------
    sid_data : list of dict
        Each dict must have a "SID" key.
    collection : str
        Earth Engine collection (default: Sentinel-2 SR harmonized).

    Returns
    -------
    list of dict
        Original sid_data with an extra "glint_alpha" field for each entry.
    """
    out = []
    for item in tqdm(sid_data):
        sid = item["SID"]

        # Load the image
        img = (
            ee.ImageCollection(collection)
            .filter(ee.Filter.eq("system:index", sid))
            .first()
        )
        if img is None:
            item["glint_alpha"] = None
            out.append(item)
            continue

        # Compute glint_alpha
        img = calculate_sunglint_alpha(img)
        try:
            alpha = img.get("glint_alpha").getInfo()

            # Store result
            item = item.copy()
            item["glint_alpha"] = alpha
        except Exception as e:
            print(f"Error computing glint_alpha for {sid}: {e}")
            item = item.copy()
            item["glint_alpha"] = None
        out.append(item)

    # Sort by glint_alpha (ignoring Nones)
    out_sorted = sorted(
        out,
        key=lambda d: float("inf") if d["glint_alpha"] is None else d["glint_alpha"],
    )
    return out_sorted


def calculate_sunglint_alpha(image: ee.Image):
    """
    Calculates the glint alpha metadata for estimating sun glint in a scene
    """
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


def get_image(sid):
    col = ee.ImageCollection("COPERNICUS/S2")
    return col.filter(ee.Filter.eq("system:index", sid)).first()


def add_cloud_probability(s2_img):
    # Get the granule ID from the system:index
    granule_id = s2_img.get("system:index")

    # Match with the cloud probability collection
    s2_cloud = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.eq("system:index", granule_id))
        .first()
    )

    # Add the cloud probability band as "probability"
    return s2_img.addBands(s2_cloud.rename("probability"))


class ViewerState:
    """
    Stores information about the current view
    """

    def __init__(self):
        self.sid = None  # will store currently displayed SID


def show_granule_viewer(
    sid_data,
    b11_b12_max=[3000, 2800],
    zoom=12,
    mbsp_min_max=[-0.2, 0.2],
    extra_gdf=None,
    starting_idx=0,
    layers=["B11", "B12", "RGB", "MBSP", "Flaring", "Detected Flare"],
):
    """
    Displays an interactive viewer for Sentinel-2 granules with flaring evaluation.

    Parameters
    ----------
    sid_data : list of dict
        Each dict must have a "SID" key and lat/lon.
    b11_b12_max : list of float
        Maximum values for B11 and B12 visualization (default: [3000, 2800]).
    zoom : int
        Initial zoom level for the map (default: 12).
    mbsp_min_max : list of float
        Minimum and maximum values for MBSP visualization (default: [-0.2, 0.2]).
    extra_gdf : GeoDataFrame, optional
        Additional GeoDataFrame to overlay on the map (default: None).
    starting_idx : int
        Index of the initial SID to display (default: 0).
    layers : list of str
        List of layers to display on the map (default: ["B11", "B12", "RGB", "MBSP", "Flaring", "Detected Flare"]).
    """
    index = starting_idx
    state = ViewerState()

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
        # --- remove old layers ---
        for lyr_name in ["B11", "B12", "RGB", "MBSP", "Flaring"]:
            if lyr_name in m.ee_layers:
                try:
                    m.remove_layer(m.ee_layers[lyr_name]["ee_layer"])
                except Exception:
                    pass

        state.sid = sid_data[idx]["SID"]
        lat, lon = sid_data[idx]["lat"], sid_data[idx]["lon"]
        img = get_image(state.sid)

        region = ee.Geometry.Point([lon, lat]).buffer(200)
        img = add_cloud_probability(img)
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

        # --- flaring & mbsp logic ---
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

            # --- add only requested layers ---
            if "TAI" in layers:
                m.addLayer(
                    tai,
                    {"min": 0, "max": 1, "palette": ["black", "white"]},
                    "TAI",
                    False,
                )
            if "B12" in layers:
                m.addLayer(img.select("B12"), vis_params_b12, "B12")
            if "B11" in layers:
                m.addLayer(img.select("B11"), vis_params_b11, "B11")
            if "MBSP" in layers:
                m.addLayer(mbsp, vis_params_mbsp, "MBSP", False)
            if "RGB" in layers:
                m.addLayer(img.select(["B4", "B3", "B2"]), vis_params_rgb, "RGB", False)
            if "Flaring" in layers:
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

            if (
                img_flaring_data["properties"]["flare_present"]
                and "Detected Flare" in layers
            ):
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
                f"Showing {state.sid} ({idx + 1}/{len(sid_data)} "
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

    update_map(index)
    controls = widgets.HBox([prev_btn, next_btn, b12_max_slider, update_btn])
    display(controls, out, m)

    return state


def gdf_to_sid_list(gdf):
    """
    Converts a GeoDataFrame with 'system_index' and 'geometry' columns to a list of dictionaries with 'SID', 'lat', and 'lon' keys.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing 'system_index' and 'geometry' columns.

    Outputs
    -------
    sid_data : list of dict
        A list of dictionaries, each containing 'SID', 'lat', and 'lon' keys corresponding to the system index and coordinates of each geometry in the GeoDataFrame.
    """
    sid_data = []
    for i, row in gdf.iterrows():
        sid_data.append(
            {
                "SID": row["system_index"],
                "lat": row["geometry"].y,
                "lon": row["geometry"].x,
            }
        )
    return sid_data


# %%
# sid_data = [
#     {
#         "SID": "20170705T164319_20170705T165225_T15RXL",
#         "lon": -90.968,
#         "lat": 27.292,
#     },
#     {
#         "SID": "20230716T162841_20230716T164533_T15QWB",
#         "lon": -92.2361,
#         "lat": 19.56586,
#     },
#     {
#         "SID": "20240421T162841_20240421T164310_T15QWB",
#         "lon": -92.23655,
#         "lat": 19.56582,
#     },
#     {
#         "SID": "20220707T032531_20220707T033631_T48NTP",
#         "lon": 102.986983,
#         "lat": 7.592794,
#     },
# ]

# state = show_granule_viewer(
#     sid_data,
# )

# %%
