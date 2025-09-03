# %%
import ee
import matplotlib.pyplot as plt
import pandas as pd
import geemap
import ipywidgets as widgets
from IPython.display import display
import sys
import os
from tqdm import tqdm
from datetime import timedelta

mbsp_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "offshore_methane"
)
if mbsp_path not in sys.path:
    sys.path.append(mbsp_path)

from mbsp import mbsp_simple_ee
from masking import build_mask_for_C, build_mask_for_MBSP

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
        except:
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


class ViewerState:
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


def create_structure_flaring_timeline(
    df, date_col="granule_date", filter_outlier=False, flare_override_absent=True
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    # Map to colors
    def status_color_override(row):
        if row["flaring"] == 1:
            return "green"
        elif row["structure_present"] == 0:
            return "red"
        else:
            return "yellow"

    def status_color(row):
        if row["structure_present"] == 0:
            return "red"
        elif row["flaring"] == 1:
            return "green"
        else:
            return "yellow"

    if flare_override_absent:
        df["color"] = df.apply(status_color_override, axis=1)
    else:
        df["color"] = df.apply(status_color, axis=1)

    df["is_outlier"] = (df["color"] != df["color"].shift(1)) & (
        df["color"] != df["color"].shift(-1)
    )
    if filter_outlier:
        df = df[~df["is_outlier"]]
    return df


def plot_structure_flaring_timeline(
    df, date_col="granule_date", magnitude_col=None, max_mag=1
):
    plt.figure(figsize=(12, 3))

    if magnitude_col:
        # Normalize lengths so they are visually balanced
        lengths = df[magnitude_col].apply(
            lambda h: max_mag if (h > max_mag or h < -max_mag) else h
        )
        # lengths = (vals / max_mag) * 0.8  # scale to 80% of axis height
    else:
        lengths = [0.8] * len(df)

    for x, h, c in zip(df[date_col], lengths, df["color"]):
        plt.vlines(x, 0, h, color=c, linewidth=2)

    # Beautify
    plt.title("Structure & Flaring Timeline", fontsize=14)
    # plt.yticks([])
    plt.xlabel("Date")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def extract_state_blocks(df, date_col="granule_date", drop_zero_day=True):
    """
    Collapse consecutive rows of the same 'color' into blocks.

    Returns a dataframe with:
      - start
      - end
      - color
      - duration
    """
    color_to_state = {
        "green": "flaring on",
        "yellow": "flaring off",
        "red": "structure absent",
    }
    df = df.copy().sort_values(date_col)
    df = df.reset_index(drop=True)

    # extras
    # system_index = df['system_index'].iloc[0]
    lat = df["geometry"].iloc[0].x
    lon = df["geometry"].iloc[0].y
    structure_id = df["structure_id"].iloc[0]

    blocks = []
    start = df.loc[0, date_col]
    current_color = df.loc[0, "color"]
    count = 1

    for i in range(1, len(df)):
        if df.loc[i, "color"] != current_color:
            end = df.loc[i - 1, date_col]
            blocks.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "start": start,
                    "end": end,
                    "state": color_to_state[current_color],
                    "color": current_color,
                    "duration": end - start,
                    "granule_count": count,
                    # "system_index": system_index,
                    "structure_id": structure_id,
                }
            )
            # start new block
            start = df.loc[i, date_col]
            current_color = df.loc[i, "color"]
            count = 1
        else:
            count += 1

    # add final block
    end = df.loc[len(df) - 1, date_col]
    blocks.append(
        {
            "lat": lat,
            "lon": lon,
            "start": start,
            "end": end,
            "state": color_to_state[current_color],
            "color": current_color,
            "duration": end - start,
            "granule_count": count,
            # "system_index": system_index,
            "structure_id": structure_id,
        }
    )
    block_pd = pd.DataFrame(blocks)
    if drop_zero_day:
        block_pd = block_pd[block_pd["duration"] > timedelta(0)]

    return block_pd


def plot_state_blocks(blocks_df):
    """
    Plot timeline blocks with colors.
    """
    fig, ax = plt.subplots(figsize=(12, 2))

    for _, row in blocks_df.iterrows():
        ax.barh(
            y=0,
            width=(row["end"] - row["start"]).days
            + (row["end"] - row["start"]).seconds / 86400,
            left=row["start"],
            color=row["color"],
            edgecolor="black",
            height=0.5,
        )

    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title("State Blocks Timeline")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def assign_group_ids(gdf, buffer_size=0.1, geom_col="geometry", id_col="structure_id"):
    gdf = gdf.copy()
    gdf["buffer"] = gdf[geom_col].buffer(buffer_size)
    gdf[id_col] = -1

    group_id = 0
    for idx, geom in gdf["buffer"].items():
        if gdf.at[idx, id_col] == -1:  # not yet assigned
            group_id += 1
            intersects = gdf["buffer"].intersects(geom)
            gdf.loc[intersects, id_col] = group_id

    return gdf.drop(columns="buffer")


# %%
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

b11 = 3000
# state = show_granule_viewer(
#     sid_data,
#     b11_b12_max=[b11, b11],
#     zoom=16,
#     mbsp_min_max=[-0.1, 0.1],
# )

# %%
