# %%
# Imports and authentication
import ee
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Authenticate and initialize EE (run once per session)
ee.Authenticate()
ee.Initialize()

# %%
# User settings
ROI_RADIUS_M = 200


# %%
# Utility functions
def evaluateScene(image, region):
    """Evaluate one Sentinel-2 scene for cloudiness, structures, and flaring."""
    point = region.centroid()

    # Select relevant bands
    b3 = image.select("B3")  # Green
    b8 = image.select("B8")  # NIR
    b11 = image.select("B11")  # SWIR1
    b12 = image.select("B12")  # SWIR2
    b8a = image.select("B8A")  # Narrow NIR

    # --- Cloudiness ---
    system_index = image.get("system:index")
    cloud_fraction = (
        image.select("probability")
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e9
        )
        .get("probability")
    )
    cloud_fraction = ee.Number(ee.Algorithms.If(cloud_fraction, cloud_fraction, 100))

    # --- Structure presence (NDWI min) ---
    ndwi = b3.subtract(b8).divide(b3.add(b8))
    ndwi_stats = ndwi.reduceRegion(
        reducer=ee.Reducer.min(), geometry=region, scale=20, maxPixels=1e8
    )
    min_ndwi = ndwi_stats.get("B3")
    min_ndwi = ee.Algorithms.If(min_ndwi, min_ndwi, 9999)
    structure_present = ee.Algorithms.If(ee.Number(min_ndwi).lt(0), 1, 0)

    # --- Flaring detection ---
    delta_sw = b12.subtract(b11)
    tai = delta_sw.divide(b8a).rename("TAI")

    flare_reduce = tai.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
        bestEffort=True,
    )
    max_flare_diff = flare_reduce.get("TAI")
    max_flare_diff = ee.Algorithms.If(max_flare_diff, max_flare_diff, -9999)

    # Get coordinates of max flare pixel
    flare_mask = tai.eq(ee.Number(max_flare_diff))
    flare_coords = (
        flare_mask.reduceToVectors(
            scale=20, geometryType="centroid", maxPixels=1e8, bestEffort=True
        )
        .filterBounds(region)
        .geometry()
        .coordinates()
    )
    flare_coords = ee.List(flare_coords)
    flare_present = ee.Algorithms.If(ee.Number(max_flare_diff).gt(0.15), 1, 0)

    return ee.Feature(point).set(
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


def process_point(pt, joined, buffer_m=ROI_RADIUS_M):
    """Process one point over all joined Sentinel-2 + Cloud Probability images."""
    roi = pt.geometry().buffer(buffer_m)
    return joined.filterBounds(roi).map(lambda img: evaluateScene(img, roi))


# %%
# Optional: Load points from CSV
def load_points_from_df(df, lon_field="lon", lat_field="lat"):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=df.apply(lambda r: Point(r[lon_field], r[lat_field]), axis=1),
        crs="EPSG:4326",
    )
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Point":
            props = row.drop("geometry").to_dict()
            features.append(ee.Feature(ee.Geometry.Point([geom.x, geom.y]), props))
    return ee.FeatureCollection(features)


# %%
# Main export loop
def run_exports(points_fc, years, file_prefix="flaring"):
    for year in years:
        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"

        s2Sr = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(points_fc)
        )
        s2Cloud = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterDate(start_date, end_date)
            .filterBounds(points_fc)
        )
        joined = s2Sr.linkCollection(s2Cloud, ["probability"])

        results_list = points_fc.map(lambda pt: process_point(pt, joined)).flatten()
        results_fc = ee.FeatureCollection(results_list)

        task = ee.batch.Export.table.toDrive(
            collection=results_fc,
            description=f"{file_prefix}_{year}",
            fileFormat="CSV",
        )
        task.start()
        print(f"Exporting year {year} to Google Drive...")


def create_structure_flaring_timeline(
    df, date_col="granule_date", filter_outlier=False, flare_override_absent=True
):
    """
    Creates a timeline indicating structure and flaring status based on the input DataFrame.
    Parameters
    ----------
    df : DataFrame
        DataFrame containing structure and flaring evaluation results.
    date_col : str
        Name of the column containing the date of each granule (default: "granule_date").
    filter_outlier : bool
        Whether to filter out outlier points in the timeline (default: False).
    flare_override_absent : bool
        Whether to override flare status when structure is absent (default: True).
    Returns
    -------
    DataFrame
        A DataFrame with additional columns for color coding and outlier detection for the timeline visualization.
    """
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
    """
    plots a timeline of structure and flaring status based on the input DataFrame.
    Parameters
    ----------
    df : DataFrame
        DataFrame containing structure and flaring evaluation results.
    date_col : str
        Name of the column containing the date of each granule (default: "granule_date").
    magnitude_col : str, optional
        Name of the column to display as the heights of the date bars (default: None).
    max_mag : float
        Maximum magnitude for scaling the date bars (default: 1).
    """
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
# Optional: Manual points
# sid_data = [
#     {"lat": -25.39439, "lon": -42.76179},
#     {"lat": -25.448847759194713, "lon": -42.75307163955262},
#     {"lat": -25.602820518570194, "lon": -42.82065823421971},
# ]
# features = [ee.Feature(ee.Geometry.Point(d["lon"], d["lat"])) for d in sid_data]
# centroidsAll = ee.FeatureCollection(features)
# Example call
# %%
# Expample flaring evaluation
# years = list(range(2015, 2026))
# run_exports(centroidsAll, years, file_prefix="gfw_fxo_flaring")
