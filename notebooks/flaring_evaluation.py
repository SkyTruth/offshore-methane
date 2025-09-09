# %%
# Imports and authentication
import ee
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta

# Authenticate and initialize EE (run once per session)
ee.Authenticate()
ee.Initialize()

# %%
# User settings
ROI_RADIUS_M = 200


# %%


def process_point(pt, joined, buffer_m=ROI_RADIUS_M):
    """Process one point over all joined Sentinel-2 + Cloud Probability images."""
    roi = pt.geometry().buffer(buffer_m)
    return joined.filterBounds(roi).map(lambda img: evaluateScene(img, roi))


# %%
# Optional: Load points from Earth Engine asset
def load_points_from_asset(asset_id):
    return ee.FeatureCollection(asset_id)


# Example
points = load_points_from_asset(
    "projects/benshostak-skytruth/assets/custom_flaring_data-4"
)
print("points:", points.size().getInfo())


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


# %%
gfw_df = pd.read_csv(r"C:\Users\ebeva\SkyTruth\methane\brazil_gfw_reviewed.csv")
anchored_vessels_df = gfw_df[
    (gfw_df["note"] == "anchored vessel") & (~gfw_df["lon"].isna())
]
centroidsAll = load_points_from_df(anchored_vessels_df)
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
years = list(range(2015, 2026))
# run_exports(centroidsAll, years)
