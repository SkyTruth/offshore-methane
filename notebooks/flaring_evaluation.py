# %%
# Imports and authentication
import ee
import geopandas as gpd
from shapely.geometry import Point

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


# %%
# Optional: Load from csv
# gfw_df = pd.read_csv(r"")
# anchored_vessels_df = gfw_df[
#     (gfw_df["note"] == "anchored vessel") & (~gfw_df["lon"].isna())
# ]
# centroidsAll = load_points_from_df(anchored_vessels_df)
# %%
# Optional: Manual points
sid_data = [
    {"lat": -25.39439, "lon": -42.76179},
    {"lat": -25.448847759194713, "lon": -42.75307163955262},
    {"lat": -25.602820518570194, "lon": -42.82065823421971},
]
features = [ee.Feature(ee.Geometry.Point(d["lon"], d["lat"])) for d in sid_data]
centroidsAll = ee.FeatureCollection(features)
# Example call
years = list(range(2015, 2026))
run_exports(centroidsAll, years)
