# %%
import ee
import pandas as pd
import random
from shapely.geometry import Point
import geopandas as gpd
from google.cloud import storage
import matplotlib.pyplot as plt
from offshore_methane.masking import build_mask_for_C, build_mask_for_MBSP

ee.Initialize()


def list_system_indexes(bucket_name):
    """
    Returns a list of system_index
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Get all blobs
    all_blobs = list(bucket.list_blobs())

    system_indexes = []
    for blob in all_blobs:
        name = blob.name
        if name.endswith(".geojson"):
            parts = name.split("/")
            if len(parts) != 2:
                continue  # Skip nested or malformed paths

            system_index, filename = parts
            system_indexes.append(system_index)
    return set(system_indexes)


def sample_image(system_index: str, n_points: int = 500):
    # Load S2 image
    img = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", system_index))
        .first()
    )
    img = img.unmask(0)

    # Compute SGI = (B12 - mean(B2,B3,B4)) / (B12 + mean(B2,B3,B4))
    b_vis = (
        img.select("B2")
        .add(img.select("B3"))
        .add(img.select("B4"))
        .divide(3)
        .rename("B_vis")
    )
    img = img.addBands(b_vis)
    sgi = img.normalizedDifference(["B12", "B_vis"]).rename("SGI")
    img = img.addBands(sgi)
    # Load SGA from GCS and rename band
    sga = ee.Image.loadGeoTIFF(
        f"gs://offshore_methane/{system_index}/{system_index}_SGA.tif"
    )
    img = img.addBands(sga.rename("SGA"))

    # Define AOI from image footprint (or buffer/contract if needed)

    # Generate N random points in AOI
    # Build masks
    aoi = img.geometry()
    points_fc = ee.FeatureCollection.randomPoints(aoi, n_points, seed=42)

    mask_c = build_mask_for_C(img, aoi.centroid(1))  # assumes point somewhere central
    mask_m = build_mask_for_MBSP(img, aoi.centroid(1))

    # # Add masks to image
    img = img.addBands(mask_c.rename("mask_c"))
    img = img.addBands(mask_m.rename("mask_mbsp"))

    # Sample image at those points
    img = img.unmask(0)
    samples = img.sampleRegions(collection=points_fc, scale=20, geometries=True)
    sample_info = samples.getInfo()
    # return aoi
    # return img
    # Convert to DataFrame
    records = []
    print(sample_info)
    for feat in sample_info["features"]:
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]
        records.append(
            {
                "system_index": system_index,
                "lon": coords[0],
                "lat": coords[1],
                "SGA": props.get("SGA"),
                "SGI": props.get("SGI"),
                "mask_c": props.get("mask_c"),
                "mask_mbsp": props.get("mask_mbsp"),
            }
        )

    return pd.DataFrame.from_records(records), img


def sample_multiple_system_indexes(
    system_indexes: list[str], n_points=500
) -> pd.DataFrame:
    all_dfs = []
    for sid in system_indexes:
        try:
            print(f"Sampling {sid}...")
            df, _ = sample_image(sid, n_points=n_points)
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed on {sid}: {e}")
    return pd.concat(all_dfs, ignore_index=True)


# %%
list_of_ids = list(list_system_indexes("offshore_methane"))
# %%
sampled_pd = sample_multiple_system_indexes(list_of_ids[0:1])

# %%
