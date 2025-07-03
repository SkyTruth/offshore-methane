# %%
import ee
import pandas as pd
from google.cloud import storage
from offshore_methane.masking import (
    saturation_mask,
    cloud_mask,
    outlier_mask,
    ndwi_mask,
    sga_mask,
    sgi_mask,
)
from offshore_methane.config import MASK_PARAMS as DEFAULT_MASK_PARAMS
from tqdm import tqdm

ee.Initialize()


def list_detections_from_gcs(bucket_name):
    """
    Returns a list of (system_index, coordinates) tuples for detections
    that do not yet have corresponding HITL annotations.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    all_unreviewed = []

    # Get all blobs
    all_blobs = list(bucket.list_blobs())

    # Group by system_index folders
    from collections import defaultdict

    detections = defaultdict(list)
    hitl_files = set()

    for blob in all_blobs:
        name = blob.name
        if name.endswith(".geojson"):
            parts = name.split("/")
            if len(parts) != 2:
                continue  # Skip nested or malformed paths

            system_index, filename = parts
            if filename.startswith(f"{system_index}_VEC_"):
                coords = filename[len(f"{system_index}_VEC_") : -len(".geojson")]
                detections[system_index].append(coords)
            elif filename.startswith(f"{system_index}_HITL_"):
                coords = filename[len(f"{system_index}_HITL_") : -len(".geojson")]
                hitl_files.add((system_index, coords))

    for system_index, coords_list in detections.items():
        for coords in coords_list:
            if (system_index, coords) not in hitl_files:
                all_unreviewed.append((system_index, coords))

    return sorted(all_unreviewed)


def sample_image(system_index: tuple, n_points: int = 500):
    # ---- Parse input ----
    # system_index, coords = system_index_coord
    # lon, lat = map(float, coords.split("_"))
    # centre = ee.Geometry.Point(lon, lat)

    # ---- Load Sentinel-2 image ----
    img = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", system_index))
        .first()
    )
    if img is None:
        raise ValueError(f"Image not found for system_index {system_index}")

    # ---- Add SGI ----
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

    # ---- Add SGA ----
    sga = ee.Image.loadGeoTIFF(
        f"gs://offshore_methane/{system_index}/{system_index}_SGA.tif"
    ).rename("SGA")
    img = img.addBands(sga)

    # centre = img.geometry().centroid()

    # ---- Add masks ----
    p = DEFAULT_MASK_PARAMS
    masks = {
        # "mask_c": build_mask_for_C(img, centre),
        # "mask_mbsp": build_mask_for_MBSP(img, centre),
        "saturation_mask": saturation_mask(img, p),
        "cloud_mask": cloud_mask(img, p),
        "outlier_mask": outlier_mask(img, p),
        "ndwi_mask": ndwi_mask(img, p),
        "sga_mask": sga_mask(img, p),
        "sgi_mask": sgi_mask(img, p),
    }

    for name, mask in masks.items():
        img = img.addBands(mask.rename(name))

    # ---- Generate sample points ----
    aoi = img.geometry()
    points_fc = ee.FeatureCollection.randomPoints(aoi, n_points, seed=42)

    # ---- Sample image ----
    img = img.unmask(0)
    samples = img.sampleRegions(collection=points_fc, scale=20, geometries=True)
    sample_info = samples.getInfo()

    # ---- Convert to DataFrame ----
    records = []
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
                # "mask_c": props.get("mask_c"),
                # "mask_mbsp": props.get("mask_mbsp"),
                "saturation_mask": props.get("saturation_mask"),
                "cloud_mask": props.get("cloud_mask"),
                "outlier_mask": props.get("outlier_mask"),
                "ndwi_mask": props.get("ndwi_mask"),
                "sga_mask": props.get("sga_mask"),
                "sgi_mask": props.get("sgi_mask"),
            }
        )

    return pd.DataFrame.from_records(records), img


def sample_multiple_system_indexes(
    system_indexes: list[str], n_points=500
) -> pd.DataFrame:
    all_dfs = []
    for sid in tqdm(system_indexes):
        try:
            print(f"Sampling {sid}...")
            df, _ = sample_image(sid, n_points=n_points)
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed on {sid}: {e}")
    return pd.concat(all_dfs, ignore_index=True)


# %%
list_of_ids = list(list_detections_from_gcs("offshore_methane"))
list_of_sid = list(set([sid for sid, _ in list_of_ids]))
# list_of_sid_zero = [(sid, ("0.0_0.0")) for sid in set([sid for sid, _ in list_of_ids])]
# %%
# sampled_pd = sample_multiple_system_indexes(list_of_sid[0:2])
sampled_pd = sample_multiple_system_indexes(list_of_sid)
sampled_pd.to_csv(r"..\SkyTruth\methane\sgi_sga_mask_sampled.csv", index=False)

# %%
