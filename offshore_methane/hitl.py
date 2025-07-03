# %%
import os
import json
import ee
import geemap
from google.cloud import storage
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

ee.Initialize()


# === GCS Save Helper ===
def save_drawn_feature_to_gcs(
    geometry,
    system_index,
    bucket_name="offshore_methane",
    hitl_prefix="hitl",
):
    """Save drawn geometry as GeoJSON to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    filename = f"{system_index}_HITL.geojson"
    blob = bucket.blob(f"{system_index}/{filename}")

    # Format as GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {"system_index": system_index},
            }
        ],
    }

    # Upload to GCS
    blob.upload_from_string(json.dumps(geojson), content_type="application/geo+json")
    print(f"✔️ Saved plume annotation to: gs://{bucket_name}/{hitl_prefix}/{filename}")


# === Reviewer Interface ===
def start_hitl_review_loop(detections, bucket_name="offshore_methane"):
    scene_selector = widgets.Dropdown(
        options=[(f"{sid} @ {coords}", (sid, coords)) for sid, coords in detections],
        description="Scene:",
        layout=widgets.Layout(width="auto"),
    )
    load_btn = widgets.Button(description="Load Scene")  # <- NEW
    save_btn = widgets.Button(description="Save Plume")
    no_plume_btn = widgets.Button(description="Save as No Plume")
    status = widgets.Label()
    out = widgets.Output()
    current_map = None  # for later access

    def display_scene(scene):
        nonlocal current_map
        system_index, coords = scene
        current_map = display_s2_with_geojson_from_gcs(
            system_index, coords, bucket_name=bucket_name
        )
        current_map.add_draw_control()
        out.clear_output()
        with out:
            display(current_map)

    def on_load_clicked(b):  # <- NEW
        try:
            display_scene(scene_selector.value)
            status.value = f"✔️ Loaded scene: {scene_selector.value[0]}"
        except Exception as e:
            status.value = f"❗ Error loading scene: {e}"

    def on_save_clicked(b):
        system_index, coords = scene_selector.value
        if not current_map.user_roi:
            status.value = "❗ Draw a geometry before saving."
            return
        geometry = current_map.user_roi
        geometry_geojson = geometry.getInfo()
        save_drawn_feature_to_gcs(geometry_geojson, system_index, bucket_name)
        status.value = f"✔️ Saved plume for {system_index}"

    def on_no_plume_clicked(b):
        system_index, coords = scene_selector.value
        empty_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {"system_index": system_index},
                }
            ],
        }
        save_drawn_feature_to_gcs(empty_geojson, system_index, bucket_name)
        status.value = f"✔️ Marked {system_index} as no plume"

    # Event bindings
    load_btn.on_click(on_load_clicked)
    save_btn.on_click(on_save_clicked)
    no_plume_btn.on_click(on_no_plume_clicked)

    # Display initial UI
    display(widgets.HBox([scene_selector, load_btn]))
    display(widgets.HBox([save_btn, no_plume_btn]))
    display(status)
    display(out)


# === GCS Data Helper ===
def deduplicate_by_date_and_coords(detections):
    seen = set()
    unique = []

    for system_index, coords in detections:
        date = system_index.split("_")[0]  # Extract date part like '20210507T162901'
        key = (date, coords)
        if key not in seen:
            unique.append((system_index, coords))
            seen.add(key)

    return unique


def list_unreviewed_detections_from_gcs(bucket_name):
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


def display_s2_with_geojson_from_gcs(
    system_index, coords, bucket_name="offshore_methane", vectors_prefix="vectors"
):
    """
    Load a GeoJSON file from GCS and display:
    - RGB Sentinel-2 image
    - B11 (for SGI correction)
    - C-corrected B12
    - Overlay vector annotation
    """
    from google.cloud import storage

    # === Load GeoJSON from GCS ===
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    filename = f"{system_index}_VEC_{coords}.geojson"
    blob_path = f"{system_index}/{filename}"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        print(f"GeoJSON file not found in GCS: {blob_path}")
        return

    geojson_str = blob.download_as_text()
    geojson = json.loads(geojson_str)
    fc = geemap.geojson_to_ee(geojson)
    aoi = fc.geometry()

    # === Get Sentinel-2 image by exact system:index ===
    s2_collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    s2_image = s2_collection.filter(ee.Filter.eq("system:index", system_index)).first()

    if s2_image is None:
        print(f"No Sentinel-2 image found with system:index = {system_index}")
        return

    # === Perform Linear Fit ===
    def linearFit(img):
        xVar = img.select("B12")
        yVar = img.select("B11")
        imgRegress = ee.Image.cat(xVar, yVar)

        fit = imgRegress.reduceRegion(
            reducer=ee.Reducer.linearRegression(numX=1, numY=1),
            geometry=aoi,
            scale=20,
            bestEffort=True,
        )

        coefList = ee.Array(fit.get("coefficients")).toList()
        b0 = ee.Number(ee.List(coefList.get(0)).get(0))

        # Apply correction
        corrected_b12 = img.select("B12").multiply(b0).rename("B12_corrected")

        # Attach corrected band to original image
        return img.addBands(corrected_b12).set({"c_fit": b0, "coef_list": coefList})

    corrected_img = linearFit(s2_image)

    # === Display Map ===
    Map = geemap.Map()
    Map.centerObject(fc, 15)

    # RGB composite
    Map.addLayer(
        s2_image,
        {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
        f"RGB {system_index}",
    )

    # B11
    Map.addLayer(corrected_img.select("B11"), {"min": 0, "max": 3000}, "B11")

    # C-corrected B12
    Map.addLayer(
        corrected_img.select("B12_corrected"), {"min": 0, "max": 3000}, "B12 * c_fit"
    )

    mbsp = ee.Image.loadGeoTIFF(
        f"gs://offshore_methane/{system_index}/{system_index}_MBSP.tif"
    )

    Map.addLayer(
        mbsp,
        {"min": -0.1, "max": 0.1, "palette": ["red", "white", "blue"]},
        "MBSP",
    )

    hitl_blob_path = f"{system_index}/{system_index}_HITL.geojson"
    hitl_blob = bucket.blob(hitl_blob_path)
    if hitl_blob.exists():
        hitl_str = hitl_blob.download_as_text()
        hitl_geojson = json.loads(hitl_str)
        print(hitl_geojson)
        if hitl_geojson["features"][0]["geometry"]:
            hitl_fc = geemap.geojson_to_ee(hitl_geojson)
            Map.addLayer(hitl_fc, {"color": "red"}, "Existing HITL Plume")

    # Vector overlay
    Map.addLayer(fc, {}, f"MBSPs_{system_index}")

    return Map


# %%
list_of_detections = list_unreviewed_detections_from_gcs("offshore_methane")
list_deduplicate = deduplicate_by_date_and_coords(list_of_detections)
start_hitl_review_loop(list_deduplicate)

# %%

# %%
