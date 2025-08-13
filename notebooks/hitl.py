# %%
import json
import os
import geopandas as gpd
import ee
import geemap
import ipywidgets as widgets
from google.cloud import storage
from IPython.display import display
from shapely.geometry import Point
import pandas as pd
from collections import defaultdict

# Initialize the Earth Engine API
ee.Initialize()


# === GCS Save Helper ===
def save_drawn_feature_to_gcs(
    geometry,
    system_index,
    coords,
    bucket_name="offshore_methane",
    hitl_prefix="hitl",
    local_dir=None,
):
    """
    Save drawn feature geometry as a GeoJSON file to Google Cloud Storage and optionally locally.

    Args:
        geometry (dict): GeoJSON geometry of the drawn plume.
        system_index (str): Sentinel-2 system:index identifier.
        coords (str): Coordinate string for the scene.
        bucket_name (str): GCS bucket name.
        hitl_prefix (str): GCS subfolder/prefix for HITL annotations.
        local_dir (str, optional): Local directory to save the file. If None, skips local save.
    """
    filename = f"{system_index}_HITL_{coords}.geojson"

    # Prepare the GeoJSON content
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

    # === Save locally if requested ===
    if local_dir is not None:
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, system_index, filename)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved plume annotation locally to: {local_path}")
    else:
        # === Save to GCS ===
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{system_index}/{filename}")
        blob.upload_from_string(
            json.dumps(geojson), content_type="application/geo+json"
        )
        print(
            f"✔️ Saved plume annotation to: gs://{bucket_name}/{hitl_prefix}/{system_index}/{filename}"
        )


# === Reviewer Interface ===
def start_hitl_review_loop(
    detections,
    bucket_name="offshore_methane",
    save_thumbnail=False,
    from_local=None,
):
    """
    Launch an interactive review interface for HITL plume annotation.

    Args:
        detections (List[Tuple[str, str]]): List of (system_index, coords) tuples.
        bucket_name (str): GCS bucket name.
    """
    save_thumbnail_folder = "../data/thumbnails" if save_thumbnail else None

    scene_selector = widgets.Dropdown(
        options=[
            (
                f"{row['system_index']} @ {row['coords']}",
                (row["system_index"], row["coords"]),
            )
            for _, row in detections.iterrows()
        ],
        description="Scene:",
        layout=widgets.Layout(width="auto"),
    )
    load_btn = widgets.Button(description="Load Scene")
    save_btn = widgets.Button(description="Save Plume")
    no_plume_btn = widgets.Button(description="Save as No Plume")
    status = widgets.Label()
    out = widgets.Output()
    current_map = None

    def display_scene(system_index, coords):
        nonlocal current_map
        if current_map is not None:
            current_map.close()

        current_map = display_s2_with_geojson(
            system_index,
            coords,
            bucket_name=bucket_name,
            save_thumbnail_folder=save_thumbnail_folder,
            local_path=from_local,
        )
        if current_map is None:
            status.value = "❗ No scene loaded."
            return
        current_map.add_draw_control()
        out.clear_output()
        with out:
            display(current_map)

    def on_load_clicked(b):
        # try:
        # global system_index, coords
        system_index, coords = scene_selector.value
        display_scene(system_index, coords)
        status.value = f"✔️ Loaded scene: {system_index}"
        # except Exception as e:
        #     status.value = f"❗ Error loading scene: {e}"

    def on_save_clicked(b):
        system_index, coords = scene_selector.value
        if current_map is None:
            status.value = "❗ No scene loaded."
            return
        if not current_map.user_roi:
            status.value = "❗ Draw a geometry before saving."
            return
        geometry = current_map.user_roi
        geometry_geojson = geometry.getInfo()
        save_drawn_feature_to_gcs(
            geometry_geojson, system_index, coords, bucket_name, local_dir=from_local
        )
        status.value = f"✔️ Saved plume for {system_index}"

    def on_no_plume_clicked(b):
        system_index, coords = scene_selector.value
        empty_geometry = None
        save_drawn_feature_to_gcs(
            empty_geometry, system_index, coords, bucket_name, local_dir=from_local
        )
        status.value = f"✔️ Marked {system_index} as no plume"

    load_btn.on_click(on_load_clicked)
    save_btn.on_click(on_save_clicked)
    no_plume_btn.on_click(on_no_plume_clicked)

    display(widgets.HBox([scene_selector, load_btn]))
    display(widgets.HBox([save_btn, no_plume_btn]))
    display(status)
    display(out)


def list_detections(bucket_name=None, local_path=None):
    """
    Return a GeoDataFrame of detections, marking which have HITL review
    and the size of the HITL file in bytes (None if not available or local_path used).

    Args:
        bucket_name (str): GCS bucket name (optional if local_path is provided).
        local_path (str): Local folder path with same structure as GCS bucket.

    Returns:
        GeoDataFrame: Columns = [system_index, coords, reviewed, geometry, hitl_file_size]
    """
    detections = defaultdict(list)
    hitl_files = set()
    hitl_file_sizes = {}  # Only used for GCS mode
    detections_gdf = gpd.GeoDataFrame(
        columns=["system_index", "coords", "reviewed", "geometry", "hitl_file_size"]
    )

    if local_path:
        # Local folder mode
        for root, _, files in os.walk(local_path):
            for filename in files:
                if filename.endswith(".geojson"):
                    rel_path = os.path.relpath(os.path.join(root, filename), local_path)
                    parts = rel_path.replace("\\", "/").split("/")
                    if len(parts) != 2:
                        continue
                    system_index, filename = parts
                    if filename.startswith(f"{system_index}_VEC_"):
                        coords = filename[
                            len(f"{system_index}_VEC_") : -len(".geojson")
                        ]
                        detections[system_index].append(coords)
                    elif filename.startswith(f"{system_index}_HITL_"):
                        coords = filename[
                            len(f"{system_index}_HITL_") : -len(".geojson")
                        ]
                        hitl_files.add((system_index, coords))
                        hitl_file_sizes[(system_index, coords)] = (
                            None  # Local mode: always None
                        )
    else:
        # GCS mode
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for blob in bucket.list_blobs():
            name = blob.name
            if name.endswith(".geojson"):
                parts = name.split("/")
                if len(parts) != 2:
                    continue
                system_index, filename = parts
                if filename.startswith(f"{system_index}_VEC_"):
                    coords = filename[len(f"{system_index}_VEC_") : -len(".geojson")]
                    detections[system_index].append(coords)
                elif filename.startswith(f"{system_index}_HITL_"):
                    coords = filename[len(f"{system_index}_HITL_") : -len(".geojson")]
                    hitl_files.add((system_index, coords))
                    blob.reload()  # ensure metadata is up-to-date
                    hitl_file_sizes[(system_index, coords)] = blob.size

    # Build GeoDataFrame
    for system_index, coords_list in detections.items():
        for coords in coords_list:
            lon, lat = map(float, coords.split("_"))
            reviewed = (system_index, coords) in hitl_files
            hitl_file_size = hitl_file_sizes.get((system_index, coords), None)
            new_row = gpd.GeoDataFrame(
                [
                    {
                        "system_index": system_index,
                        "coords": coords,
                        "reviewed": reviewed,
                        "geometry": Point(lon, lat),
                        "hitl_file_size": hitl_file_size,
                    }
                ],
                columns=[
                    "system_index",
                    "coords",
                    "reviewed",
                    "geometry",
                    "hitl_file_size",
                ],
            )
            detections_gdf = pd.concat([detections_gdf, new_row], ignore_index=True)

    return detections_gdf


class ThumbnailCreator:
    def __init__(
        self,
        system_index,
        region,
        dimensions=512,
        format="png",
        save_thumbnail_folder="thumbnails",
    ):
        self.system_index = system_index
        self.region = region
        self.dimensions = dimensions
        self.format = format
        self.save_folder = save_thumbnail_folder

    def create_thumbnail(self, image, postfix, vis_params=None):
        """
        Save a PNG thumbnail of an EE Image (already visualised) without relying on
        geemap.get_image_thumbnail(), which is currently broken because it passes an
        invalid `region` kw-arg to ee.Image.visualize().

        The logic is:
        1. Clip the image to the AOI so we don’t need a `region` kw-arg.
        2. Convert it to an RGB image with .visualize(**vis_params).
        3. Ask Earth Engine for a thumbnail URL and stream it to disk.
        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        out_path = os.path.join(self.save_folder, f"{self.system_index}_{postfix}.png")

        # 1) Spatial subset
        clipped = image.clip(self.region)

        # 2) Apply visualisation parameters (can be empty)
        vis_image = clipped.visualize(**(vis_params or {}))

        # 3) Build thumbnail request and download
        params = {
            "dimensions": str(self.dimensions),  # e.g. "512"
            "format": self.format,  # e.g. "png"
        }
        thumb_url = vis_image.getThumbURL(params)

        import urllib.request

        urllib.request.urlretrieve(thumb_url, out_path)


def display_s2_with_geojson(
    system_index,
    coords,
    bucket_name="offshore_methane",
    save_thumbnail_folder=None,
    local_path=None,
):
    """
    Visualize Sentinel-2 imagery and annotation vectors for a given detection.

    Args:
        system_index (str): Sentinel-2 system:index.
        coords (str): Coordinate identifier.
        bucket_name (str): GCS bucket name.
        save_thumbnail_folder (str): Optional folder for saving thumbnails.
        local_path (str): Local folder path with same structure as the GCS bucket.

    Returns:
        geemap.Map: Interactive map with layers and annotations.
    """

    # === Load VEC GeoJSON ===
    filename = f"{system_index}_VEC_{coords}.geojson"

    if local_path:
        vec_path = os.path.join(local_path, system_index, filename)
        if not os.path.exists(vec_path):
            print(f"GeoJSON file not found locally: {vec_path}")
            return
        with open(vec_path, "r") as f:
            geojson = json.load(f)
    else:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_path = f"{system_index}/{filename}"
        blob = bucket.blob(blob_path)
        if not blob.exists():
            print(f"GeoJSON file not found in GCS: {blob_path}")
            return
        geojson = json.loads(blob.download_as_text())

    fc = geemap.geojson_to_ee(geojson)
    aoi = fc.geometry()

    # === Load Sentinel-2 image from Earth Engine ===
    s2_image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", system_index))
        .first()
    )

    if s2_image is None:
        print(f"No Sentinel-2 image found with system:index = {system_index}")
        return

    # === Linear fit function ===
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

        coef = fit.get("coefficients")

        def apply_correction():
            coef_list = ee.Array(coef).toList()
            b0 = ee.Number(ee.List(coef_list.get(0)).get(0))
            corrected_b12 = xVar.multiply(b0).rename("B12_corrected")
            return img.addBands(corrected_b12).set(
                {"c_fit": b0, "coef_list": coef_list}
            )

        def no_correction():
            return img.addBands(xVar.rename("B12_corrected")).set(
                {"c_fit": None, "coef_list": None}
            )

        return ee.Image(ee.Algorithms.If(coef, apply_correction(), no_correction()))

    # === Median B11 value ===
    global B11_median
    B11_median = (
        s2_image.select("B11")
        .reduceRegion(reducer=ee.Reducer.median(), bestEffort=True, scale=20)
        .getInfo()["B11"]
    )
    if B11_median is None:
        B11_median = 1500
    # B11_median

    flare_threshold = 1500
    rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    swir_vis = {"min": 0, "max": 2 * B11_median}
    mbsp_vis = {"min": -0.2, "max": 0.2, "palette": ["red", "white", "blue"]}
    yellow_vis = {
        "palette": ["yellow", "red"],
        "min": flare_threshold,
        "max": flare_threshold * 2,
    }
    fc_vis = {"color": "green", "width": 2}

    corrected_img = linearFit(s2_image)
    lon, lat = map(float, coords.split("_"))
    Map = geemap.Map()
    point = ee.Geometry.Point(lon, lat)

    Map.addLayer(s2_image, rgb_vis, f"RGB {system_index}")
    Map.addLayer(s2_image.select("B11"), swir_vis, "B11")
    Map.addLayer(corrected_img.select("B12_corrected"), swir_vis, "B12 * c_fit")

    # === Load MBSP layer ===
    mbsp_filename = f"{system_index}_MBSP.tif"
    if local_path:
        mbsp_path = os.path.join(local_path, system_index, mbsp_filename)
        if not os.path.exists(mbsp_path):
            print(f"MBSP file not found locally: {mbsp_path}")
            return
        Map.add_raster(
            mbsp_path,
            vmin=-0.2,
            vmax=0.2,
            colormap=["red", "white", "blue"],
            layer_name="MBSP",
            zoom_to_layer=False,
        )
    else:
        mbsp = ee.Image.loadGeoTIFF(
            f"gs://{bucket_name}/{system_index}/{mbsp_filename}"
        )
        Map.addLayer(mbsp, mbsp_vis, "MBSP")

    # === Optional HITL layer ===
    hitl_filename = f"{system_index}_HITL_{coords}.geojson"
    if local_path:
        hitl_path = os.path.join(local_path, system_index, hitl_filename)
        if os.path.exists(hitl_path):
            with open(hitl_path, "r") as f:
                hitl_geojson = json.load(f)
            if hitl_geojson["features"][0]["geometry"] is not None:
                hitl_fc = geemap.geojson_to_ee(hitl_geojson)
                Map.addLayer(hitl_fc, {"color": "red"}, "Existing HITL Plume")
    else:
        hitl_blob_path = f"{system_index}/{hitl_filename}"
        hitl_blob = bucket.blob(hitl_blob_path)
        if hitl_blob.exists():
            hitl_geojson = json.loads(hitl_blob.download_as_text())
            if hitl_geojson["features"][0]["geometry"] is not None:
                hitl_fc = geemap.geojson_to_ee(hitl_geojson)
                Map.addLayer(hitl_fc, {"color": "red"}, "Existing HITL Plume")

    # === Add detection vector and point ===
    Map.addLayer(fc, {}, "Vector")
    Map.addLayer(point, {"color": "red"}, "Target")
    Map.centerObject(point, 14)

    # === Flare mask ===
    b8a = s2_image.select("B8A")
    b11 = s2_image.select("B11")
    b12 = s2_image.select("B12")
    delta_sw = b12.subtract(b11)
    tai = delta_sw.divide(b8a)
    flare_mask = (
        tai.gte(0.45).And(delta_sw.gte(b11.subtract(b8a))).And(b12.gte(flare_threshold))
    )
    flare_mask_layer = s2_image.select("B12").updateMask(flare_mask)
    Map.addLayer(flare_mask_layer, yellow_vis, "Flaring", False)

    # === Optional thumbnail creation ===
    if save_thumbnail_folder is not None:
        thumbnail_creator = ThumbnailCreator(
            system_index,
            region=aoi,
            save_thumbnail_folder=save_thumbnail_folder,
        )
        thumbnail_creator.create_thumbnail(s2_image, "rgb", rgb_vis)
        if not local_path:
            thumbnail_creator.create_thumbnail(mbsp, "mbsp", mbsp_vis)
        thumbnail_creator.create_thumbnail(corrected_img.select("B11"), "b11", swir_vis)
        thumbnail_creator.create_thumbnail(
            corrected_img.select("B12_corrected"), "b12_corrected", swir_vis
        )
        fc_style = fc.style(**fc_vis).visualize()
        if not local_path:
            mbsp_img = mbsp.visualize(**mbsp_vis)
            blended = mbsp_img.blend(fc_style)
            thumbnail_creator.create_thumbnail(blended, "blended", vis_params={})
        if "hitl_fc" in locals():
            fc_style = fc.filterBounds(hitl_fc.geometry()).style(**fc_vis).visualize()
            if not local_path:
                mbsp_img = mbsp.visualize(**mbsp_vis)
                blended = mbsp_img.blend(fc_style)
                thumbnail_creator.create_thumbnail(
                    blended, "blended_hitl", vis_params={}
                )
    # Map.set_center(lon, lat, 14)
    return Map


# %%
detections_gdf = list_detections("offshore_methane")
start_hitl_review_loop(detections_gdf)
# %%
# %%
# # Example pull from Local Folder path
# detections_gdf = list_detections("offshore_methane", local_path="../data")
# start_hitl_review_loop(detections_gdf, from_local="../data")
#
# # Example specify manually
# scenes = pd.DataFrame(
#     [
#         {
#             "system_index": "20170705T164319_20170705T165225_T15RYL",
#             "coords": "-90.968_27.292",
#         },
#     ]
# )
# # Example pull from sites CSV
# sites = pd.read_csv("../data/sites.csv")
# sites["coords"] = (
#     sites["lon"].round(3).astype(str) + "_" + sites["lat"].round(3).astype(str)
# )

# %%
