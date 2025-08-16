# %%
import csv
import json
import os

import ee
import geemap
import ipywidgets as widgets
from google.cloud import storage
from IPython.display import display

import offshore_methane.config as cfg


# === GCS Save Helper ===
def save_drawn_feature_to_gcs(
    geometry,
    system_index,
    bucket_name="offshore_methane",
    hitl_prefix="hitl",
):
    """
    Save drawn feature geometry as a GeoJSON file to Google Cloud Storage.

    Args:
        geometry (dict): GeoJSON geometry of the drawn plume.
        system_index (str): Sentinel-2 system:index identifier.
        bucket_name (str): GCS bucket name.
        hitl_prefix (str): GCS subfolder/prefix for HITL annotations.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    filename = f"{system_index}_HITL.geojson"
    blob = bucket.blob(f"{system_index}/{filename}")

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

    blob.upload_from_string(json.dumps(geojson), content_type="application/geo+json")
    print(f"✔️ Saved plume annotation to: gs://{bucket_name}/{hitl_prefix}/{filename}")


# === Reviewer Interface ===
def start_hitl_review_loop(
    detections, bucket_name="offshore_methane", save_thumbnail=False
):
    """
    Launch an interactive review interface for HITL plume annotation.

    Args:
        detections (List[Tuple[str, str]]): List of (system_index, coords) tuples.
        bucket_name (str): GCS bucket name.
    """
    save_thumbnail_folder = "../data/thumbnails" if save_thumbnail else None

    scene_selector = widgets.Dropdown(
        options=[(f"{sid} @ {coords}", (sid, coords)) for sid, coords in detections],
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

        current_map = display_s2_with_geojson_from_gcs(
            system_index,
            coords,
            bucket_name=bucket_name,
            save_thumbnail_folder=save_thumbnail_folder,
        )
        if current_map is None:
            status.value = "❗ No scene loaded."
            return
        current_map.add_draw_control()
        out.clear_output()
        with out:
            display(current_map)

    def on_load_clicked(b):
        try:
            system_index, coords = scene_selector.value
            display_scene(system_index, coords)
            status.value = f"✔️ Loaded scene: {system_index}"
        except Exception as e:
            status.value = f"❗ Error loading scene: {e}"

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

    load_btn.on_click(on_load_clicked)
    save_btn.on_click(on_save_clicked)
    no_plume_btn.on_click(on_no_plume_clicked)

    display(widgets.HBox([scene_selector, load_btn]))
    display(widgets.HBox([save_btn, no_plume_btn]))
    display(status)
    display(out)


# === GCS Data Helper ===
def deduplicate_by_date_and_coords(detections):
    """
    Deduplicate detections by date and coordinate combination.

    Args:
        detections (List[Tuple[str, str]]): List of (system_index, coords) tuples.

    Returns:
        List[Tuple[str, str]]: Deduplicated list.
    """
    seen = set()
    unique = []

    for system_index, coords in detections:
        date = system_index.split("_")[0]
        key = (date, coords)
        if key not in seen:
            unique.append((system_index, coords))
            seen.add(key)

    return unique


def list_unreviewed_detections_from_gcs(bucket_name):
    """
    Return a list of (system_index, coordinates) tuples for detections
    missing a HITL annotation in the GCS bucket.

    Args:
        bucket_name (str): GCS bucket name.

    Returns:
        List[Tuple[str, str]]: Unreviewed detections.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    all_unreviewed = []

    from collections import defaultdict

    detections = defaultdict(list)
    hitl_files = set()

    for blob in list(bucket.list_blobs()):
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

    for system_index, coords_list in detections.items():
        for coords in coords_list:
            if (system_index, coords) not in hitl_files:
                all_unreviewed.append((system_index, coords))

    return sorted(all_unreviewed)


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


def display_s2_with_geojson_from_gcs(
    system_index,
    coords,
    bucket_name="offshore_methane",
    save_thumbnail_folder=None,
):
    """
    Visualize Sentinel-2 imagery and annotation vectors for a given detection.

    Args:
        system_index (str): Sentinel-2 system:index.
        coords (str): Coordinate identifier.
        bucket_name (str): GCS bucket name.

    Returns:
        geemap.Map: Interactive map with layers and annotations.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    filename = f"{system_index}_VEC_{coords}.geojson"
    blob_path = f"{system_index}/{filename}"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        print(f"GeoJSON file not found in GCS: {blob_path}")
        return

    geojson = json.loads(blob.download_as_text())
    fc = geemap.geojson_to_ee(geojson)
    aoi = fc.geometry()

    s2_image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", system_index))
        .first()
    )

    if s2_image is None:
        print(f"No Sentinel-2 image found with system:index = {system_index}")
        return

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
        corrected_b12 = img.select("B12").multiply(b0).rename("B12_corrected")
        return img.addBands(corrected_b12).set({"c_fit": b0, "coef_list": coefList})

    B11_max = (
        s2_image.select("B11")
        .reduceRegion(reducer=ee.Reducer.max(), geometry=aoi, scale=20)
        .getInfo()["B11"]
    )
    flare_threshold = 1500

    rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    swir_vis = {"min": 0, "max": B11_max}
    mbsp_vis = {"min": -0.2, "max": 0.2, "palette": ["red", "white", "blue"]}
    yellow_vis = {
        "palette": ["yellow", "red"],
        "min": flare_threshold,
        "max": flare_threshold * 2,
    }
    fc_vis = {"color": "green", "width": 2}

    corrected_img = linearFit(s2_image)
    lon, lat = map(float, coords.split("_"))
    Map = geemap.Map(center=[lat, lon], zoom=15)
    point = ee.Geometry.Point(lon, lat)
    Map.addLayer(
        s2_image,
        rgb_vis,
        f"RGB {system_index}",
    )
    Map.addLayer(corrected_img.select("B11"), swir_vis, "B11")
    Map.addLayer(corrected_img.select("B12_corrected"), swir_vis, "B12 * c_fit")

    mbsp = ee.Image.loadGeoTIFF(
        f"gs://offshore_methane/{system_index}/{system_index}_MBSP.tif"
    )
    Map.addLayer(mbsp, mbsp_vis, "MBSP")
    hitl_blob_path = f"{system_index}/{system_index}_HITL.geojson"
    hitl_blob = bucket.blob(hitl_blob_path)
    if hitl_blob.exists():
        hitl_str = hitl_blob.download_as_text()
        hitl_geojson = json.loads(hitl_str)
        if hitl_geojson["features"][0]["geometry"]:
            hitl_fc = geemap.geojson_to_ee(hitl_geojson)
            Map.addLayer(hitl_fc, {"color": "red"}, "Existing HITL Plume")

    Map.addLayer(fc, {}, "Vector")
    Map.addLayer(point, {"color": "red"}, "Target")

    b8a = s2_image.select("B8A")  # 0.86 µm (narrow NIR)
    b11 = s2_image.select("B11")  # 1.61 µm (SWIR-1)
    b12 = s2_image.select("B12")  # 2.19 µm (SWIR-2)

    delta_sw = b12.subtract(b11)  # B12 - B11
    tai = delta_sw.divide(b8a)  # (B12 - B11) / B8A

    flare_mask = (
        tai.gte(0.45)  # ① TAI ≥ 0.45
        .And(delta_sw.gte(b11.subtract(b8a)))  # ② ΔSWIR ≥ ΔNIR-SWIR
        .And(b12.gte(flare_threshold))  # ③ keep very hot pixels (original gate)
    )

    flare_mask_layer = s2_image.select("B12").updateMask(flare_mask)
    Map.addLayer(flare_mask_layer, yellow_vis, "Flaring", False)

    if save_thumbnail_folder is not None:
        # Create thumbnails for each band and the blended image
        thumbnail_creator = ThumbnailCreator(
            system_index,
            region=aoi,
            save_thumbnail_folder=save_thumbnail_folder,
        )
        thumbnail_creator.create_thumbnail(s2_image, "rgb", rgb_vis)
        thumbnail_creator.create_thumbnail(mbsp, "mbsp", mbsp_vis)
        thumbnail_creator.create_thumbnail(corrected_img.select("B11"), "b11", swir_vis)
        thumbnail_creator.create_thumbnail(
            corrected_img.select("B12_corrected"), "b12_corrected", swir_vis
        )

        # Style the FeatureCollection
        fc_style = fc.style(**fc_vis).visualize()
        mbsp_img = mbsp.visualize(**mbsp_vis)
        blended = mbsp_img.blend(fc_style)
        thumbnail_creator.create_thumbnail(blended, "blended", vis_params={})

        if "hitl_fc" in locals():
            fc_style = fc.filterBounds(hitl_fc.geometry()).style(**fc_vis).visualize()
            mbsp_img = mbsp.visualize(**mbsp_vis)
            blended = mbsp_img.blend(fc_style)
            thumbnail_creator.create_thumbnail(blended, "blended_hitl", vis_params={})
    return Map


# %%
# list_of_detections = list_unreviewed_detections_from_gcs("offshore_methane")
# scenes = deduplicate_by_date_and_coords(list_of_detections)

# scenes = [
#     ("20170705T164319_20170705T165225_T15RYL", "-90.968_27.292"),
#     ("20230716T162841_20230716T164533_T15QWB", "-92.237_19.566"),
#     ("20240421T162841_20240421T164310_T15QWB", "-92.237_19.566"),
#     ("20240417T032521_20240417T033918_T48NTP", "102.987_7.593"),
#     ("20230830T162839_20230830T164019_T15QWB", "-92.291_19.601"),
# ]
row_idx = 1
sid = ""
from offshore_methane.ee_utils import sentinel2_system_indexes  # noqa

with open(cfg.EVENTS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

row = rows[row_idx]
lon, lat = float(row["lon"]), float(row["lat"])
start, end = row.get("start"), row.get("end")

if not sid:
    from datetime import datetime, timedelta

    def _add_days(ds: str, days: int, fmt: str = "%Y-%m-%d") -> str:
        return (datetime.strptime(ds, fmt) + timedelta(days=days)).strftime(fmt)

    sids = sentinel2_system_indexes(
        ee.Geometry.Point([lon, lat]), start, _add_days(end, 1)
    )
    if not sids:
        raise RuntimeError("No Sentinel-2 scenes found for selected event.")
    sid = sids[0]

print(f"sid: {sid}, lon: {lon}, lat: {lat}")
scenes = [(sid, f"{lon:.3f}_{lat:.3f}")]


start_hitl_review_loop(scenes)

# %%
