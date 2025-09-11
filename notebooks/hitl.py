# %%
import json
import os
from collections import defaultdict

import ee
import geemap
import geopandas as gpd
import ipywidgets as widgets
import pandas as pd
from google.cloud import storage
from IPython.display import display
from shapely.geometry import Point


def _maybe_init_ee() -> bool:
    """Quietly initialize EE; return True on success, False otherwise."""
    try:
        from offshore_methane import config as cfg  # type: ignore

        ee.Initialize(opt_url=getattr(cfg, "EE_ENDPOINT", None))
        return True
    except Exception:
        try:
            ee.Initialize()
            return True
        except Exception:
            return False


def _ipyleaflet_available() -> bool:
    try:
        import ipyleaflet  # noqa: F401

        return True
    except Exception:
        return False


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
        # Ensure both the root and per-scene subfolder exist
        os.makedirs(local_dir, exist_ok=True)
        scene_dir = os.path.join(local_dir, system_index)
        os.makedirs(scene_dir, exist_ok=True)
        local_path = os.path.join(scene_dir, filename)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved plume annotation locally to: {local_path}")
    else:
        # === Save to GCS ===
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        # Store alongside other artefacts for the scene (MBSP, VEC, etc.)
        blob = bucket.blob(f"{system_index}/{filename}")
        blob.upload_from_string(
            json.dumps(geojson), content_type="application/geo+json"
        )
        # Reflect the actual write location (hitl_prefix is not used in path)
        print(
            f"✔️ Saved plume annotation to: gs://{bucket_name}/{system_index}/{filename}"
        )


# === Reviewer Interface ===
def start_hitl_review_loop(
    detections: pd.DataFrame,
    bucket_name: str = "offshore_methane",
    save_thumbnail: bool = False,
    from_local: str | None = None,
):
    """
    Launch an interactive review interface for HITL plume annotation.

    Args:
        detections (List[Tuple[str, str]]): List of (system_index, coords) tuples.
        bucket_name (str): GCS bucket name.
    """
    save_thumbnail_folder = "../data/thumbnails" if save_thumbnail else None

    # Ensure expected columns exist
    required_cols = {"system_index", "coords"}
    missing = required_cols - set(detections.columns)
    if missing:
        raise ValueError(f"detections is missing required columns: {sorted(missing)}")

    # Optional window_id for recording HITL decisions back to CSVs
    include_wid = "window_id" in detections.columns

    def _option_value(r):
        # Encode optional fields: window_id, lon, lat for better centering
        base = [str(r["system_index"]), str(r["coords"])]
        if include_wid and pd.notna(r.get("window_id")):
            base.append(int(r["window_id"]))
        if pd.notna(r.get("lon")) and pd.notna(r.get("lat")):
            base.extend([float(r["lon"]), float(r["lat"])])
        return tuple(base)

    def _label(row):
        return f"{row['system_index']} @ {row['coords']}" + (
            f"  (wid {int(row['window_id'])})"
            if include_wid and pd.notna(row.get("window_id"))
            else ""
        )

    tmp = detections.copy()
    tmp["__label"] = tmp.apply(_label, axis=1)
    tmp = tmp.sort_values(["__label"]).reset_index(drop=True)

    scene_selector = widgets.Dropdown(
        options=[(_label(row), _option_value(row)) for _, row in tmp.iterrows()],
        description="Scene:",
        layout=widgets.Layout(width="auto"),
    )
    load_btn = widgets.Button(description="Load Scene")
    save_btn = widgets.Button(description="Save Plume")
    no_plume_btn = widgets.Button(description="Save as No Plume")
    status = widgets.Label()
    out = widgets.Output(layout=widgets.Layout(height="650px", width="100%"))
    current_map = None

    def display_scene(system_index, coords, window_id=None):
        nonlocal current_map
        if current_map is not None:
            current_map.close()

        # Preflight: if both VEC and MBSP are missing, show an inline message instead of an empty map
        vec_name = f"{system_index}_VEC_{coords}.geojson"
        mbsp_name = f"{system_index}_MBSP_{coords}.tif"
        missing_msgs = []
        try:
            if from_local:
                base = os.path.join(from_local, system_index)
                if not os.path.exists(os.path.join(base, vec_name)):
                    missing_msgs.append(f"Missing vector: {base}/{vec_name}")
                if not os.path.exists(os.path.join(base, mbsp_name)):
                    missing_msgs.append(f"Missing MBSP: {base}/{mbsp_name}")
            else:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                if not bucket.blob(f"{system_index}/{vec_name}").exists():
                    missing_msgs.append(
                        f"Missing vector: gs://{bucket_name}/{system_index}/{vec_name}"
                    )
                if not bucket.blob(f"{system_index}/{mbsp_name}").exists():
                    missing_msgs.append(
                        f"Missing MBSP: gs://{bucket_name}/{system_index}/{mbsp_name}"
                    )
        except Exception:
            # If storage check fails (e.g., auth), proceed to map rendering
            missing_msgs = []

        if len(missing_msgs) == 2:
            out.clear_output()
            with out:
                display(
                    widgets.HTML(
                        """
                        <div style='padding:12px;border:2px solid #c00;color:#c00;font-weight:600;'>
                          Required artefacts not found on the selected source. Please run the pipeline or pick another scene.
                          <ul style='margin-top:6px;'>
                            {items}
                          </ul>
                        </div>
                        """.replace(
                            "{items}",
                            "".join([f"<li>{m}</li>" for m in missing_msgs]),
                        )
                    )
                )
            status.value = "❗ Missing both VEC and MBSP artefacts."
            return

        current_map = display_s2_with_geojson(
            system_index,
            coords,
            bucket_name=bucket_name,
            save_thumbnail_folder=save_thumbnail_folder,
            local_path=from_local,
            window_id=window_id,
        )
        if current_map is None:
            status.value = "❗ No scene loaded."
            return
        # Ensure a reasonable on-screen size
        try:
            current_map.layout = widgets.Layout(height="600px", width="100%")
        except Exception:
            pass
        current_map.add_draw_control()
        out.clear_output()
        with out:
            if missing_msgs:
                msg = widgets.HTML(
                    """
                    <div style='margin-bottom:8px;padding:10px;border:1px solid #cc8;color:#a60;background:#fff8e6;'>
                      Some artefacts are missing for this scene:
                      <ul style='margin-top:6px;'>
                        {items}
                      </ul>
                    </div>
                    """.replace(
                        "{items}", "".join([f"<li>{m}</li>" for m in missing_msgs])
                    )
                )
                display(widgets.VBox([msg, current_map]))
            else:
                display(current_map)

    def on_load_clicked(b):
        val = scene_selector.value
        if isinstance(val, tuple) and len(val) >= 2:
            system_index, coords = val[0], val[1]
            wid = None
            try:
                if len(val) >= 3 and isinstance(val[2], (int, float)):
                    wid = int(val[2])
            except Exception:
                wid = None
            display_scene(system_index, coords, window_id=wid)
            status.value = f"✔️ Loaded scene: {system_index}"
        else:
            status.value = "❗ Invalid selection."

    def on_save_clicked(b):
        val = scene_selector.value
        if isinstance(val, tuple) and len(val) >= 2:
            system_index, coords = val[0], val[1]
        else:
            status.value = "❗ Invalid selection."
            return
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
        # Record HITL label in process_runs.csv if a window_id is available
        try:
            from offshore_methane.csv_utils import update_run_metrics

            window_id = None
            try:
                if (
                    isinstance(val, tuple)
                    and len(val) >= 3
                    and isinstance(val[2], (int, float))
                ):
                    window_id = int(val[2])
            except Exception:
                window_id = None
            if window_id is not None:
                update_run_metrics(window_id, system_index, hitl_value=1)
        except Exception:
            pass
        status.value = f"✔️ Saved plume for {system_index}"

    def on_no_plume_clicked(b):
        val = scene_selector.value
        if isinstance(val, tuple) and len(val) >= 2:
            system_index, coords = val[0], val[1]
        else:
            status.value = "❗ Invalid selection."
            return
        empty_geometry = None
        save_drawn_feature_to_gcs(
            empty_geometry, system_index, coords, bucket_name, local_dir=from_local
        )
        # Record HITL = 0 (no plume) when possible
        try:
            from offshore_methane.csv_utils import update_run_metrics

            window_id = None
            try:
                if (
                    isinstance(val, tuple)
                    and len(val) >= 3
                    and isinstance(val[2], (int, float))
                ):
                    window_id = int(val[2])
            except Exception:
                window_id = None
            if window_id is not None:
                update_run_metrics(window_id, system_index, hitl_value=0)
        except Exception:
            pass
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
            # coords may be "<structure_id>" now; geometry unknown without DB
            lon = lat = None
            try:
                lon, lat = map(float, coords.split("_"))
            except Exception:
                pass
            reviewed = (system_index, coords) in hitl_files
            hitl_file_size = hitl_file_sizes.get((system_index, coords), None)
            new_row = gpd.GeoDataFrame(
                [
                    {
                        "system_index": system_index,
                        "coords": coords,
                        "reviewed": reviewed,
                        "geometry": Point(lon, lat)
                        if (lon is not None and lat is not None)
                        else None,
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


def list_detections_from_db(**filters) -> pd.DataFrame:
    """
    Build the detections table from the CSV virtual database.

    Uses csv_utils.virtual_db to join windows ↔ runs ↔ granules and produces
    a DataFrame with the columns expected by the review UI:
      - system_index (granule id)
      - coords       (structure_id string used as filename suffix)
      - window_id    (for recording HITL results back to process_runs.csv)
      - reviewed     (bool; True when process_runs.hitl_value is present)

    Optional filters are passed through to csv_utils.virtual_db (e.g.,
    scene_cloud_pct, scene_sga_range, local_sga_range, local_sgi_range).
    """
    from offshore_methane.csv_utils import virtual_db

    db = virtual_db(**filters)
    if db.empty:
        return pd.DataFrame(columns=["system_index", "coords", "window_id", "reviewed"])  # type: ignore[list-item]

    # Keep rows with a concrete granule id and coordinates
    df = db.copy()
    df = df[df["system_index"].astype(str).str.len() > 0]
    df = df[
        df["lon"].notna() & df["lat"].notna()
    ]  # expect coords from windows/structures

    # Use structure_id as suffix and label; keep lon/lat for centering
    if "structure_id" in df.columns:
        df["coords"] = df["structure_id"].astype(str)
    else:
        df["coords"] = df.apply(
            lambda r: f"{float(r['lon']):.3f}_{float(r['lat']):.3f}", axis=1
        )
    reviewed_col = "hitl_value" if "hitl_value" in df.columns else None
    df["reviewed"] = df[reviewed_col].notna() if reviewed_col else False

    # Deduplicate by (system_index, coords); prefer unreviewed first
    df = df.sort_values(["reviewed"]).drop_duplicates(
        subset=["system_index", "coords"], keep="first"
    )

    keep_cols = ["system_index", "coords", "window_id", "reviewed"]
    if "lon" in df.columns and "lat" in df.columns:
        keep_cols += ["lon", "lat"]
    return df[keep_cols].reset_index(drop=True)


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
    window_id=None,
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

    # Ensure EE is initialized (avoid noisy stack traces in notebooks)
    if not _maybe_init_ee():
        print(
            "Earth Engine is not initialized. Run ee.Authenticate(); ee.Initialize() and retry."
        )
        return None

    # === Resolve lon/lat and a default AOI ===
    lon = lat = None
    # Try to parse from the selector value by re-reading last selection via a global
    # or attempt to parse coords as "lon_lat" for backward compatibility.
    try:
        lon, lat = map(float, coords.split("_"))
    except Exception:
        pass
    # If not parseable, try to derive from DB
    if lon is None or lat is None:
        try:
            from offshore_methane.csv_utils import df_for_granule

            df = df_for_granule(system_index)
            if window_id is not None:
                try:
                    df = df[df["window_id"].astype("Int64") == int(window_id)]
                except Exception:
                    pass
            if not df.empty and df["lon"].notna().any() and df["lat"].notna().any():
                lon = float(df.iloc[0]["lon"])  # type: ignore[arg-type]
                lat = float(df.iloc[0]["lat"])  # type: ignore[arg-type]
        except Exception:
            pass
    if lon is None or lat is None:
        print("Could not resolve lon/lat for the selected scene.")
        return None
    default_radius = 5000
    try:
        from offshore_methane import config as _cfg  # local import to avoid cycles

        default_radius = int(_cfg.MASK_PARAMS["dist"]["export_radius_m"])  # type: ignore[index]
    except Exception:
        pass
    point = ee.Geometry.Point(lon, lat)
    aoi = point.buffer(default_radius)

    # === Try to load VEC GeoJSON (optional) ===
    fc = None
    filename = f"{system_index}_VEC_{coords}.geojson"
    client = None
    bucket = None
    if local_path:
        vec_path = os.path.join(local_path, system_index, filename)
        if os.path.exists(vec_path):
            try:
                with open(vec_path, "r") as f:
                    geojson = json.load(f)
                fc = geemap.geojson_to_ee(geojson)
                aoi = fc.geometry()
            except Exception:
                pass
        else:
            print(f"Vector not found locally (continuing without it): {vec_path}")
    else:
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_path = f"{system_index}/{filename}"
            blob = bucket.blob(blob_path)
            if blob.exists():
                geojson = json.loads(blob.download_as_text())
                fc = geemap.geojson_to_ee(geojson)
                aoi = fc.geometry()
            else:
                print(f"Vector not found in GCS (continuing without it): {blob_path}")
        except Exception as e:
            print(f"Error loading vector from GCS: {e}")

    # === Load Sentinel-2 image from Earth Engine ===
    s2_image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", system_index))
        .first()
    )

    # EE returns an object even when empty; proceed and let visualization fail gracefully

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
    try:
        B11_median = (
            s2_image.select("B11")
            .reduceRegion(
                reducer=ee.Reducer.median(), geometry=aoi, bestEffort=True, scale=20
            )
            .getInfo()
            .get("B11", 1500)
        )
        if B11_median is None:
            B11_median = 1500
    except Exception:
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
    # fc_vis = {"color": "green", "width": 2}

    corrected_img = linearFit(s2_image)
    Map = geemap.Map()
    try:
        Map.layout = widgets.Layout(height="600px", width="100%")
    except Exception:
        pass

    Map.addLayer(s2_image, rgb_vis, f"RGB {system_index}")
    Map.addLayer(s2_image.select("B11"), swir_vis, "B11")
    Map.addLayer(corrected_img.select("B12_corrected"), swir_vis, "B12 * c_fit")

    # === Load MBSP layer (optional) ===
    mbsp_filename = f"{system_index}_MBSP_{coords}.tif"
    if local_path:
        mbsp_path = os.path.join(local_path, system_index, mbsp_filename)
        if not os.path.exists(mbsp_path):
            print(f"MBSP file not found locally (skipping): {mbsp_path}")
        else:
            Map.add_raster(
                mbsp_path,
                vmin=-0.2,
                vmax=0.2,
                colormap=["red", "white", "blue"],
                layer_name="MBSP",
                zoom_to_layer=False,
            )
    else:
        try:
            # Check existence before attempting to visualize
            if bucket is None:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
            mbsp_blob = bucket.blob(f"{system_index}/{mbsp_filename}")
            if mbsp_blob.exists():
                mbsp = ee.Image.loadGeoTIFF(
                    f"gs://{bucket_name}/{system_index}/{mbsp_filename}"
                )
                Map.addLayer(mbsp, mbsp_vis, "MBSP")
            else:
                print(
                    f"MBSP not found in GCS (skipping): {system_index}/{mbsp_filename}"
                )
        except Exception as e:
            print(f"Error loading MBSP from GCS: {e}")

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
        try:
            if bucket is None:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
            hitl_blob_path = f"{system_index}/{hitl_filename}"
            hitl_blob = bucket.blob(hitl_blob_path)
            if hitl_blob.exists():
                hitl_geojson = json.loads(hitl_blob.download_as_text())
                if hitl_geojson["features"][0]["geometry"] is not None:
                    hitl_fc = geemap.geojson_to_ee(hitl_geojson)
                    Map.addLayer(hitl_fc, {"color": "red"}, "Existing HITL Plume")
        except Exception as e:
            print(f"Error loading HITL layer: {e}")

    # === Add detection vector and point ===
    if fc is not None:
        Map.addLayer(fc, {}, "Vector")
    else:
        # Add a simple buffered ROI as context when VEC is absent
        Map.addLayer(aoi, {"color": "green"}, "ROI")
    Map.addLayer(point, {"color": "red"}, "Target")
    Map.centerObject(point, 14)
    try:
        Map.addLayerControl()
    except Exception:
        pass

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


# %%
from offshore_methane import config as cfg  # noqa: E402

# Preferred: build from CSV virtual DB (honours default scene filters)
detections_df = list_detections_from_db()
if detections_df.empty:
    # Fallback: scan bucket/local for available vector/HITL files
    src = cfg.EXPORT_PARAMS.get("bucket") or "offshore_methane"
    detections_df = list_detections(src)

# Decide storage source for artefacts
bucket = cfg.EXPORT_PARAMS.get("bucket", "offshore_methane")
use_local = (
    str(cfg.EXPORT_PARAMS.get("preferred_location", "bucket")).lower() == "local"
)
local_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

start_hitl_review_loop(
    detections_df,
    bucket_name=bucket,
    from_local=(local_data if use_local else None),
)

# %%
# # Optional: use local folder instead of bucket
# detections_local = list_detections(cfg.EXPORT_PARAMS.get("bucket"), local_path="../data")
# start_hitl_review_loop(detections_local, from_local="../data")


# %%
