# %%
# ee_utils.py
"""
Thin wrappers around the Earth-Engine Python client.
"""

import json
import math
import subprocess
import time
from pathlib import Path

import ee
import geemap
import requests
from requests.exceptions import ConnectionError, HTTPError

import offshore_methane.config as cfg
from offshore_methane.gcp_utils import gsutil_cmd

ee.Initialize()  # single global EE session


def quick_view(system_index, region=None, bands=["B4", "B3", "B2"]):
    """
    Display a Sentinel-2 image by system:index from S2_HARMONIZED using geemap.
    Optionally, zoom to a given region (ee.Geometry).
    Allows custom bands and autoscaled visualization.
    """
    system_index = system_index[:38]

    # Find the image by system:index
    coll = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filter(
        ee.Filter.eq("system:index", system_index)
    )
    img = coll.first()

    # Autoscale: get min/max for each band in the region or image footprint
    scale_region = region if region is not None else img.geometry()
    stats = (
        img.select(bands)
        .reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=scale_region,
            scale=20,
            bestEffort=True,
        )
        .getInfo()
    )

    # Build min/max lists for visualization
    min_vals = [stats.get(f"{b}_p2", 0) for b in bands]
    max_vals = [stats.get(f"{b}_p98", 3000) for b in bands]

    vis_params = {
        "bands": bands,
        "min": min_vals,
        "max": max_vals,
        "gamma": 1.2,
    }

    Map = geemap.Map()
    Map.addLayer(img, vis_params, f"S2 {system_index}")
    if region is not None:
        Map.centerObject(region, 10)
    else:
        Map.centerObject(img, 10)
    Map.addLayerControl()
    return Map


# ------------------------------------------------------------------
def sentinel2_system_indexes(
    point: ee.Geometry,
    start: str,
    end: str,
) -> list[dict]:
    """
    Return scene metadata for S2 scenes over 'point' in [start, end):
      [{ 'system_index': str, 'sunglint': float|None, 'cloudiness': float|None }, ...]

    Applies only existence checks for solar/incidence angles; no filtering by
    SGA or cloudiness.
    """
    from offshore_methane.masking import scene_sga_filter

    # Require solar angles to exist
    solar_required = ee.Filter.notNull(
        ["MEAN_SOLAR_AZIMUTH_ANGLE", "MEAN_SOLAR_ZENITH_ANGLE"]
    )

    # Require B8A incidence-angle pair to exist
    inc_filter = ee.Filter.notNull(["MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A", "MEAN_INCIDENCE_ZENITH_ANGLE_B8A"])

    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(point)
        .filter(solar_required)
        .filter(inc_filter)
        .map(lambda img: ee.Image(img).set("SGA_SCENE", scene_sga_filter(img, cfg.MASK_PARAMS)))
    )

    sids = coll.aggregate_array("system:index").getInfo()
    sgas = coll.aggregate_array("SGA_SCENE").getInfo()
    clouds = coll.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    def _sensing_iso_from_sid(sid: str) -> str:
        # system:index looks like YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_TxxYYY
        try:
            first = sid.split("_")[0]
            date = first[:8]
            time = first[9:15]
            return f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:{time[4:6]}"
        except Exception:
            return ""

    times = [_sensing_iso_from_sid(s) for s in sids]

    out = []
    for sid, sga, cl, ts in zip(sids, sgas, clouds, times):
        out.append(
            {
                "system_index": sid,
                "sunglint": None if sga is None else float(sga),
                "cloudiness": None if cl is None else float(cl),
                "timestamp": ts,
            }
        )
    return out




def _ee_asset_info(asset_id: str) -> dict | None:
    """
    Return the parsed JSON from `earthengine asset info <id>` or None
    if the asset does not exist (return-code ≠ 0).
    """
    res = subprocess.run(
        ["earthengine", "asset", "info", asset_id],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return None


def ee_asset_exists(asset_id: str) -> bool:
    """True if *any* EE asset with that ID exists (Image, Table, …)."""
    return _ee_asset_info(asset_id) is not None


def ee_asset_ready(asset_id: str) -> bool:
    """
    Ready ⇔ asset exists *and* (for Images) has non-empty 'bands'.
    For non-image assets we treat mere existence as 'ready'.
    """
    info = _ee_asset_info(asset_id)
    if not info:
        return False
    if info.get("type") == "Image":
        return bool(info.get("bands"))
    return True


def process_sid_into_gcs_xml_address(sid: str) -> str:
    """
    Get the product URI and granule ID for a given system index.
    """
    image = ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")

    tile_ids = sid.split("_")[2]

    # These will be used to get our product of interest.
    tile_num = tile_ids[1:3]
    lat_band = tile_ids[3]
    grid_square = tile_ids[4:]
    product_uri = image.get("PRODUCT_ID").getInfo()
    granule_id = image.get("GRANULE_ID").getInfo()

    # Name of xml product to query, which holds metadata for solar angles.
    # ex: gcp-public-data-sentinel-2/tiles/15/R/XL/S2B_MSIL1C_20170705T164319_N0205_R126_T15RXL_20170705T165225.SAFE/GRANULE/L1C_T15RXL_A001725_20170705T165225/MTD_TL.xml
    file_id = f"tiles/{tile_num}/{lat_band}/{grid_square}/{product_uri}.SAFE/GRANULE/{granule_id}/MTD_TL.xml"
    return file_id


# ------------------------------------------------------------------
#  Simple URL→file helper used by local exports
# ------------------------------------------------------------------
def _download_url(url, dest, chunk=8192, *, max_retries=5, backoff=1.5):
    """
    Stream-download a signed EE URL with automatic exponential-backoff retries.

    Parameters
    ----------
    url : str
        Signed Earth-Engine download URL.
    dest : pathlib.Path
        Output file path.
    chunk : int
        Bytes per streamed block.
    max_retries : int
        Attempts before giving up.
    backoff : float
        Initial sleep (s); doubled each retry.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    attempt = 0
    while True:
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for block in r.iter_content(chunk):
                        if block:  # ignore keep-alives
                            fh.write(block)
            return  # success
        except HTTPError as exc:
            code = exc.response.status_code if exc.response else None
            attempt += 1
            if attempt > max_retries or code not in (429, 500, 502, 503, 504):
                raise  # unrecoverable
            sleep = backoff * (2 ** (attempt - 1))
            if cfg.VERBOSE:
                print(f"  ↻ HTTP {code}, retry {attempt}/{max_retries} in {sleep:.1f}s")
            time.sleep(sleep)
        except ConnectionError:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep = backoff * (2 ** (attempt - 1))
            if cfg.VERBOSE:
                print(
                    f"  ↻ connection error, retry {attempt}/{max_retries} in {sleep:.1f}s"
                )
            time.sleep(sleep)


# ------------------------------------------------------------------
#  Uniform wrappers around EE batch export APIs.
# ------------------------------------------------------------------
def _prepare_asset(
    asset_id: str,
    *,
    overwrite: bool = False,
    timeout: int = 300,
    datatype: str = "asset",
) -> bool:
    """
    Decide whether the caller should start a new export for *asset_id*.

    Returns
    -------
    bool
        True  → caller must export (asset missing or we deleted it for overwrite)
        False → caller should SKIP (asset is already ready / became ready)

    Behaviour
    ---------
    • overwrite = True
        - if the asset exists (ready or ingesting) → delete it, return True
        - if it doesn't exist                           → return True
    • overwrite = False
        - if the asset is ready                        → return False
        - if ingesting → wait ≤ timeout until ready    → return False
        - if missing                                   → return True
    """
    # ------------------------------------------------------------------ overwrite branch
    if overwrite and ee_asset_exists(asset_id):
        if cfg.VERBOSE:
            print(f"  ↻ deleting existing {datatype} asset → {asset_id}")
        try:
            ee.data.deleteAsset(asset_id)
        except Exception as exc:
            # EE occasionally throws 404 if asset vanished in the meantime
            if "Asset not found" not in str(exc):
                raise
        # ensure eventual-consistency before we re-export
        t0 = time.time()
        while ee_asset_exists(asset_id):
            if time.time() - t0 > 30:  # safety: 30 s should be plenty
                raise RuntimeError(f"Timed out deleting {asset_id}")
            time.sleep(2)
        return True  # caller must export afresh

    # -------------------------------------------------------------- non-overwrite branch
    if not ee_asset_exists(asset_id):
        return True  # missing → export required

    if ee_asset_ready(asset_id):
        if cfg.VERBOSE:
            print(f"  ✓ {datatype} asset exists → {asset_id} (skipped)")
        return False  # already good

    # asset exists but still ingesting
    if cfg.VERBOSE:
        print(f"  … waiting for {datatype} asset ingestion → {asset_id}")
    t0 = time.time()
    while not ee_asset_ready(asset_id):
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"{datatype} asset {asset_id} still ingesting after {timeout}s"
            )
        time.sleep(5)
    if cfg.VERBOSE:
        print(f"  ✓ {datatype} asset now ready → {asset_id} (skipped)")
    return False


def export_image(
    image: ee.Image,
    sid: str,
    region: ee.Geometry,
    preferred_location: str,
    bucket: str,
    ee_asset_folder: str,
    **kwargs,
) -> tuple[ee.batch.Task | None, bool]:
    """
    Returns (task_or_None, exported_bool)
    """
    datatype = "MBSP"
    overwrite: bool = kwargs.get("overwrite", False)
    timeout: int = kwargs.get("timeout", 300)
    roi = region.bounds().coordinates().getInfo()
    task, exported = None, False

    if preferred_location == "bucket":
        # Skip if object exists and overwrite is False
        gcs_path = f"gs://{bucket}/{sid}/{sid}_{datatype}.tif"
        already = (
            subprocess.run(
                [gsutil_cmd(), "ls", gcs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if already and not overwrite:
            if cfg.VERBOSE:
                print(f"  ✓ raster exists → {gcs_path} (skipped)")
            return None, False

        utm = image.select(datatype).projection()
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=f"{sid}_{datatype}",
            bucket=bucket,
            fileNamePrefix=f"{sid}/{sid}_{datatype}",
            region=roi,
            scale=20,
            crs=utm,
            maxPixels=1 << 36,
        )
        exported = True

    elif preferred_location == "ee_asset_folder":
        asset_id = f"{ee_asset_folder}/{sid}_{datatype}"
        if _prepare_asset(
            asset_id, overwrite=overwrite, timeout=timeout, datatype="raster"
        ):
            utm = image.select(datatype).projection()
            task = ee.batch.Export.image.toAsset(
                image=image,
                description=f"{sid}_{datatype}",
                assetId=asset_id,
                region=roi,
                scale=20,
                crs=utm,
                maxPixels=1 << 36,
                pyramidingPolicy={datatype: "sample"},
            )
            exported = True

    else:  # preferred_location == "local"
        out_path = Path("../data") / sid / f"{sid}_{datatype}.tif"
        if out_path.is_file() and not overwrite:
            if cfg.VERBOSE:
                print(f"  ✓ raster exists → {out_path} (skipped)")
            return None, False
        url = image.clip(region).getDownloadURL(
            {
                "scale": 20,
                "region": roi,
                "crs": image.select(datatype).projection(),
                "format": "GEO_TIFF",
            }
        )
        if cfg.VERBOSE:
            print(f"  ↓ raster → {out_path}")
        _download_url(url, out_path)
        exported = True

    if task:
        task.start()
    return task, exported


def export_polygons(
    fc: ee.FeatureCollection,
    sid: str,
    suffix: str,
    preferred_location: str,
    bucket: str,
    ee_asset_folder: str,
    **kwargs,
) -> tuple[ee.batch.Task | None, bool]:
    """
    Returns (task_or_None, exported_bool)
    """
    datatype = "VEC"
    overwrite: bool = kwargs.get("overwrite", False)
    timeout: int = kwargs.get("timeout", 300)
    task, exported = None, False

    if preferred_location == "bucket":
        gcs_path = f"gs://{bucket}/{sid}/{sid}_{datatype}_{suffix}.geojson"
        already = (
            subprocess.run(
                [gsutil_cmd(), "ls", gcs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if already and not overwrite:
            if cfg.VERBOSE:
                print(f"  ✓ vectors exist → {gcs_path} (skipped)")
            return None, False

        task = ee.batch.Export.table.toCloudStorage(
            collection=fc,
            description=f"{sid}_{datatype}",
            bucket=bucket,
            fileNamePrefix=f"{sid}/{sid}_{datatype}_{suffix}",
            fileFormat="GeoJSON",
        )
        exported = True

    elif preferred_location == "ee_asset_folder":
        asset_id = f"{ee_asset_folder}/{sid}_{datatype}_{suffix}"
        if _prepare_asset(
            asset_id, overwrite=overwrite, timeout=timeout, datatype="vector"
        ):
            task = ee.batch.Export.table.toAsset(
                collection=fc,
                description=f"{sid}_{datatype}",
                assetId=asset_id,
            )
            exported = True

    else:  # preferred_location == "local"
        out_path = Path("../data") / sid / f"{sid}_{datatype}_{suffix}.geojson"
        if out_path.is_file() and not overwrite:
            if cfg.VERBOSE:
                print(f"  ✓ vectors exist → {out_path} (skipped)")
            return None, False
        url = fc.getDownloadURL(filetype="geojson")
        if cfg.VERBOSE:
            print(f"  ↓ vectors → {out_path}")
        _download_url(url, out_path)
        exported = True

    if task:
        task.start()
    return task, exported


def get_wind_layers(img: ee.Image, time_window: int = 3) -> ee.Image:
    """
    Compute wind speed (m s-1) and direction (degrees) at the timestamp of *img*
    using NOAA/CFSV2/FOR6H re-analysis.

    Parameters
    ----------
    img : ee.Image
        Any EE image that carries a “system:time_start” property.
    time_window : int, optional
        Hours before/after *img*’s timestamp to search for the closest
        forecast record (default: 3 h).

    Returns
    -------
    ee.Image
        Two-band image:
          • wind_speed (m s-1)
          • wind_dir   (degrees, 0 = east, counter-clockwise positive)
    """
    # Timestamp of the reference image
    t0 = ee.Date(img.get("system:time_start"))

    # Grab the CFSv2 record closest in time to t0
    met = (
        ee.ImageCollection("NOAA/CFSV2/FOR6H")
        .filterDate(t0.advance(-time_window, "hour"), t0.advance(time_window, "hour"))
        .sort("system:time_start")
        .first()
    )

    if met is None:
        raise ValueError("No CFSv2 image found in the requested window.")

    # Re-project to ~1 km for smoother spatial gradients
    met = met.resample("bilinear").reproject(crs="EPSG:4326", scale=1_000)

    # Wind components (10 m above ground)
    u = met.select("u-component_of_wind_height_above_ground")
    v = met.select("v-component_of_wind_height_above_ground")

    # Magnitude
    speed = u.hypot(v).rename("wind_speed")

    # Direction (0° = east, counter-clockwise +)
    direction = (
        v.atan2(u)  # radians
        .multiply(180 / math.pi)  # → degrees
        .add(360)
        .mod(360)
        .rename("wind_dir")
    )

    return speed.addBands(direction).copyProperties(met, ["system:time_start"])


# %%
# quick_view("20170705T164319_20170705T165225_T15RXL")
# %%
