# ee_utils.py
"""
Thin wrappers around the Earth-Engine Python client.
"""

import json
import subprocess
import time
from pathlib import Path

import ee
import numpy as np
import geemap

ee.Initialize()  # single global EE session


def quick_view(system_index, region=None, bands=["B4", "B3", "B2"]):
    """
    Display a Sentinel-2 image by system:index from S2_HARMONIZED using geemap.
    Optionally, zoom to a given region (ee.Geometry).
    Allows custom bands and autoscaled visualization.
    """
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
    sga_range: tuple[float, float] = (0.0, 25.0),
) -> list[str]:
    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(point)
        .filter(scene_cloud_filter(MP))
        .map(lambda img: scene_sga_filter(img, MP))
        .filter(ee.Filter.eq("SGA_OK", 1))
    )

    return sorted(set(coll.aggregate_array("system:index").getInfo()))


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


# ------------------------------------------------------------------
#  Simple URL→file helper used by local exports
# ------------------------------------------------------------------
def _download_url(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """
    Stream `url` to `dest`, creating parent folders if needed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for block in r.iter_content(chunk):
                fh.write(block)


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
        print(f"  ✓ {datatype} asset exists → {asset_id} (skipped)")
        return False  # already good

    # asset exists but still ingesting
    print(f"  … waiting for {datatype} asset ingestion → {asset_id}")
    t0 = time.time()
    while not ee_asset_ready(asset_id):
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"{datatype} asset {asset_id} still ingesting after {timeout}s"
            )
        time.sleep(5)
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
                ["gsutil", "ls", gcs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if already and not overwrite:
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
                ["gsutil", "ls", gcs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if already and not overwrite:
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
            print(f"  ✓ vectors exist → {out_path} (skipped)")
            return None, False
        url = fc.getDownloadURL(filetype="geojson")
        print(f"  ↓ vectors → {out_path}")
        _download_url(url, out_path)
        exported = True

    if task:
        task.start()
    return task


# ------------------------------------------------------------------
#  Scene-level & local cloud / glint rejection tests.
# ------------------------------------------------------------------
def _mask_land(image):
    """
    Use JAXA/GCOM-C LST as a land mask. We take the mean over Jan 2024,
    then mask out any pixel where LST_AVE is non-zero (i.e., keep non-land).
    """
    # Filter the land surface temperature collection to January 2024.
    lst_collection = ee.ImageCollection("JAXA/GCOM-C/L3/LAND/LST/V3").filterDate(
        "2024-01-01", "2024-02-01"
    )
    lst_mosaic = lst_collection.mean()
    # Any pixel with a non-zero LST_AVE is considered land.
    land_masker = lst_mosaic.select("LST_AVE").reduce(ee.Reducer.anyNonZero())

    # Keep only pixels where land_masker == 0 (i.e., water or non-land).
    return image.updateMask(land_masker.unmask(0).eq(0))


def _qa60_cloud_mask(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloudy = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
    return img.updateMask(cloudy.Not())


def _cloud_color_mask(img: ee.Image) -> ee.Image:
    """Adds bands to identify clouds based on RGB whiteness.

    Adds:
        - 'color_cloud': binary mask (1 = likely cloud)
        - 'brightness': mean of RGB reflectance
        - 'rgb_std': standard deviation of RGB reflectance
    """

    # Select and scale RGB bands (assuming DN values; scale to reflectance)
    rgb = img.select(["B4", "B3", "B2"])  # Red, Green, Blue

    # Calculate brightness (mean) and std deviation (color similarity)
    brightness = rgb.reduce(ee.Reducer.mean()).rename("brightness")
    rgb_std = rgb.reduce(ee.Reducer.stdDev()).rename("rgb_std")

    # Default thresholds (can be adjusted experimentally)
    bright_thresh = 2000
    std_thresh = 200

    # Identify bright, low-saturation (white-ish) pixels
    # cloud_mask = brightness.gt(bright_thresh).And(rgb_std.lt(std_thresh))
    cloud_mask = brightness.lt(bright_thresh).Or(rgb_std.gt(std_thresh))

    # Add all three bands
    # return img.addBands([brightness, rgb_std, cloud_mask.rename("color_cloud").uint8()])
    return img.updateMask(cloud_mask)


# ------------------------------------------------------------------
#  Scene-level sun-glint (metadata only – cheap)
# ------------------------------------------------------------------
def add_sga_ok(img: ee.Image, sga_range: tuple[float, float] = (0.0, 25.0)) -> ee.Image:
    """Add a Boolean property 'SGA_OK' based on scene-glint angle limits."""
    # metadata → Numbers
    sza = ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE"))
    saa = ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    vza = ee.Number(img.get("MEAN_INCIDENCE_ZENITH_ANGLE_B11"))
    vaa = ee.Number(img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B11"))

    # radians
    rad = ee.Number(np.pi).divide(180)
    sza_r = sza.multiply(rad)
    vza_r = vza.multiply(rad)
    dphi_r = saa.subtract(vaa).abs().multiply(rad)

    cos_sga = (
        sza_r.cos()
        .multiply(vza_r.cos())
        .subtract(sza_r.sin().multiply(vza_r.sin()).multiply(dphi_r.cos()))
    )
    sga_deg = cos_sga.acos().multiply(180 / np.pi)

    ok = sga_deg.gt(sga_range[0]).And(sga_deg.lt(sga_range[1]))

    # keep images that *lack* metadata
    return img.set(
        {
            "SGA_DEG": sga_deg,
            "SGA_OK": ee.Algorithms.If(
                img.propertyNames().contains("MEAN_SOLAR_ZENITH_ANGLE"), ok, True
            ),
        }
    )


# ------------------------------------------------------------------
# Local-level glint – SGI
# ------------------------------------------------------------------


def add_sgi(image: ee.Image) -> ee.Image:
    image = _cloud_color_mask(image)
    image = _mask_land(image)
    # --- bands to 0-1 reflectance ----------------------------------
    b12 = image.select("B12").divide(10000)  # SWIR 20 m
    b11 = image.select("B11").divide(10000)

    # average VIS (B02 + B03 + B04) → resample to 20 m
    b02 = (
        image.select("B2")
        .divide(10000)
        .resample("bilinear")
        .reproject(crs=b11.projection())
    )
    b03 = (
        image.select("B3")
        .divide(10000)
        .resample("bilinear")
        .reproject(crs=b11.projection())
    )
    b04 = (
        image.select("B4")
        .divide(10000)
        .resample("bilinear")
        .reproject(crs=b11.projection())
    )
    b_vis = b02.add(b03).add(b04).divide(3).rename("Bvis_20m")

    # Sun–glint index (Varon 2021, eqn 4)
    denom = b12.add(b_vis).max(1e-4)
    sgi = b12.subtract(b_vis).divide(denom).clamp(-1, 1).rename("sgi")
    return image.addBands(sgi)


def add_sgi_b3(image):
    image = _cloud_color_mask(image)
    image = _mask_land(image)

    # --- bands to 0-1 reflectance ----------------------------------
    b12 = image.select("B12").divide(10000)  # SWIR 20 m
    b11 = image.select("B11").divide(10000)

    b03 = (
        image.select("B3")
        .divide(10000)
        .resample("bilinear")
        .reproject(crs=b11.projection())
    )

    # Sun–glint index (Varon 2021, eqn 4)
    denom = b12.add(b03).max(1e-4)
    sgi = b12.subtract(b03).divide(denom).clamp(-1, 1).rename("sgi_b3")
    return image.addBands(sgi)


# ------------------------------------------------------------------
#  Local (5 km) tests – require pixels/coarse SGA grid
# ------------------------------------------------------------------
def _wind_test(s2, centre, aoi_radius_m, max_wind_10m):
    acq_time = ee.Date(s2.get("system:time_start"))
    img = (
        ee.ImageCollection("NOAA/CFSV2/FOR6H")
        .filterDate(
            acq_time.advance(-3, "hour"),  # ±1 h catches the nearest 00:30 stamp
            acq_time.advance(3, "hour"),
        )
        .first()
    )

    u = img.select(
        "u-component_of_wind_height_above_ground"
    )  # or CFSv2 band names if you chose that
    v = img.select("v-component_of_wind_height_above_ground")
    w = u.hypot(v).rename("wind10")

    local_mean = (
        w.reduceRegion(
            ee.Reducer.mean(), centre.buffer(aoi_radius_m), 20_000, bestEffort=True
        )
        .values()
        .get(0)
    )

    return ee.Algorithms.If(
        local_mean, ee.Number(local_mean).lte(max_wind_10m), ee.Number(1)
    )


def _cloud_test(s2, centre, aoi_radius_m, local_max_cloud):
    cloud_frac = (
        _qa60_cloud_mask(s2)
        .Not()
        .reduceRegion(
            ee.Reducer.mean(),
            centre.buffer(aoi_radius_m),
            60,
            bestEffort=True,
        )
        .values()
        .get(0)
    )
    return ee.Number(cloud_frac).multiply(100).lte(local_max_cloud)


def _glint_test(sga_img, centre, aoi_radius_m, local_sga_range):
    sga_mean = (
        sga_img.reduceRegion(
            ee.Reducer.mean(), centre.buffer(aoi_radius_m), 5000, bestEffort=True
        )
        .values()
        .get(0)
    )

    return (
        ee.Number(sga_mean)
        .gt(local_sga_range[0])
        .And(ee.Number(sga_mean).lt(local_sga_range[1]))
    )


def product_ok(
    s2: ee.Image,
    sga_img: ee.Image,
    centre: ee.Geometry,
    aoi_radius_m: int,
    local_max_cloud: int,
    local_sga_range: tuple[float, float],
    max_wind_10m: float,
) -> ee.ComputedObject:
    """
    True/False quality gate for a Sentinel-2 product:
        • cloud fraction inside AOI ≤ local_max_cloud
        • mean SGA inside AOI within local_sga_range
        • CFSv2 10 m wind (local) ≤ max_wind_10m
    """
    # ---------- combined verdict -----------------------------------------
    return (
        _cloud_test(s2, centre, aoi_radius_m, local_max_cloud)
        .And(_glint_test(sga_img, centre, aoi_radius_m, local_sga_range))
        .And(_wind_test(s2, centre, aoi_radius_m, max_wind_10m))
    )
