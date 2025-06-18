"""
Thin wrappers around the Earth-Engine Python client.
"""

import ee
import numpy as np

ee.Initialize()  # single global EE session


# ------------------------------------------------------------------
def sentinel2_product_ids(
    point: ee.Geometry, start: str, end: str, cloud_pct: int
) -> list[str]:
    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(point)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    return sorted(set(coll.aggregate_array("PRODUCT_ID").getInfo()))


def ee_asset_exists(asset_id: str) -> bool:
    try:
        ee.Image(asset_id).getInfo()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------
#  Uniform wrappers around EE batch export APIs.
# ------------------------------------------------------------------
def export_image(
    image: ee.Image,
    description: str,
    region: ee.Geometry,
    preferred_location: str,
    bucket: str,
    ee_asset_folder: str,
    **kwargs,
):
    roi = region.bounds().coordinates().getInfo()
    task = None

    if preferred_location == "bucket":
        utm = image.select("MBSP").projection()
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=description,
            bucket=bucket,
            fileNamePrefix=f"MBSP/{description}",
            region=roi,
            scale=20,
            crs=utm,
            maxPixels=1 << 36,
        )
    elif preferred_location == "ee_asset_folder":
        utm = image.select("MBSP").projection()
        task = ee.batch.Export.image.toAsset(
            image=image,
            description=description,
            assetId=f"{ee_asset_folder}/{description}",
            region=roi,
            scale=20,
            crs=utm,
            maxPixels=1 << 36,
            pyramidingPolicy={"MBSP": "sample"},
        )
    if task:
        task.start()
    return task


def export_polygons(
    fc: ee.FeatureCollection,
    description: str,
    preferred_location: str,
    bucket: str,
    ee_asset_folder: str,
    **kwargs,
):
    task = None

    if preferred_location == "bucket":
        task = ee.batch.Export.table.toCloudStorage(
            collection=fc,
            description=f"{description}_vect",
            bucket=bucket,
            fileNamePrefix=f"vectors/{description}",
            fileFormat="GeoJSON",
        )
    elif preferred_location == "ee_asset_folder":
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description=f"{description}_vect",
            assetId=f"{ee_asset_folder}/{description}_vect",
        )
    if task:
        task.start()
    return task


# ------------------------------------------------------------------
#  Scene-level & local cloud / glint rejection tests.
# ------------------------------------------------------------------
def _qa60_cloud_mask(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloudy = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
    return img.updateMask(cloudy.Not())


# ------------------------------------------------------------------
#  Scene-level sun-glint (metadata only – cheap)
# ------------------------------------------------------------------
def scene_glint_check(img: ee.Image, scene_sga_range: tuple[float, float]) -> bool:
    try:
        sza = ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE"))
        saa = ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        vza = ee.Number(img.get("MEAN_INCIDENCE_ZENITH_ANGLE_B11"))
        vaa = ee.Number(img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B11"))

        sza_r = sza.multiply(np.pi / 180)
        vza_r = vza.multiply(np.pi / 180)
        dphi_r = saa.subtract(vaa).abs().multiply(np.pi / 180)

        cos_sga = (
            sza_r.cos()
            .multiply(vza_r.cos())
            .subtract(sza_r.sin().multiply(vza_r.sin()).multiply(dphi_r.cos()))
        )
        sga_deg = cos_sga.acos().multiply(180 / np.pi)

        ok = sga_deg.gt(scene_sga_range[0]).And(sga_deg.lt(scene_sga_range[1]))
        return ok.getInfo()
    except Exception:
        return True  # missing metadata → keep scene


# ------------------------------------------------------------------
#  Local (5 km) tests – require pixels/coarse SGA grid
# ------------------------------------------------------------------
def local_cloud_check(
    img: ee.Image, centre: ee.Geometry, aoi_radius_m: int, local_max_cloud: int
) -> bool:
    cloudy = _qa60_cloud_mask(img).Not()
    frac = (
        cloudy.reduceRegion(
            ee.Reducer.mean(), centre.buffer(aoi_radius_m), 60, bestEffort=True
        )
        .values()
        .get(0)
    )
    return ee.Number(frac).multiply(100).lte(local_max_cloud).getInfo()


def local_glint_check(
    sga_img: ee.Image,
    centre: ee.Geometry,
    aoi_radius_m: int,
    local_sga_range: tuple[float, float],
) -> bool:
    sga_mean = (
        sga_img.reduceRegion(ee.Reducer.mean(), centre.buffer(aoi_radius_m), 5000)
        .values()
        .get(0)
    )
    ok = (
        ee.Number(sga_mean)
        .gt(local_sga_range[0])
        .And(ee.Number(sga_mean).lt(local_sga_range[1]))
    )
    return ok.getInfo()
