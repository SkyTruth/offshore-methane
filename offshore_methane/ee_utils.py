# ee_utils.py
"""
Thin wrappers around the Earth-Engine Python client.
"""

import ee
import numpy as np

ee.Initialize()  # single global EE session


# ------------------------------------------------------------------
def sentinel2_system_indexes(
    point: ee.Geometry,
    start: str,
    end: str,
    cloud_pct: int,
    sga_range: tuple[float, float] = (0.0, 25.0),
) -> list[str]:
    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(point)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(lambda img: add_sga_ok(img, sga_range))
        .filter(ee.Filter.eq("SGA_OK", 1))
    )

    return sorted(set(coll.aggregate_array("system:index").getInfo()))


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
        "SGA_OK",
        ee.Algorithms.If(
            img.propertyNames().contains("MEAN_SOLAR_ZENITH_ANGLE"), ok, True
        ),
    )


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
