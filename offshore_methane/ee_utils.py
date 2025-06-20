# ee_utils.py
"""
Thin wrappers around the Earth-Engine Python client.
"""

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
