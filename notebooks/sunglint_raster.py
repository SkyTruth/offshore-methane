import ee

ee.Initialize()

from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime, timezone


def expanded_mosaic(ref_img):
    ref_time = ee.Date(ref_img.get("system:time_start"))

    # Step 3: Filter all S2_SR tiles within ±1 min of this timestamp
    collection = ee.ImageCollection("COPERNICUS/S2_SR").filterDate(
        ref_time.advance(-1, "minute"), ref_time.advance(1, "minute")
    )
    # Step 4: Mosaic
    return collection.mosaic()


def print_s2_metadata_angles(img):
    """
    Prints Sentinel-2 solar angles from metadata + Pysolar estimate,
    and assumes nadir (0° zenith) satellite view for simplicity.
    """
    # 1. Get image center and time
    centroid = img.geometry().centroid()
    coords = centroid.getInfo()["coordinates"]
    lon, lat = coords[0], coords[1]

    timestamp_ms = img.get("system:time_start").getInfo()
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    # 2. Read metadata properties (these are scalar per image)
    solar_zenith_img = img.get("MEAN_SOLAR_ZENITH_ANGLE").getInfo()
    solar_azimuth_img = img.get("MEAN_SOLAR_AZIMUTH_ANGLE").getInfo()

    # 3. Estimate sun angles with pysolar
    solar_altitude = get_altitude(lat, lon, dt)
    solar_zenith_py = 90.0 - solar_altitude
    solar_azimuth_py = get_azimuth(lat, lon, dt)

    # 4. Print results
    print(f"\n🛰️  Sentinel-2 Image ID: {img.id().getInfo()}")
    print(f"🕒  Acquisition time (UTC): {dt.isoformat()}")
    print(f"📍  Location (lat, lon): ({lat:.4f}, {lon:.4f})\n")

    print("📡 Satellite viewing angles (assumed):")
    print(f"   - Zenith:  0.00°  (nadir)")
    print(f"   - Azimuth: unknown\n")

    print("☀️  Solar angles (from metadata):")
    print(f"   - Zenith:  {solar_zenith_img:.2f}°")
    print(f"   - Azimuth: {solar_azimuth_img:.2f}°\n")

    print("🔆 Solar angles (computed by Pysolar):")
    print(f"   - Zenith:  {solar_zenith_py:.2f}°")
    print(f"   - Azimuth: {solar_azimuth_py:.2f}°")


# 1. Cloud and Cloud Shadow Masking
# Uses QA60 band bits (10: clouds, 11: cirrus). Source: Google Earth Engine Sentinel-2 QA documentation citehttps://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands
def mask_clouds_s2(image):
    qa = image.select("QA60")
    # Create mask where both cloud bits are zero (no cloud, no cirrus)
    cloud_free = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_free)


def add_cloud_probability(image):
    cloud_prob = ee.Image(
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.eq("system:index", image.get("system:index")))
        .first()
    )
    return image.addBands(cloud_prob.rename("cloud_probability"))


# 2. Water Masking
# NDWI: McFeeters (1996), 'The use of the Normalized Difference Water Index (NDWI)' citeMcFeeters1996
def mask_water_s2(image):
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    water_mask = ndwi.gt(0)
    return image.updateMask(water_mask)


# 6. Spatial Consistency Filter (Median smoothing)
# Source: Gonzalez & Woods (2002), 'Digital Image Processing', focal_median in GEE citeGEE_API
def smooth_glint_score(glint_score, radius=1):
    return glint_score.focal_median(radius=radius, units="pixels")


def mask_land(image):
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


def add_b8a_b3_sgi(image):
    img = mask_clouds_s2(image)
    img = mask_water_s2(img)
    nir = img.select("B8A")
    green = img.select("B3")
    gi = nir.subtract(green).divide(nir.add(green)).rename("GlintIndex")
    return img.addBands(gi)


def add_abascal_sun_glint_bands(image):
    img = mask_clouds_s2(image)
    img = mask_water_s2(img)

    swir2 = img.select("B12")
    # 10th percentile SWIR2 for offset
    p10 = swir2.reduceRegion(
        reducer=ee.Reducer.percentile([10]),
        geometry=img.geometry(),
        scale=20,
        bestEffort=True,
    ).get("B12")
    swir_min = ee.Image.constant(p10)
    # Raw glint score: percent increase relative to minimum SWIR2 (Abascal-Zorrilla et al. 2019, Eq. 9)
    # glint_score = ((SWIR2 - SWIR2_min) / SWIR2_min) * 100
    # See Abascal-Zorrilla, N., et al. (2019). "Automated SWIR based empirical sun glint correction of Landsat 8-OLI data..."
    # Compute glint_score
    glint = (
        swir2.subtract(swir_min)
        .divide(swir_min)
        .multiply(100)
        .rename("abascal_glint_score")
    )
    glint_smooth = smooth_glint_score(glint)
    return img.addBands([glint, glint_smooth])
