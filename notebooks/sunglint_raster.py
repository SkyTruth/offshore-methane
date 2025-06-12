import ee

ee.Initialize()

from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import pandas as pd


def expanded_into_mosaic(ref_img, band_operation_fn=None, band_name=None):
    ref_time = ee.Date(ref_img.get("system:time_start"))

    # Step 1: Filter S2_SR images within ±1 minute
    collection = ee.ImageCollection("COPERNICUS/S2_SR").filterDate(
        ref_time.advance(-1, "minute"), ref_time.advance(1, "minute")
    )

    # Step 2: Optionally apply a band operation to each image
    if band_operation_fn is not None:
        collection = collection.map(band_operation_fn)

    # Step 3: Mosaic the images
    mosaic = collection.mosaic()

    # Step 4: If a specific band is requested, return just that band
    if band_name is not None:
        return mosaic.select(band_name)

    # Otherwise return the full mosaic
    return mosaic


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
# def mask_clouds_s2(image):
#     qa = image.select("QA60")
#     # Create mask where both cloud bits are zero (no cloud, no cirrus)
#     cloud_free = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
#     return image.updateMask(cloud_free)


def add_cloud_probability(image):
    cloud_prob = ee.Image(
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.eq("system:index", image.get("system:index")))
        .first()
    )
    return image.addBands(cloud_prob.rename("cloud_probability"))


# # 2. Water Masking
# # NDWI: McFeeters (1996), 'The use of the Normalized Difference Water Index (NDWI)' citeMcFeeters1996
# def mask_water_s2(image):
#     ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
#     water_mask = ndwi.gt(0)
#     return image.updateMask(water_mask)


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


def mask_clouds(image, prob_threshold=50, buffer_meters=None):
    cloud_prob = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.eq("system:index", image.get("system:index")))
        .first()
        .select("probability")
    )

    # Create cloud mask: True where cloud probability is below threshold
    cloud_mask = cloud_prob.lt(prob_threshold)

    # If buffering is requested, expand the cloudy areas (i.e., shrink the clear-sky mask)
    if buffer_meters:
        cloud_mask = cloud_mask.focal_max(radius=buffer_meters, units="meters")

    return image.updateMask(cloud_mask)


# ChatGPT et al.
def add_sgi_b8a_b3(image):
    img = image
    # img = mask_clouds(image, buffer_meters=None)
    img = mask_land(img)
    # img = image
    nir = img.select("B8A")
    green = img.select("B3")
    gi = nir.subtract(green).divide(nir.add(green)).rename("sgi_b8a_b3")
    return img.addBands(gi)


def add_sgi_b12_b3(image):
    img = image
    # img = mask_clouds(image, buffer_meters=None)
    img = mask_land(img)

    swir = img.select("B12")  # SWIR band (~2200 nm)
    green = img.select("B3")  # Green band (~560 nm)

    # Zhang et al. (2022) SGI: (SWIR - Green) / (SWIR + Green)
    sgi = swir.subtract(green).divide(swir.add(green)).rename("sgi_b12_b3")

    return img.addBands(sgi)


def add_mean_sgi_metadata(image, aoi):
    # Select reflectance bands (scale if needed)
    b12 = image.select("B12").divide(10000)  # SWIR
    b3 = image.select("B3").divide(10000)  # Green

    # Compute SGI = (B12 - B3) / (B12 + B3)
    sgi = b12.subtract(b3).divide(b12.add(b3)).rename("sgi_b12_b3")

    # Mask SGI values outside valid range (e.g., from divide-by-zero or noisy areas)
    sgi = sgi.updateMask(sgi.gte(-1).And(sgi.lte(1)))

    # Reduce SGI to mean over the AOI
    mean_sgi = sgi.reduceRegion(
        reducer=ee.Reducer.mean(),
        scale=20,  # match native B12 resolution
        bestEffort=True,
    ).get("sgi_b12_b3")

    # Return image with SGI band and metadata attached
    return image.addBands(sgi).set("SGI_mean", mean_sgi)


def add_abascal_sun_glint_bands(image):
    img = mask_clouds(image)
    img = mask_land(img)
    # img = image

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


# def plot_mosaic_scatter(
#     band_x_img, band_y_img, roi, scale=10, max_points=5000, bx_name="x", by_name="y"
# ):
#     # Combine bands into a single image
#     combined = band_x_img.rename("x").addBands(band_y_img.rename("y"))

#     # Sample pixel values in the ROI
#     samples = combined.sample(
#         region=roi, scale=scale, numPixels=max_points, geometries=False
#     )

#     # Bring the sample to the client
#     features = samples.getInfo()["features"]

#     # Extract values
#     x_vals = [f["properties"]["x"] for f in features]
#     y_vals = [f["properties"]["y"] for f in features]

#     # Plot
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(6, 6))
#     plt.scatter(x_vals, y_vals, s=1, alpha=0.5)
#     plt.xlabel(bx_name)
#     plt.ylabel(by_name)
#     plt.title(f"Scatterplot of {bx_name} vs {by_name}")
#     plt.grid(True)
#     plt.show()


def mosaic_scatter(band_x_img, band_y_img, roi, scale=10, max_points=5000):
    # Combine the two bands into one image
    combined = band_x_img.rename("x").addBands(band_y_img.rename("y"))

    # Sample the image in the ROI
    samples = combined.sample(
        region=roi, scale=scale, numPixels=max_points, geometries=False
    )

    # Fetch results to client
    features = samples.getInfo()["features"]

    # Parse into DataFrame
    df = pd.DataFrame(
        [
            {"x": f["properties"]["x"], "y": f["properties"]["y"]}
            for f in features
            if "x" in f["properties"] and "y" in f["properties"]
        ]
    )

    return df


def plot_mosaic_scatter(dfs, colors=None, labels=None, alpha=0.5, size=5):
    plt.figure(figsize=(6, 6))

    # Handle defaults
    if colors is None:
        colors = ["blue", "red", "green", "orange", "purple"]
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(dfs))]

    # Plot each dataset
    for i, df in enumerate(dfs):
        plt.scatter(
            df["x"],
            df["y"],
            s=size,
            alpha=alpha,
            color=colors[i % len(colors)],
            label=labels[i] if labels else None,
        )

    plt.xlabel("Band X")
    plt.ylabel("Band Y")
    plt.title("Mosaic Band Scatterplot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
