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


def add_cloud_probability(image):
    cloud_prob = ee.Image(
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.eq("system:index", image.get("system:index")))
        .first()
    )
    return image.addBands(cloud_prob.rename("cloud_probability"))


def mask_clouds(image, prob_threshold=100, buffer_meters=None):
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


def add_sgi(image):
    image = mask_clouds(image)
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
    image = mask_clouds(image)
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
