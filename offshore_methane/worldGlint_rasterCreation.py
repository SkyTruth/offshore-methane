import ee
import datetime as dt

ee.Authenticate()
ee.Initialize()

# Global simplified ocean polygons (replace with your own asset if needed)
ocean = ee.FeatureCollection("users/christian/General_Data/Marine/simplifiedOceans")


def calculate_sunglint_alpha(image: ee.Image):
    pi = 3.141592653589793

    # --- Solar geometry (radians) ---
    theta_naught = (
        ee.Number(image.get("MEAN_SOLAR_ZENITH_ANGLE")).multiply(pi).divide(180)
    )
    phi_naught = (
        ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE")).multiply(pi).divide(180)
    )

    # --- View geometry for bands B11 & B12 (radians) ---
    b11_zenith = (
        ee.Number(image.get("MEAN_INCIDENCE_ZENITH_ANGLE_B11")).multiply(pi).divide(180)
    )
    b11_azimuth = (
        ee.Number(image.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B11"))
        .multiply(pi)
        .divide(180)
    )
    b12_zenith = (
        ee.Number(image.get("MEAN_INCIDENCE_ZENITH_ANGLE_B12")).multiply(pi).divide(180)
    )
    b12_azimuth = (
        ee.Number(image.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B12"))
        .multiply(pi)
        .divide(180)
    )

    # --- Mean view zenith (θ) and azimuth (φ) ---
    theta = b11_zenith.add(b12_zenith).divide(2)
    phi = b11_azimuth.add(b12_azimuth).divide(2)

    # --- Intermediate terms (Hedley et al.) ---
    A = theta_naught.add(theta).cos().add(theta_naught.subtract(theta).cos())
    B = theta_naught.add(theta).cos().subtract(theta_naught.subtract(theta).cos())
    C = phi_naught.subtract(phi).cos()

    arg = A.add(B.multiply(C)).divide(2)
    alpha = arg.acos()  # radians
    alpha_deg = alpha.multiply(180 / pi)

    # Convert to image, keep original mask, rename band
    alpha_img = (
        ee.Image(alpha_deg)
        .updateMask(image.select("B12").mask())
        .double()
        .rename("glint_alpha")
    )
    return image.addBands(alpha_img)


def generate_sunglint_worldRaster(start_date):
    """
    Create min/max sunglint-alpha rasters for one month starting at `start_date`
    and export them to your Assets with names like 'june2024_sunglintAlpha_min'.

    Parameters
    ----------
    start_date : str | datetime.date | datetime.datetime
        First day of the month you want to process (e.g. '2024-06-01').
    ocean      : ee.Geometry | ee.FeatureCollection
        The AOI you already have for ocean pixels.
    """
    # ---------- 1. normalise `start_date` ----------
    if isinstance(start_date):
        py_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    elif isinstance(start_date, (dt.datetime, dt.date)):
        py_date = (
            start_date
            if isinstance(start_date, dt.datetime)
            else dt.datetime.combine(start_date, dt.time())
        )
    else:
        raise TypeError("start_date must be 'YYYY-MM-DD' or a datetime/date object")

    month_name = py_date.strftime("%B").lower()  # 'june'
    year_str = py_date.strftime("%Y")  # '2024'

    ee_start = ee.Date(start_date)
    ee_end = ee_start.advance(1, "month")

    # ---------- 2. build the collection ----------
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(ee_start, ee_end)
        .filterBounds(ocean)
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 20)
        .filterMetadata("MEAN_INCIDENCE_ZENITH_ANGLE_B11", "greater_than", 0)
        .filterMetadata("MEAN_INCIDENCE_ZENITH_ANGLE_B12", "greater_than", 0)
        .filterMetadata("MEAN_INCIDENCE_AZIMUTH_ANGLE_B11", "greater_than", 0)
        .filterMetadata("MEAN_INCIDENCE_AZIMUTH_ANGLE_B12", "greater_than", 0)
        .filterMetadata("MEAN_SOLAR_ZENITH_ANGLE", "greater_than", 0)
        .filterMetadata("MEAN_SOLAR_AZIMUTH_ANGLE", "greater_than", 0)
        .map(calculate_sunglint_alpha)
    )

    # ---------- 3. min / max rasters ----------
    min_raster = s2.select("glint_alpha").min()
    max_raster = s2.select("glint_alpha").max()

    # ---------- 4. dynamic export names ----------
    base_name_min = f"{month_name}{year_str}_sunglintAlpha_min"
    base_name_max = f"{month_name}{year_str}_sunglintAlpha_max"

    task_min = ee.batch.Export.image.toAsset(
        image=min_raster,
        description=base_name_min,
        assetId=base_name_min,  # or 'users/<username>/' + base_name_min
        scale=1000,
        maxPixels=1e10,
    )
    task_max = ee.batch.Export.image.toAsset(
        image=max_raster,
        description=base_name_max,
        assetId=base_name_max,
        scale=1000,
        maxPixels=1e10,
    )

    # ---------- 5. start the tasks ----------
    task_min.start()
    task_max.start()
    print(f"Export tasks started: {base_name_min} and {base_name_max}")
