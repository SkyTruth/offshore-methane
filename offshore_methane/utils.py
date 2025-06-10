# Imports
import datetime
import ee
import geemap


def parse_sentinel2_id(id: str) -> tuple[str, datetime.datetime]:
    """Return filter key and acquisition date for a Sentinel-2 identifier."""
    if id.startswith(("S2A", "S2B")):
        date_str = id.split("_")[2].split("T")[0]
        key = "PRODUCT_ID"
    elif id.startswith("L1C"):
        date_str = id.split("_")[3].split("T")[0]
        key = "GRANULE_ID"
    else:
        date_str = id.split("_")[0].split("T")[0]
        key = "system:index"

    date_obj = datetime.datetime(
        int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:])
    )
    return key, date_obj


def fetch_sentinel2_image(key: str, id: str, date: datetime.datetime):
    """Fetch an EE image collection filtered by identifier and date."""
    orig_date = ee.Date(date)
    return (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(orig_date, orig_date.advance(1, "day"))
        .filter(ee.Filter.eq(key, id))
    )


def create_sentinel2_map(collection, id: str):
    """Return a ``geemap.Map`` for the provided image collection."""
    m = geemap.Map()
    m.addLayer(collection, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}, id)
    m.centerObject(collection, 12)
    return m

# Pass in a Sentinel-2 L1C (TOA) EE Scene ID, Product ID, or Granule ID.
# Returns the image on a geemap.
def sentinel2_geemap(id):
    """Return a ``geemap.Map`` showing the requested Sentinel-2 scene."""
    key, date = parse_sentinel2_id(id)
    collection = fetch_sentinel2_image(key, id, date)

    if collection.size().getInfo() == 0:
        return "Image not found in Sentinel-2 TOA repo."

    return create_sentinel2_map(collection, id)
