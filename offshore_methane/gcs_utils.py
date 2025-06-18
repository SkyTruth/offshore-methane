# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: methane
#     language: python
#     name: python3
# ---

# %%
import google.auth
import ee
from google.cloud import storage

ee.Authenticate()
ee.Initialize()

# %%
# Could also consider passing an ee.Image in as input so we can map the function in the EE env.
def download_xml(index):
    # Credentials for writing to 
    credentials, std_proj = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])

    # Load in image of interest.
    image = ee.Image(f'COPERNICUS/S2_HARMONIZED/{index}')

    tile_ids = index.split('_')[2]

    # These will be used to get our product of interest.
    tile_num = tile_ids[1:3]
    lat_band = tile_ids[3]
    grid_square = tile_ids[4:]
    product_uri = image.get('PRODUCT_ID').getInfo()
    granule_id = image.get('GRANULE_ID').getInfo()

    # Name of xml product to query, which holds metadata for solar angles.
    # ex: gcp-public-data-sentinel-2/tiles/15/R/XL/S2B_MSIL1C_20170705T164319_N0205_R126_T15RXL_20170705T165225.SAFE/GRANULE/L1C_T15RXL_A001725_20170705T165225/MTD_TL.xml
    file_id = f"tiles/{tile_num}/{lat_band}/{grid_square}/{product_uri}.SAFE/GRANULE/{granule_id}/MTD_TL.xml"


    client = storage.Client()
    s2_bucket = client.get_bucket("gcp-public-data-sentinel-2")
    blob = s2_bucket.blob(file_id)
    
    
    return blob.download_as_bytes()
