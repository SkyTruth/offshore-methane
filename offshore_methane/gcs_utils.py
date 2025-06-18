import ee
import google.auth
import numpy as np
from google.cloud import storage

ee.Authenticate()
ee.Initialize()

# For use within the sunglint map function below.
def get_grid_values_from_xml(tree_node, xpath_str):
    '''Receives a XML tree node and a XPath parsing string and search for children matching the string.
       Then, extract the VALUES in <values> v1 v2 v3 </values> <values> v4 v5 v6 </values> format as numpy array
       Loop through the arrays to compute the mean.
    '''
    node_list = tree_node.xpath(xpath_str)

    arrays_lst = []
    for node in node_list:
        values_lst = node.xpath('.//VALUES/text()')
        values_arr = np.array(list(map(lambda x: x.split(' '), values_lst))).astype('float')
        arrays_lst.append(values_arr)

    return np.nanmean(arrays_lst, axis=0)

def download_xml(index):
    # Credentials for reading/writing in GCS.
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
