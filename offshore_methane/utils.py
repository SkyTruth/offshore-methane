# Imports
import ee
import math
import geemap
import datetime
import rasterio 
import google.auth
import numpy as np
from lxml import etree
from google.cloud import storage
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

credentials, std_proj = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])

ee.Authenticate()
ee.Initialize(project = 'skytruth-tech')

# Pass in a Sentinel-2 L1C (TOA) EE Scene ID, Product ID, or Granule ID.
# Returns the image on a geemap.
def sentinel2_geemap(id):
    
    # Cases where the PRODUCT_ID is provided.
    # Ex: S2B_MSIL1C_20240101T000749_N0510_R130_T50CND_20240101T004831 
    if id.startswith(('S2A','S2B')):

        imDate = id.split('_')[2].split('T')[0]

        imDate_form = datetime.datetime(int(imDate[0:4]),int(imDate[4:6]),int(imDate[6:]))

        orig_date = ee.Date(imDate_form)

        s2Image = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                .filterDate(orig_date,orig_date.advance(1,'day'))
                .filter(ee.Filter.eq('PRODUCT_ID',id)))

        map = geemap.Map()
        map.addLayer(s2Image,{'bands':['B4','B3','B2'],'min':0,'max':3000},id)
        map.centerObject(s2Image,12)

    # Cases where the GRANULE_ID is provided.
    # Ex: L1C_T50CND_A035620_20240101T000751
    elif id.startswith('L1C'):
            
        imDate = id.split('_')[3].split('T')[0]
        
        imDate_form = datetime.datetime(int(imDate[0:4]),int(imDate[4:6]),int(imDate[6:]))

        orig_date = ee.Date(imDate_form)

        s2Image = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                .filterDate(orig_date,orig_date.advance(1,'day'))
                .filter(ee.Filter.eq('GRANULE_ID',id)))
        
        map = geemap.Map()
        map.addLayer(s2Image,{'bands':['B4','B3','B2'],'min':0,'max':3000},id)
        map.centerObject(s2Image,12)
    
    # This will capture the scene IDs, in theory.
    # 20230611T162839_20230611T164034_T16RBT
    else:
        
        imDate = id.split('_')[0].split('T')[0]
        
        imDate_form = datetime.datetime(int(imDate[0:4]),int(imDate[4:6]),int(imDate[6:]))

        orig_date = ee.Date(imDate_form)

        s2Image = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                .filterDate(orig_date,orig_date.advance(1,'day'))
                .filter(ee.Filter.eq('system:index',id)))
    
        map = geemap.Map()
        map.addLayer(s2Image,{'bands':['B4','B3','B2'],'min':0,'max':3000},id)
        map.centerObject(s2Image,12)
        

    if s2Image.size().getInfo() == 0:
        return 'Image not found in Sentinel-2 TOA repo.'
    else:
        return map
    
# function for calculation of alpha metadata property
# Input expects a Sentinel-2 image.
def calculateSunglint_alpha(image):

    # All values are converted to radians
    theta_naught = ee.Number(image.get('MEAN_SOLAR_ZENITH_ANGLE')).multiply(math.pi).divide(180)
    phi_naught = ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')).multiply(math.pi).divide(180)
    
    # I'm going to take the avg of the viewing angle zenith / azimuth btwn B11 and B12.
    b11Zenith = ee.Number(image.get('MEAN_INCIDENCE_ZENITH_ANGLE_B11')).multiply(math.pi).divide(180)
    b11Azimuth = ee.Number(image.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B11')).multiply(math.pi).divide(180)
    b12Zenith = ee.Number(image.get('MEAN_INCIDENCE_ZENITH_ANGLE_B12')).multiply(math.pi).divide(180)
    b12Azimuth = ee.Number(image.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B12')).multiply(math.pi).divide(180)

    theta = (b11Zenith.add(b12Zenith)).divide(2)
    phi = (b11Azimuth.add(b12Azimuth)).divide(2)

    # A = cos(θo + θ) + cos(θo − θ)
    a = (theta_naught.add(theta).cos().add(theta_naught.subtract(theta).cos()))

    # B = cos(θo + θ) − cos(θo − θ)
    b = (theta_naught.add(theta).cos().subtract(theta_naught.subtract(theta).cos()))

    c = phi_naught.subtract(phi).cos()

    arg = (a.add((b.multiply(c)))).divide(2)

    # glint_intensity = arccos(argument)
    alpha = arg.acos()
    
    # return image with new metadata value "glint_intensity"
    return image.set('glint_alpha',alpha.multiply(180/math.pi))

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

# Creates a glint map that gets output to GCS.
# index expects a system:index.
# Ex: 20230505T163839_20230505T165653_T15QWB.

# Could also consider passing an ee.Image in as input so we can map the function in the EE env.
def create_sunglint_map_s2(index):

    # Load in image of interest.
    image = ee.Image(f'COPERNICUS/S2_HARMONIZED/{index}')

    # Pull out original name for final file naming.
    orig_name = image.get('PRODUCT_ID').getInfo()

    # Find SR equivalent image.
    sr_img = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(image.geometry())
             .filter(ee.Filter.eq('system:index',index)).first())

    # Retireving information from the equivalent SR image.
    sr_index = sr_img.get('system:index').getInfo()
    projection_info = sr_img.select('B12').projection().getInfo()
    transform = projection_info['transform']
    crs = projection_info['crs']
    transform[0] = 5000
    transform[4] = -5000

    tile_ids = sr_index.split('_')[2]

    # These will be used to get our product of interest.
    tile_num = tile_ids[1:3]
    lat_band = tile_ids[3]
    grid_square = tile_ids[4:]
    product_uri = sr_img.get('PRODUCT_ID').getInfo()+'.SAFE'
    granule_id = sr_img.get('GRANULE_ID').getInfo()

    # Name of xml product to query, which holds metadata for solar angles.
    # ex: gs://gcp-public-data-sentinel-2/L2/tiles/01/C/CV/S2B_MSIL2A_20181213T210519_N0211_R071_T01CCV_20181213T221546.SAFE/GRANULE/L2A_T01CCV_A009249_20181213T210519/MTD_TL.xml
    file_id = f"L2/tiles/{tile_num}/{lat_band}/{grid_square}/{product_uri}/GRANULE/{granule_id}/MTD_TL.xml"


    client = storage.Client()
    s2_bucket = client.get_bucket("gcp-public-data-sentinel-2")
    blob = s2_bucket.blob(file_id)
    glint_bytes = blob.download_as_bytes()
    
    # create a XML parser to parse the metadata and retrieve its root
    parser = etree.XMLParser(no_network=True, remove_blank_text=True)
    root = etree.fromstring(glint_bytes, parser=parser)
    
    # Retrieve our angle rasters from the metadata.
    sun_zenith = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Zenith')
    sun_azimuth = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Azimuth')

    view_zenith = get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Zenith')
    view_azimuth = get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Azimuth')

    
    # convert angles arrays to radians
    theta_naught = np.deg2rad(sun_zenith)
    phi_naught = np.deg2rad(sun_azimuth)

    theta = np.deg2rad(view_zenith)
    phi = np.deg2rad(view_azimuth)

    # Step-by-step computation
    a = np.cos(theta_naught + theta) + np.cos(theta_naught - theta)
    b = np.cos(theta_naught + theta) - np.cos(theta_naught - theta)
    c = np.cos(phi_naught - phi)

    arg = (a + b * c) / 2.0
    alpha = np.arccos(arg)  # Resulting glint angle

    # convert results to degrees
    glint_array = np.degrees(alpha)
    glint_array[np.isnan(glint_array)] = -1

    # Prepare out_bucket for receiving COG file.
    image_bucket = client.bucket('offshore-methane')
    # Name for GCS object.
    source_file_name = f'glint_{orig_name}_cog.tif'

    with rasterio.open(
        source_file_name, 'w',
        driver='GTiff',
        height=glint_array.shape[0],
        width=glint_array.shape[1],
        count=1,
        dtype=glint_array.dtype,
        crs=crs,
        transform=transform
    ) as dst:
       dst.write(glint_array,1)
       cog_translate(
                dst,
                source_file_name,
                cog_profiles.get('deflate'),
                in_memory=False
            )
                

    blob = image_bucket.blob(source_file_name)
    blob.upload_from_filename(source_file_name)

    # Load the image into Google Earth Engine
    gcs_image_url = f"gs://offshore-methane/{source_file_name}"
    
    # Not really sure what would be useful to return, so I figured that I'd return the GCS 
    return gcs_image_url