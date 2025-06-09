# Imports
import ee
import geemap
import datetime

ee.Authenticate()
ee.Initialize()

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