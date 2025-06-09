from offshore_methane.utils import sentinel2_geemap
import ee

ee.Authenticate()
ee.Initialize()

sentinel2_geemap('20230611T162839_20230611T164034_T16RBT')