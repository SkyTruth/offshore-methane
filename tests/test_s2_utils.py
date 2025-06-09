from unittest import mock

from offshore_methane import s2_utils


class DummyImage:
    def __init__(self, img_id):
        self.id = img_id


def test_load_scene_id():
    identifier = "20191001T101031_20191001T101659_T31SGR"
    with mock.patch("offshore_methane.s2_utils.ee.Image", DummyImage):
        img = s2_utils.load_s2_image(identifier)
    assert isinstance(img, DummyImage)
    assert img.id == (
        "COPERNICUS/S2_SR_HARMONIZED/20191001T101031_20191001T101659_T31SGR"
    )


def test_load_granule_id():
    identifier = "S2A_MSIL2A_20191001T101031_N0208_R022_T31SGR_20191001T121008"
    with mock.patch("offshore_methane.s2_utils.ee.Image", DummyImage):
        img = s2_utils.load_s2_image(identifier)
    assert img.id == (
        "COPERNICUS/S2_SR_HARMONIZED/20191001T101031_20191001T121008_T31SGR"
    )

