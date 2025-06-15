import datetime
from offshore_methane import utils


def test_parse_sentinel2_id_product():
    key, date = utils.parse_sentinel2_id(
        "S2A_MSIL1C_20240101T000749_N0510_R130_T50CND_20240101T004831"
    )
    assert key == "PRODUCT_ID"
    assert date == datetime.datetime(2024, 1, 1)


def test_parse_sentinel2_id_granule():
    key, date = utils.parse_sentinel2_id("L1C_T50CND_A035620_20240101T000751")
    assert key == "GRANULE_ID"
    assert date == datetime.datetime(2024, 1, 1)


def test_parse_sentinel2_id_scene():
    key, date = utils.parse_sentinel2_id("20230611T162839_20230611T164034_T16RBT")
    assert key == "system:index"
    assert date == datetime.datetime(2023, 6, 11)


def test_sentinel2_geemap_returns_map(monkeypatch):
    class DummySize:
        def getInfo(self):
            return 1

    class DummyCollection:
        def size(self):
            return DummySize()

    def fake_fetch(key, id, date):
        return DummyCollection()

    def fake_create(collection, id):
        return "dummy_map"

    monkeypatch.setattr(utils, "fetch_sentinel2_image", fake_fetch)
    monkeypatch.setattr(utils, "create_sentinel2_map", fake_create)

    result = utils.sentinel2_geemap("20230611T162839_20230611T164034_T16RBT")
    assert result == "dummy_map"


def test_sentinel2_geemap_handles_missing(monkeypatch):
    class DummySize:
        def getInfo(self):
            return 0

    class DummyCollection:
        def size(self):
            return DummySize()

    def fake_fetch(key, id, date):
        return DummyCollection()

    monkeypatch.setattr(utils, "fetch_sentinel2_image", fake_fetch)
    monkeypatch.setattr(utils, "create_sentinel2_map", lambda *args, **kwargs: None)

    result = utils.sentinel2_geemap("20230611T162839_20230611T164034_T16RBT")
    assert result == "Image not found in Sentinel-2 TOA repo."
