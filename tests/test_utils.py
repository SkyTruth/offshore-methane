from offshore_methane.utils import sentinel2_geemap


def test_sentinel2_geemap_callable():
    """Ensure the sentinel2_geemap function is defined and callable."""
    assert callable(sentinel2_geemap)
