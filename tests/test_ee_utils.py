from unittest.mock import MagicMock, patch

import importlib
import pytest
from requests.exceptions import HTTPError

with patch('offshore_methane.ee_utils.ee.Initialize'):
    ee_utils = importlib.import_module('offshore_methane.ee_utils')

_prepare_asset = ee_utils._prepare_asset
_download_url = ee_utils._download_url


class DummyResp:
    def __init__(self, content=b"data", status=200):
        self.status_code = status
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            err = HTTPError()
            err.response = MagicMock(status_code=self.status_code)
            raise err

    def iter_content(self, chunk):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass



def test_prepare_asset_missing(no_sleep):
    with patch('offshore_methane.ee_utils.ee_asset_exists', return_value=False):
        assert _prepare_asset('id') is True


def test_prepare_asset_ready(no_sleep):
    with patch('offshore_methane.ee_utils.ee_asset_exists', return_value=True), \
         patch('offshore_methane.ee_utils.ee_asset_ready', return_value=True):
        assert _prepare_asset('id') is False


def test_prepare_asset_ingesting_becomes_ready(no_sleep):
    with patch('offshore_methane.ee_utils.ee_asset_exists', return_value=True), \
         patch('offshore_methane.ee_utils.ee_asset_ready', side_effect=[False, True]):
        assert _prepare_asset('id', timeout=1) is False


def test_prepare_asset_overwrite(no_sleep):
    with patch('offshore_methane.ee_utils.ee_asset_exists', side_effect=[True, False]), \
         patch('offshore_methane.ee_utils.ee_asset_ready', return_value=True), \
         patch('offshore_methane.ee_utils.ee.data.deleteAsset') as deleter:
        assert _prepare_asset('id', overwrite=True) is True
        deleter.assert_called_once_with('id')


def test_download_url_retry(tmp_path, no_sleep):
    dest = tmp_path / 'out.bin'
    calls = []

    def mock_get(url, stream=True, timeout=60):
        if not calls:
            calls.append('first')
            err = HTTPError()
            err.response = MagicMock(status_code=503)
            raise err
        return DummyResp(b'abc')

    with patch('offshore_methane.ee_utils.requests.get', side_effect=mock_get):
        _download_url('http://x', dest)

    assert dest.read_bytes() == b'abc'
    assert len(calls) == 1


def test_download_url_fails(tmp_path, no_sleep):
    dest = tmp_path / 'out.bin'

    def mock_get(url, stream=True, timeout=60):
        err = HTTPError()
        err.response = MagicMock(status_code=400)
        raise err

    with patch('offshore_methane.ee_utils.requests.get', side_effect=mock_get):
        with pytest.raises(HTTPError):
            _download_url('http://x', dest, max_retries=1)

