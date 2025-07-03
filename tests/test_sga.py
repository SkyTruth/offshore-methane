from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

import pytest

from offshore_methane.sga import _is_cog, gcs_stage


def test_is_cog_true():
    with patch('rasterio.open') as mock_open:
        mock_open.return_value.__enter__.return_value.is_tiled = True
        assert _is_cog(Path('dummy.tif')) is True


def test_is_cog_error():
    with patch('rasterio.open', side_effect=Exception):
        assert _is_cog(Path('dummy.tif')) is False


def test_gcs_stage_success(tmp_path):
    f = tmp_path / 'test.txt'
    f.write_text('data')
    with (
        patch('offshore_methane.sga.gsutil_cmd', return_value='gsutil'),
        patch('subprocess.run') as run,
    ):
        run.return_value = MagicMock(returncode=0)
        url = gcs_stage(f, 'SID', 'SGA', 'bucket')
        run.assert_called_once()
        assert url == f'gs://bucket/SID/{f.name}'


def test_gcs_stage_failure(tmp_path):
    f = tmp_path / 'test.txt'
    f.write_text('data')
    with patch('offshore_methane.sga.gsutil_cmd', return_value='gsutil'), \
         patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd', stderr=b'err')):
        with pytest.raises(RuntimeError):
            gcs_stage(f, 'SID', 'SGA', 'bucket')


def test_gcs_stage_missing(tmp_path):
    missing = tmp_path / 'none.txt'
    with pytest.raises(FileNotFoundError):
        gcs_stage(missing, 'SID', 'SGA', 'bucket')


