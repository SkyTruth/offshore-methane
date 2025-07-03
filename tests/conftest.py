from unittest.mock import patch
import types
import sys

import pytest

@pytest.fixture
def no_sleep():
    """Patch time.sleep to avoid delays during tests."""
    with patch('time.sleep', return_value=None):
        yield


_stub = types.SimpleNamespace(
    Initialize=lambda: None,
    data=types.SimpleNamespace(deleteAsset=lambda *_: None),
    batch=types.SimpleNamespace(Task=type('Task', (), {})),
    FeatureCollection=object,
    Image=object,
    Geometry=object,
    Feature=object,
)
sys.modules.setdefault('ee', _stub)
sys.modules.setdefault('geemap', types.SimpleNamespace(Map=lambda *a, **k: None))
