import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def block_live_http(monkeypatch):
    """Every test must supply a service mock; accidental HTTP always fails."""

    def blocked(*args, **kwargs):
        raise AssertionError("Tests are offline; mock the HTTP service")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
