# cdse.py
"""
All interaction with the Copernicus Dataspace ecosystem lives here.
The module can now be imported even when CDSE credentials are absent.
Network-touching helpers lazily add / refresh the bearer token when
credentials are available, and raise a clear error otherwise.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

_LOGIN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"
_DL_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"

load_dotenv("../.env")  # CDSE creds
CDSE_USER = os.getenv("CDSE_USERNAME") or ""
CDSE_PW = os.getenv("CDSE_PASSWORD") or ""
_AUTH_AVAILABLE = bool(CDSE_USER and CDSE_PW)

# ------------------------------------------------------------------
#  Lazy session (token added when first needed)
# ------------------------------------------------------------------
_session = requests.Session()


def _cdse_token(user: str, pw: str, client: str = "cdse-public") -> str:
    r = requests.post(
        _LOGIN,
        data={
            "username": user,
            "password": pw,
            "grant_type": "password",
            "client_id": client,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _ensure_token() -> None:
    """Attach a fresh bearer token to the module-wide session if possible."""
    if "Authorization" in _session.headers:
        return  # already have one
    if not _AUTH_AVAILABLE:
        raise RuntimeError(
            "CDSE credentials not configured - set CDSE_USERNAME / CDSE_PASSWORD "
            "in the environment or a .env file to use remote CDSE features."
        )
    _session.headers["Authorization"] = f"Bearer {_cdse_token(CDSE_USER, CDSE_PW)}"


# ------------------------------------------------------------------
#  Helper regex & tiny wrappers
# ------------------------------------------------------------------
_SYSTEM_INDEX_RE = re.compile(
    r"^(?P<start>\d{8}T\d{6})_"  # sensing UTC
    r"(?P<proc>\d{8}T\d{6})_"  # processing UTC
    r"T(?P<tile>\d{2}[A-Z]{3})$"  # MGRS tile
)


def parse_sid(sid: str) -> dict:
    """Parse a Sentinel-2 GEE system:index into its components."""
    m = _SYSTEM_INDEX_RE.match(sid)
    if not m:
        raise ValueError(f"Unrecognised Sentinel-2 system:index: {sid}")
    d = m.groupdict()
    d["date"] = d["start"][:8]
    return d


def _list_nodes(uri: str) -> list[dict]:
    js = _get(uri).json()
    if "value" in js:  # /odata/v1
        return js["value"]
    if "result" in js:  # /Products('uuid')/Nodes
        return js["result"]
    if "d" in js and "results" in js["d"]:  # SAP-ish
        return js["d"]["results"]
    return []


def _cdse_uuid(sid: str, *, prefer_l2a: bool = False):
    """Map a GEE Sentinel-2 system:index to its CDSE UUID."""
    p = parse_sid(sid)
    ts = p["start"]
    tile = p["tile"]

    lvl_order = ("MSIL2A", "MSIL1C") if prefer_l2a else ("MSIL1C", "MSIL2A")
    base = f"{_CATALOG}/Products?$select=Id,Name&$top=1&$orderby=ContentDate/Start desc"

    for lvl in lvl_order:
        url = (
            f"{base}"
            f"&$filter=contains(Name,'{ts}') "
            f"and contains(Name,'T{tile}') "
            f"and contains(Name,'{lvl}')"
        )
        items = _get(url).json().get("value", [])
        if items:
            return items[0]["Id"]

    return None


def granule_uri(sid: str) -> str:
    uuid = _cdse_uuid(sid)
    if uuid is None:
        raise RuntimeError(f"No UUID for {sid}")
    root_uri = f"{_DL_BASE}({uuid})/Nodes"
    safe_dir = next(n for n in _list_nodes(root_uri) if n["Name"].endswith(".SAFE"))
    granule_dir = next(
        n for n in _list_nodes(safe_dir["Nodes"]["uri"]) if n["Name"] == "GRANULE"
    )
    return next(iter(_list_nodes(granule_dir["Nodes"]["uri"])))["Nodes"][
        "uri"
    ].removesuffix("/Nodes")


def download_object(url: str, out_path: Path):
    _ensure_token()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with _get(url, stream=True) as r, open(tmp, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    tmp.rename(out_path)


def download_xml(sid: str, xml_path: Path):
    """Download MTD_TL.xml for *system_index* unless it already exists on disk."""
    if xml_path.is_file():
        return
    download_object(f"{granule_uri(sid)}/Nodes(MTD_TL.xml)/$value", xml_path)


# ------------------------------------------------------------------
#  Robust GET helper - transparently refreshes (or acquires) the token
# ------------------------------------------------------------------


def _get(uri: str, **kw) -> requests.Response:
    """GET *uri* with automatic token handling."""
    try:
        _ensure_token()
    except RuntimeError:
        # No creds: proceed without token - some endpoints are public.
        pass

    r = _session.get(uri, timeout=30, **kw)

    if r.status_code == 403 and _AUTH_AVAILABLE:
        # maybe token expired (~10 min TTL) → refresh once
        _session.headers["Authorization"] = f"Bearer {_cdse_token(CDSE_USER, CDSE_PW)}"
        r = _session.get(uri, timeout=30, **kw)

    r.raise_for_status()
    return r
