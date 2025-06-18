# cdse.py
"""
All interaction with the Copernicus Dataspace ecosystem lives here.
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
CDSE_USER = os.getenv("CDSE_USERNAME")
CDSE_PW = os.getenv("CDSE_PASSWORD")
if not (CDSE_USER and CDSE_PW):
    raise SystemExit("Need CDSE_USERNAME / CDSE_PASSWORD in .env")


# ------------------------------------------------------------------
#  Session with bearer token
# ------------------------------------------------------------------
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


_session = requests.Session()
_session.headers["Authorization"] = f"Bearer {_cdse_token(CDSE_USER, CDSE_PW)}"

# ------------------------------------------------------------------
#  Helper regex & tiny wrappers
# ------------------------------------------------------------------
_PRODUCT_RE = re.compile(
    r"^(S2[AB])_MSIL(?P<lvl>[12])[AC]_"
    r"(?P<start>\d{8}T\d{6})_"
    r"N\d{4}_R\d{3}_T(?P<tile>\d{2}[A-Z]{3})_"
    r"(?P<proc>\d{8}T\d{6})$"
)


def parse_pid(pid: str) -> dict:
    m = _PRODUCT_RE.match(pid)
    if not m:
        raise ValueError(f"Unrecognised Sentinel-2 Product ID: {pid}")
    d = m.groupdict()
    d["sc"] = "Sentinel-2A" if m.group(1) == "S2A" else "Sentinel-2B"
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


def _cdse_uuid(product_id: str) -> str | None:
    p = parse_pid(product_id)
    ts = p["start"]
    tile = p["tile"]
    levelstr = f"MSIL{p['lvl']}C"

    url = (
        f"{_CATALOG}/Products?$select=Id,Name&$top=1"
        f"&$orderby=ContentDate/Start desc"
        f"&$filter=contains(Name,'{ts}') and contains(Name,'T{tile}') "
        f"and contains(Name,'{levelstr}')"
    )
    items = _get(url).json().get("value", [])
    return items[0]["Id"] if items else None


def granule_uri(product_id: str) -> str:
    uuid = _cdse_uuid(product_id)
    if uuid is None:
        raise RuntimeError(f"No UUID for {product_id}")
    root_uri = f"{_DL_BASE}({uuid})/Nodes"
    safe_dir = next(n for n in _list_nodes(root_uri) if n["Name"].endswith(".SAFE"))
    granule_dir = next(
        n for n in _list_nodes(safe_dir["Nodes"]["uri"]) if n["Name"] == "GRANULE"
    )
    return next(iter(_list_nodes(granule_dir["Nodes"]["uri"])))["Nodes"][
        "uri"
    ].removesuffix("/Nodes")


def download_object(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with _session.get(url, stream=True, timeout=120) as r, open(tmp, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            f.write(chunk)
    tmp.rename(out_path)


def download_xml(product_id: str, xml_path: Path):
    """
    Download MTD_TL.xml for *product_id* unless it already exists on disk.
    """
    if xml_path.is_file():
        return
    download_object(f"{granule_uri(product_id)}/Nodes(MTD_TL.xml)/$value", xml_path)


# ------------------------------------------------------------------
#  Robust GET helper – transparently refreshes the bearer token
# ------------------------------------------------------------------


def _get(uri: str, **kw) -> requests.Response:
    """
    GET *uri* using the module-wide session.
    If the token has expired (403) we fetch a fresh one and retry once.
    """
    r = _session.get(uri, timeout=30, **kw)
    if r.status_code == 403:  # token probably timed-out (≈10 min TTL)
        _session.headers["Authorization"] = f"Bearer {_cdse_token(CDSE_USER, CDSE_PW)}"
        r = _session.get(uri, timeout=30, **kw)
    r.raise_for_status()
    return r
