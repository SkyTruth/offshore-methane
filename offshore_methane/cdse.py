# cdse.py

import os
import re
from pathlib import Path

import dotenv
import numpy as np
import rasterio
import requests
from lxml import etree
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------
#  Copernicus Data Space (CDSE) helpers
# ---------------------------------------------------------------------
#  credentials
dotenv.load_dotenv("../.env")
user = os.getenv("CDSE_USERNAME")
pw = os.getenv("CDSE_PASSWORD")
if not (user and pw):
    raise SystemExit("Need CDSE_USERNAME / CDSE_PASSWORD")

_LOGIN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"
_DL_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"


def cdse_token(user: str, pw: str, client="cdse-public") -> str:
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


# ----- CDSE session -----
session = requests.Session()
session.headers["Authorization"] = f"Bearer {cdse_token(user, pw)}"


def cdse_uuid(product_id: str, *, debug: bool = False) -> str | None:
    """
    Return the UUID (Id) of the *latest-baseline* product that matches the
    datatake-start + tile of `product_id`.

    Works for both L1C and L2A; returns None when nothing is found.

    Parameters
    ----------
    product_id : str
        Full ESA product name coming from Earth Engine, e.g.
        ``S2B_MSIL1C_20170705T164319_N0205_R126_T15RXL_20170705T165225``.
    debug : bool, default False
        When True, the function prints the query URL and server response size
        whenever no product is returned.
    """
    # -----------------------------------------------------------------
    # break the ID down once ─ we only need the sensing-start & tilename
    # -----------------------------------------------------------------
    parts = _parse_pid(product_id)  # <-- your helper, already imported
    # datatake start (YYYYMMDDTHHMMSS) and tile (without leading "T")
    core_ts = parts["start"]  # 20170705T164319
    tile = parts["tile"]  # 15RXL
    levelstr = f"MSIL{parts['lvl']}C"  # MSIL1C  or  MSIL2A

    # -----------------------------------------------------------------
    # OData query:
    #   • select only Id+Name (lighter payload)
    #   • contains() on the unique temporal stamp  AND  tile code
    #     (baseline-agnostic)
    #   • results come back newest-first ⇒ grab the first one
    # -----------------------------------------------------------------
    url = (
        f"{_CATALOG}/Products"
        f"?$select=Id,Name"
        f"&$top=1"
        f"&$orderby=ContentDate/Start desc"
        f"&$filter="
        f"contains(Name,'{core_ts}') and "
        f"contains(Name,'T{tile}') and "
        f"contains(Name,'{levelstr}')"
    )

    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        items = r.json().get("value", [])
        if items:
            return items[0]["Id"]  # UUID in CDSE
        if debug:  # <-- only prints when we failed
            print(
                f"[cdse_uuid] no hit → {url}\n            response: {len(r.content)} B"
            )
        return None
    except requests.RequestException as exc:
        if debug:
            print(f"[cdse_uuid] HTTP error {exc}")
        return None


def _parse_pid(pid: str) -> dict:
    _PRODUCT_RE = re.compile(
        r"^(S2[AB])_MSIL(?P<lvl>[12])[AC]_"  # platform & level
        r"(?P<start>\d{8}T\d{6})_"  # datatake start
        r"N\d{4}_R\d{3}_T(?P<tile>\d{2}[A-Z]{3})_"  # tile code
        r"(?P<proc>\d{8}T\d{6})$"
    )  # processing time
    m = _PRODUCT_RE.match(pid)
    if not m:
        raise ValueError(f"Unrecognised Sentinel-2 Product ID: {pid}")
    d = m.groupdict()
    d["sc"] = "Sentinel-2A" if m.group(1) == "S2A" else "Sentinel-2B"
    d["date"] = d["start"][:8]  # YYYYMMDD
    return d


def list_nodes(uri: str) -> list[dict]:
    """
    Return child nodes for an OData *Nodes* URI.

    Accepts all three response shapes seen in Copernicus services:
      • {"value": [...]}
      • {"result": [...]}
      • {"d": {"results": [...]}}
    """
    r = session.get(uri, timeout=30)
    r.raise_for_status()
    js = r.json()

    if "value" in js:  # OData v4 (current CDSE catalogue)
        return js["value"]
    if "result" in js:  # Download service / CNM
        return js["result"]
    if "d" in js and "results" in js["d"]:  # Legacy SciHub
        return js["d"]["results"]

    # "--- unexpected structure ---"
    print(f"[list_nodes] Unrecognised JSON keys at {uri}: {list(js.keys())}")
    return []


def download_object(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with session.get(url, stream=True, timeout=120) as r, open(tmp_path, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            f.write(chunk)
    tmp_path.rename(out_path)


def download_xml(prod_name: str, xml_path: Path):
    if xml_path.is_file():
        return
    try:
        granule_uri = get_granule(prod_name)
        xml_url = f"{granule_uri}/Nodes(MTD_TL.xml)/$value"
        download_object(xml_url, xml_path)
    except requests.HTTPError as e:
        print(f"⚠ {prod_name} XML {e.response.status_code}")


def get_granule(prod_name: str):
    uuid = cdse_uuid(prod_name)
    if uuid is None:
        print(f"⚠ No UUID for {prod_name}")
        return
    root_uri = f"{_DL_BASE}({uuid})/Nodes"

    # print(f"root_uri: {root_uri}")
    # print(f"list_nodes: {list_nodes(root_uri)}")
    safe_dir = next(n for n in list_nodes(root_uri) if n["Name"].endswith(".SAFE"))
    granule_dir = next(
        n for n in list_nodes(safe_dir["Nodes"]["uri"]) if n["Name"] == "GRANULE"
    )

    for g in list_nodes(granule_dir["Nodes"]["uri"]):
        granule_uri = g["Nodes"]["uri"].removesuffix("/Nodes")
        # XXX WARNING, how long is this list, is it ok if we only take the last one?
        # print(f"granule_uri: {granule_uri}")
    return granule_uri


def download_bands(prod_name: str, bands: list[str], out_root: Path):
    granule_uri = get_granule(prod_name)
    try:
        for band in bands:
            out_path = band_path(out_root, prod_name, band)
            if not out_path.is_file():
                url = band_url(granule_uri, band)
                download_object(url, out_path)
    except Exception as exc:
        print(f"⚠ {prod_name} band download fail :: {exc}")


def band_path(out_root: Path, prod_name: str, band: str) -> Path:
    return out_root / "bands" / f"{prod_name}_{band}.jp2"


# ---------------------------------------------------------------------
#  Band-download helpers (R20 m)  – works for both L1C and L2A
# ---------------------------------------------------------------------
def band_url(granule_uri: str, band_code: str) -> str:
    """
    Return the $value URL for B03, B11 or B12 at 20 m resolution.

    ─ Level-2A files: ..._<band>_20m.jp2
    ─ Level-1C files: ..._<band>.jp2          (no "_20m" suffix)

    The helper tries both directory layouts (L2A → L1C) and both
    filename variants (with & without "_20m") until it finds a match.
    """
    # candidate file suffixes, ordered: L2A first, then L1C fallback
    suffixes = [f"{band_code}_20m.jp2", f"{band_code}.jp2"]

    # candidate node folders, ordered: L2A hierarchy first, then L1C
    folders = [
        f"{granule_uri}/Nodes(IMG_DATA)/Nodes(R20m)/Nodes",  # L2A
        f"{granule_uri}/Nodes(IMG_DATA)/Nodes",  # L1C
    ]

    for base in folders:
        # some folders (e.g. R20m on L1C) simply don't exist → 404
        try:
            nodes = list_nodes(base)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                continue
            raise  # bubble up real errors

        for sfx in suffixes:
            for n in nodes:
                if n["Name"].endswith(sfx):
                    return f"{base}({n['Name']})/$value"

    raise FileNotFoundError(band_code)


# ---------------------------------------------------------------------
#  SGA computation (identical to original, kept for fallback)
# ---------------------------------------------------------------------
# (SGA math)
def _grid2array(values_list):
    rows = [np.fromstring(row, sep=" ") for row in values_list]
    return np.vstack(rows).astype(np.float32)


def _mean_detector_grid(root, band_id: str, xpath_suffix: str):
    expr = (
        f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']/{xpath_suffix}"
        "/Values_List/VALUES/text()"
    )
    blocks = root.xpath(expr)
    arrays = [
        _grid2array(blocks[i * 23 : (i + 1) * 23]) for i in range(len(blocks) // 23)
    ]
    return np.nanmean(arrays, axis=0)


def compute_sga_full(prod_name: str, sga_path: Path) -> np.ndarray:
    xml_path = sga_path.parent / f"{prod_name}.xml"
    download_xml(prod_name, xml_path)
    root = etree.parse(str(xml_path))
    sza = _grid2array(root.xpath(".//Sun_Angles_Grid/Zenith/Values_List/VALUES/text()"))
    saa = _grid2array(
        root.xpath(".//Sun_Angles_Grid/Azimuth/Values_List/VALUES/text()")
    )
    vza = _mean_detector_grid(root, "12", "Zenith")
    vaa = _mean_detector_grid(root, "12", "Azimuth")

    delta_phi = np.deg2rad(np.abs(saa - vaa))
    sga_grid = np.rad2deg(
        np.arccos(
            np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza))
            - np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(delta_phi)
        )
    ).astype(np.float32)

    grid_coord = np.arange(23, dtype=np.float32) * 5000.0
    interp = RegularGridInterpolator(
        (grid_coord, grid_coord),
        sga_grid,
        bounds_error=False,
        fill_value=np.nan,
    )

    px_coord = np.arange(10980, dtype=np.float32) * 10.0
    yy, xx = np.meshgrid(px_coord, px_coord)

    sga_full = interp(np.dstack([yy, xx]))
    save_sga_geotiff(sga_full, xml_path, sga_path)
    return sga_full


def save_sga_geotiff(sga: np.ndarray, xml_path: Path, tif_path: Path):
    root = etree.parse(str(xml_path))
    ulx = float(root.findtext(".//Geoposition/ULX"))
    uly = float(root.findtext(".//Geoposition/ULY"))
    tr = from_origin(ulx, uly, 10.0, 10.0)
    crs = root.findtext(".//Tile_Geocoding/HORIZONTAL_CS_CODE")
    profile = {
        "driver": "GTiff",
        "width": sga.shape[1],
        "height": sga.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": tr,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 2,
    }
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(sga, 1)
