# %%
#!/usr/bin/env python3
"""
process_s2_sga.py - end-to-end utility to

  1. query Sentinel-2 L2A products in Google Earth Engine for a point & date range
  2. download every GRANULE's MTD_TL.xml from Copernicus Data Space
  3. compute a 10 m sunglint-angle (SGA) raster for each granule
  4. save it as <product>.tif


Environment
-----------
.env file with CDSE_USERNAME and CDSE_PASSWORD

Dependencies
------------
numpy, pandas, lxml, scipy, google-earth-engine (ee), requests, rasterio, dotenv
"""

# ─────────────────────────────────────────────────────────────────────────────
# utils - consider moving to utils.py
# ─────────────────────────────────────────────────────────────────────────────
import concurrent.futures as cf
import os
import time

import dotenv
import ee
import numpy as np
import rasterio
import requests
from lxml import etree
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator


# ------------------- generic remote-sensing helpers -------------------------
def sentinel2_product_ids(point, start, end, cloud_pct):
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(point)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    return sorted(set(coll.aggregate_array("PRODUCT_ID").getInfo()))


# ---------------- CDSE catalogue / download helpers ------------------------
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


def cdse_uuid(prod_name: str, session: requests.Session) -> str | None:
    # exact match, '.SAFE' suffix, finally startswith fallback
    print(prod_name)
    for name in (prod_name, f"{prod_name}.SAFE"):
        url = f"{_CATALOG}/Products?$filter=Name eq '{name}'&$select=Id&$top=1"
        val = session.get(url, timeout=20).json().get("value", [])
        if val:
            return val[0]["Id"]
    url = (
        f"{_CATALOG}/Products?$filter=startswith(Name,'{prod_name}')&$select=Id&$top=1"
    )
    val = session.get(url, timeout=20).json().get("value", [])
    return val[0]["Id"] if val else None


def list_nodes(uri: str, session: requests.Session):
    return session.get(uri, timeout=30).json()["result"]


def download(url: str, out_path: str, session: requests.Session):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with session.get(url, stream=True, timeout=60) as r, open(out_path, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            f.write(chunk)


# ------------------- SGA computation helpers -------------------------------
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


def compute_sga_full(xml_path: str) -> np.ndarray:
    root = etree.parse(xml_path)
    sza = _grid2array(root.xpath(".//Sun_Angles_Grid/Zenith/Values_List/VALUES/text()"))
    saa = _grid2array(
        root.xpath(".//Sun_Angles_Grid/Azimuth/Values_List/VALUES/text()")
    )
    vza = _mean_detector_grid(root, "12", "Zenith")
    vaa = _mean_detector_grid(root, "12", "Azimuth")

    if not (sza.shape == (23, 23) == saa.shape == vza.shape == vaa.shape):
        raise ValueError("Angle grid shape mismatch")

    delta_phi = np.deg2rad(np.abs(saa - vaa))
    sga_grid = np.rad2deg(
        np.arccos(
            np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza))
            - np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(delta_phi)
        )
    ).astype(np.float32)

    grid_coord = np.arange(23, dtype=np.float32) * 5000.0  # m from UL corner
    interp = RegularGridInterpolator(
        (grid_coord, grid_coord),
        sga_grid,
        # method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )

    px_coord = np.arange(10980, dtype=np.float32) * 10.0
    yy, xx = np.meshgrid(px_coord, px_coord)  # yy→rows
    return interp(np.dstack([yy, xx]))  # (10980,10980) float32


def _geotransform_from_xml(root) -> tuple:
    """Return (transform, crs-string) from <Geoposition> & <Tile_Geocoding>."""
    ulx = float(root.findtext(".//Geoposition/ULX"))
    uly = float(root.findtext(".//Geoposition/ULY"))
    dim = float(root.findtext(".//Geoposition/XDIM"))  # 10.0 m for 10 m data
    crs = root.findtext(".//Tile_Geocoding/HORIZONTAL_CS_CODE")  # 'EPSG:32631' etc.
    return from_origin(ulx, uly, dim, dim), crs


def save_sga_geotiff(sga: np.ndarray, xml_path: str, tif_path: str):
    root = etree.parse(xml_path)
    transform, crs = _geotransform_from_xml(root)
    profile = {
        "driver": "GTiff",
        "width": sga.shape[1],
        "height": sga.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        # sensible compression/tile defaults
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 2,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(sga, 1)


# ─────────────────────────────────────────────────────────────────────────────
# main script
# ─────────────────────────────────────────────────────────────────────────────

dotenv.load_dotenv(".env")


def fetch_and_process(prod_name: str, out_root: str, session: requests.Session):
    uuid = cdse_uuid(prod_name, session)
    if uuid is None:
        print(f"⚠ No UUID for {prod_name}")
        return

    root_uri = f"{_DL_BASE}({uuid})/Nodes"
    safe_dir = next(
        n for n in list_nodes(root_uri, session) if n["Name"].endswith(".SAFE")
    )
    granule_dir = next(
        n
        for n in list_nodes(safe_dir["Nodes"]["uri"], session)
        if n["Name"] == "GRANULE"
    )

    for g in list_nodes(granule_dir["Nodes"]["uri"], session):
        granule_uri = g["Nodes"]["uri"].removesuffix("/Nodes")
        xml_url = f"{granule_uri}/Nodes(MTD_TL.xml)/$value"

        xml_path = os.path.join(out_root, f"{prod_name}.xml")
        sga_path = os.path.join(out_root, f"{prod_name}.tif")

        if not os.path.isfile(xml_path):
            try:
                download(xml_url, xml_path, session)
                print(f"↓  {g['Name']} xml")
            except requests.HTTPError as e:
                print(f"⚠ {g['Name']} {e.response.status_code}")
                continue

        # SGA computation (skipped if already exists & newer)
        if os.path.isfile(sga_path) and os.path.getmtime(sga_path) >= os.path.getmtime(
            xml_path
        ):
            continue
        try:
            sga_full = compute_sga_full(xml_path)
            save_sga_geotiff(sga_full, xml_path, sga_path)
            print(f"✔  {g['Name']} SGA → {sga_path}")

        except Exception as exc:
            print(f"⚠ {g['Name']} SGA failed :: {exc}")


# %%


lon, lat = -73.9857, 40.7484
start, end = "2025-06-01", "2025-06-15"
max_cloud = 20
threads = 6
out = "../data/sga"

user = os.getenv("CDSE_USERNAME")
pw = os.getenv("CDSE_PASSWORD")
if not (user and pw):
    raise SystemExit("Need CDSE credentials (flags or env vars).")

# Earth Engine init
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

point = ee.Geometry.Point((lon, lat))
products = sentinel2_product_ids(point, start, end, max_cloud)
if not products:
    raise SystemExit("No matching Sentinel-2 products.")
print(f"Found {len(products)} product(s)")

session = requests.Session()
session.headers["Authorization"] = f"Bearer {cdse_token(user, pw)}"

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=threads) as ex:
    list(ex.map(lambda p: fetch_and_process(p, out, session), products))
print(f"Done in {time.time() - t0:.1f}s")

# %%
