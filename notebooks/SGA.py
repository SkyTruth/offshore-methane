# %%
#!/usr/bin/env python3
"""
s2_mbsp_combined.py
~~~~~~~~~~~~~~~~~~~
End-to-end Sentinel-2 MBSP processing that leverages *pre-computed* 10 m
Sun-Glint-Angle (SGA) rasters.

  1.  Query Sentinel-2 L2A products in Google Earth Engine for a point,
      date range and cloud-cover threshold.
  2.  Ensure each granule has an on-disk SGA GeoTIFF
      («../data/sga/<product>.tif»); compute it once otherwise.
  3.  Download exactly B03,B11,B12 (20 m) JP2 files per granule
      – nothing more.
  4.  Perform the MBSP sunglint pipeline *entirely offline* with NumPy.
  5.  Save the fractional MBSP signal as
      «../data/mbsp/<product>_R.tif».

Environment
-----------
.env with CDSE_USERNAME / CDSE_PASSWORD

Dependencies
------------
numpy, rasterio, requests, lxml, scipy, pyproj, dotenv, google-earth-engine
"""

# ---------------------------------------------------------------------
#  Imports & config
# ---------------------------------------------------------------------

# (imports)
import concurrent.futures as cf
import os
import re
import time
from pathlib import Path

import dotenv
import ee
import numpy as np
import rasterio
import requests
from lxml import etree
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------
#  Generic helpers
# ---------------------------------------------------------------------
# (Earth-Engine query)
def sentinel2_product_ids(point, start, end, cloud_pct):
    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(point)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    return sorted(set(coll.aggregate_array("PRODUCT_ID").getInfo()))


# ---------------------------------------------------------------------
#  Copernicus Data Space (CDSE) helpers
# ---------------------------------------------------------------------
# (CDSE auth + catalogue)
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


def cdse_uuid(
    product_id: str, session: requests.Session, *, debug: bool = False
) -> str | None:
    """
    Return the UUID (Id) of the *latest-baseline* product that matches the
    datatake-start + tile of `product_id`.

    Works for both L1C and L2A; returns None when nothing is found.

    Parameters
    ----------
    product_id : str
        Full ESA product name coming from Earth Engine, e.g.
        ``S2B_MSIL1C_20170705T164319_N0205_R126_T15RXL_20170705T165225``.
    session : requests.Session
        An already authenticated session (``Authorization: Bearer …``).
    debug : bool, default False
        When True, the function prints the query URL and server response size
        whenever no product is returned.
    """
    # -----------------------------------------------------------------
    # break the ID down once ─ we only need the sensing-start & tilename
    # -----------------------------------------------------------------
    parts = _parse_pid(product_id)  # <-- your helper, already imported
    # datatake start (YYYYMMDDTHHMMSS) and tile (without leading “T”)
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


def pid_to_asset(pid: str) -> str:
    """
    Map an ESA Sentinel-2 Product ID to a COPERNICUS/S2*_HARMONIZED asset id.

    The function:
      • parses the Product ID once client-side,
      • narrows the server search by date (±3 h), tile and spacecraft,
      • retrieves only the 'system:index' property (37 B) instead of the full
        image manifest (≈200 kB).
    Typical latency: 1–3 s on a warm EE session.
    """
    p = _parse_pid(pid)
    coll = "COPERNICUS/S2_HARMONIZED" if p["lvl"] == "1" else "COPERNICUS/S2_HARMONIZED"
    t0 = ee.Date.parse("yyyyMMdd'T'HHmmss", p["start"])
    idx = (
        ee.ImageCollection(coll)
        # 1. narrow to 3-hour window around the datatake start
        .filterDate(t0.advance(-90, "minute"), t0.advance(90, "minute"))
        # 2. same spacecraft (A or B)
        .filter(ee.Filter.eq("SPACECRAFT_NAME", p["sc"]))
        # 3. same MGRS tile (EE stores it without the leading 'T')
        .filter(ee.Filter.eq("MGRS_TILE", p["tile"]))
        # 4. exact Product ID match (usually hits 1 image now)
        .filter(ee.Filter.eq("PRODUCT_ID", pid))
        # 5. pull just the first (only) system:index string
        .aggregate_first("system:index")
    )
    return f"{coll}/{idx.getInfo()}"


def list_nodes(uri: str, session: requests.Session) -> list[dict]:
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


def download(url: str, out_path: Path, session: requests.Session):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=120) as r, open(out_path, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            f.write(chunk)


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


def compute_sga_full(xml_path: Path) -> np.ndarray:
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
    return interp(np.dstack([yy, xx]))


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


# ---------------------------------------------------------------------
#  Band-download helpers (R20 m)  – works for both L1C and L2A
# ---------------------------------------------------------------------
def band_url(granule_uri: str, band_code: str, session: requests.Session) -> str:
    """
    Return the $value URL for B03, B11 or B12 at 20 m resolution.

    ─ Level-2A files: ..._<band>_20m.jp2
    ─ Level-1C files: ..._<band>.jp2          (no “_20m” suffix)

    The helper tries both directory layouts (L2A → L1C) and both
    filename variants (with & without “_20m”) until it finds a match.
    """
    # candidate file suffixes, ordered: L2A first, then L1C fallback
    suffixes = [f"{band_code}_20m.jp2", f"{band_code}.jp2"]

    # candidate node folders, ordered: L2A hierarchy first, then L1C
    folders = [
        f"{granule_uri}/Nodes(IMG_DATA)/Nodes(R20m)/Nodes",  # L2A
        f"{granule_uri}/Nodes(IMG_DATA)/Nodes",  # L1C
    ]

    for base in folders:
        # some folders (e.g. R20m on L1C) simply don’t exist → 404
        try:
            nodes = list_nodes(base, session)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                continue
            raise  # bubble up real errors

        for sfx in suffixes:
            for n in nodes:
                if n["Name"].endswith(sfx):
                    return f"{base}({n['Name']})/$value"

    raise FileNotFoundError(band_code)


def ensure_jp2(
    granule_uri: str, product: str, band: str, session: requests.Session, outdir: Path
) -> Path:
    out = outdir / f"{product}_{band}.jp2"
    if out.is_file():
        return out
    url = band_url(granule_uri, band, session)
    download(url, out, session)
    return out


# ---------------------------------------------------------------------
#  MBSP numeric pipeline (NumPy)
# ---------------------------------------------------------------------
# (MBSP maths)
def mbsp_numpy(b03, b11, b12, sga, transform, centre_xy, radius=50_000):
    """Return fractional MBSP raster R with glint-mask applied."""

    if sga.shape != b11.shape:  # 10 m (10980) vs 20 m (5490)
        factor = sga.shape[0] // b11.shape[0]  # =2 for Sentinel-2
        sga = sga[::factor, ::factor]  # quick nearest-neighbour down-sample

    # ---------- Step 0: pre-screen ----------
    med_b12 = np.nanmedian(b12)
    if med_b12 <= 0.01:
        return None  # not useful

    # ---------- SGI ----------
    denom = np.maximum(b12 + b03, 1e-4)
    sgi = (b12 - b03) / denom
    sgi = np.clip(sgi, -1, 1)

    # ---------- AOI mask (50 km disc) ----------
    rows, cols = b11.shape
    col_vec = np.arange(cols, dtype=np.float32)
    row_vec = np.arange(rows, dtype=np.float32)
    x_coords = transform[0] + (col_vec + 0.5) * transform[1]
    y_coords = transform[3] + (row_vec + 0.5) * transform[5]
    dx = x_coords - centre_xy[0]
    dy = y_coords - centre_xy[1]
    mask = (dy[:, None] ** 2 + dx[None, :] ** 2) < radius**2

    # ---------- Step 2: de-stripe ----------
    sgim = sgi[mask].ravel()
    b11m = b11[mask].ravel()
    A = np.column_stack((np.ones_like(sgim), sgim))
    a0, a1 = np.linalg.lstsq(A, b11m, rcond=None)[0]
    fit = a0 + a1 * sgi
    b11_flat = b11 - fit

    ratio = b12 / np.where(b11 == 0, np.nan, b11)
    b12_flat = b12 - fit * ratio

    # ---------- Step 3: slope coefficients ----------
    x1 = b12_flat[mask].ravel()
    x2 = (b12_flat * sgi)[mask].ravel()
    X = np.column_stack((x1, x2))
    beta = np.linalg.lstsq(X, b11_flat[mask].ravel(), rcond=None)[0]
    c0, c1 = beta

    c = c0 + c1 * sgi

    # ---------- Step 4: fractional signal ----------
    R = (c * b12_flat - b11_flat) / np.where(b11_flat == 0, np.nan, b11_flat)

    # ---------- Step 5: SGA mask ----------
    good = (sga > 1) & (sga < 15)
    R[~good] = np.nan
    return R.astype(np.float32)


# ---------------------------------------------------------------------
#  Granule driver
# ---------------------------------------------------------------------
# (per-product processing)
def process_granule(
    prod_name: str,
    out_root: Path,
    session: requests.Session,
    centre_lon: float,
    centre_lat: float,
):
    bands_dir = out_root / "bands"
    sga_dir = out_root / "sga"
    mbsp_dir = out_root / "mbsp"
    mbsp_dir.mkdir(parents=True, exist_ok=True)

    uuid = cdse_uuid(prod_name, session)
    if uuid is None:
        print(f"⚠ No UUID for {prod_name}")
        return
    root_uri = f"{_DL_BASE}({uuid})/Nodes"

    print(f"root_uri: {root_uri}")
    print(f"list_nodes: {list_nodes(root_uri, session)}")
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
        tile_id = g["Name"]
        product_tag = f"{prod_name}_{tile_id}"

        # ----- paths -----
        xml_path = sga_dir / f"{prod_name}.xml"
        sga_path = sga_dir / f"{prod_name}.tif"
        out_path = mbsp_dir / f"{product_tag}_R.tif"
        if out_path.is_file():
            continue  # already done

        # ----- XML download (for SGA) -----
        if not xml_path.is_file():
            xml_url = f"{granule_uri}/Nodes(MTD_TL.xml)/$value"
            try:
                download(xml_url, xml_path, session)
            except requests.HTTPError as e:
                print(f"⚠ {g['Name']} XML {e.response.status_code}")
                continue

        # ----- SGA cache / compute -----
        if not sga_path.is_file():
            try:
                sga_full = compute_sga_full(xml_path)
                save_sga_geotiff(sga_full, xml_path, sga_path)
                print(f"✔ {tile_id} SGA saved")
            except Exception as exc:
                print(f"⚠ {tile_id} SGA fail :: {exc}")
                continue

        # ----- band downloads -----
        try:
            b03_jp2 = ensure_jp2(granule_uri, product_tag, "B03", session, bands_dir)
            b11_jp2 = ensure_jp2(granule_uri, product_tag, "B11", session, bands_dir)
            b12_jp2 = ensure_jp2(granule_uri, product_tag, "B12", session, bands_dir)
        except Exception as exc:
            print(f"⚠ {tile_id} band download fail :: {exc}")
            continue

        # ----- load rasters -----
        with rasterio.open(b11_jp2) as src:
            transform = src.transform
            crs = src.crs
            b11 = src.read(1).astype(np.float32) / 10_000
        with rasterio.open(b12_jp2) as src:
            b12 = src.read(1).astype(np.float32) / 10_000
        with rasterio.open(b03_jp2) as src:
            b03_full = src.read(1).astype(np.float32) / 10_000  # 10 m in L1C
            # --- make sure B03 matches the 20 m grid of B11/B12 -----------------
            if b03_full.shape != b11.shape:  # 10980×10980 → 5490×5490
                factor = b03_full.shape[0] // b11.shape[0]  # =2 for Sentinel-2
                b03 = b03_full[::factor, ::factor]  # decimate rows/cols
            else:
                b03 = b03_full
        with rasterio.open(sga_path) as src:
            sga = src.read(1).astype(np.float32)

        # ----- centre in product CRS -----
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        cx, cy = transformer.transform(centre_lon, centre_lat)

        # ----- MBSP -----
        R = mbsp_numpy(b03, b11, b12, sga, transform, (cx, cy))
        if R is None:
            print(f"• {tile_id} skipped (low B12)")
            continue

        profile = {
            "driver": "GTiff",
            "width": R.shape[1],
            "height": R.shape[0],
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": np.nan,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 2,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(R, 1)

        print(f"✔ {tile_id} MBSP → {out_path}")


# ---------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------
# %% (main)
# ----- user parameters -----
lon, lat = -90.96802087968751, 27.29220815000002  # AOI centre (e.g. NYC test)
start, end = "2017-07-04", "2017-07-06"
max_cloud = 20
threads = 6
out_root = Path("../data")

# ----- credentials -----
dotenv.load_dotenv(".env")
user = os.getenv("CDSE_USERNAME")
pw = os.getenv("CDSE_PASSWORD")
if not (user and pw):
    raise SystemExit("Need CDSE_USERNAME / CDSE_PASSWORD")

# ----- Earth Engine product list -----
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

# ----- CDSE session -----
session = requests.Session()
session.headers["Authorization"] = f"Bearer {cdse_token(user, pw)}"

# ----- parallel processing -----
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=threads) as ex:
    list(
        ex.map(
            lambda p: process_granule(
                p, out_root, session, centre_lon=lon, centre_lat=lat
            ),
            products,
        )
    )
print(f"Done in {time.time() - t0:.1f} s")

# %%
