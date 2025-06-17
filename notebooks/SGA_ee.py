# %%
#!/usr/bin/env python3
"""
gee_mbsp.py
~~~~~~~~~~~
Sentinel-2 MBSP with maximum server-side execution in Google Earth Engine.

2025-06-16  • Upload *un-interpolated* 23 x 23 SGA grid; resample in EE.
"""

# ---------------------------------------------------------------------
#  Imports & config
# ---------------------------------------------------------------------
import os
import re
import subprocess
import time
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from lxml import etree
from pyproj import Transformer  # noqa: F401 (kept)
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator  # noqa: F401 (still used)

# ---------------------------------------------------------------------
#  User-editable parameters
# ---------------------------------------------------------------------
CENTRE_LON, CENTRE_LAT = -90.96802087968751, 27.29220815000002
START, END = "2017-07-04", "2017-07-06"
MAX_CLOUD = 20

ASSET_ROOT = "projects/cerulean-338116/assets/offshore_methane"
GCS_BUCKET = "offshore_methane"  # staging bucket
MAX_CONCURRENT_TASKS = 10
GLINT_MASK_RANGE = (0.0, 20.0)  # SGA degrees kept as "good"
AOI_RADIUS_M = 5_000
# Algorithm selection ―‒ set to True for the simpler fractional-slope MBSP (Varon et al. 2021)
USE_SIMPLE_MBSP = False

SHOW_THUMB = True  # False to silence QA thumbnails
EXPORT_PARAM = {  # Uncommment Zero or One of the following
    # "bucket": GCS_BUCKET,
    # "ee_asset": ASSET_ROOT,
}
assert len(EXPORT_PARAM) < 2, "Can only set more than one export parameter"


# ---------------------------------------------------------------------
#  0.  Sun-Glint-Angle helpers  (CDSE + new coarse-grid writer)
# ---------------------------------------------------------------------
load_dotenv("../.env")
_cdse_user = os.getenv("CDSE_USERNAME")
_cdse_pw = os.getenv("CDSE_PASSWORD")
if not (_cdse_user and _cdse_pw):
    raise SystemExit("Need CDSE_USERNAME / CDSE_PASSWORD in .env")

_LOGIN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"
_DL_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"


def _cdse_token(user: str, pw: str, client="cdse-public") -> str:
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
_session.headers["Authorization"] = f"Bearer {_cdse_token(_cdse_user, _cdse_pw)}"


def _parse_pid(pid: str) -> dict:
    _PRODUCT_RE = re.compile(
        r"^(S2[AB])_MSIL(?P<lvl>[12])[AC]_"  # platform & level
        r"(?P<start>\d{8}T\d{6})_"  # datatake start
        r"N\d{4}_R\d{3}_T(?P<tile>\d{2}[A-Z]{3})_"
        r"(?P<proc>\d{8}T\d{6})$"
    )
    m = _PRODUCT_RE.match(pid)
    if not m:
        raise ValueError(f"Unrecognised Sentinel-2 Product ID: {pid}")
    d = m.groupdict()
    d["sc"] = "Sentinel-2A" if m.group(1) == "S2A" else "Sentinel-2B"
    d["date"] = d["start"][:8]
    return d


def _cdse_uuid(product_id: str) -> str | None:
    p = _parse_pid(product_id)
    ts = p["start"]
    tile = p["tile"]
    levelstr = f"MSIL{p['lvl']}C"
    url = (
        f"{_CATALOG}/Products?$select=Id,Name&$top=1"
        f"&$orderby=ContentDate/Start desc"
        f"&$filter=contains(Name,'{ts}') and contains(Name,'T{tile}') "
        f"and contains(Name,'{levelstr}')"
    )
    r = _session.get(url, timeout=30)
    r.raise_for_status()
    items = r.json().get("value", [])
    return items[0]["Id"] if items else None


def _list_nodes(uri: str) -> list[dict]:
    r = _session.get(uri, timeout=30)
    r.raise_for_status()
    js = r.json()
    if "value" in js:
        return js["value"]
    if "result" in js:
        return js["result"]
    if "d" in js and "results" in js["d"]:
        return js["d"]["results"]
    return []


def _granule_uri(product_id: str) -> str:
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


def _download_object(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with _session.get(url, stream=True, timeout=120) as r, open(tmp, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            f.write(chunk)
    tmp.rename(out_path)


def _download_xml(product_id: str, xml_path: Path):
    if xml_path.is_file():
        return
    _download_object(f"{_granule_uri(product_id)}/Nodes(MTD_TL.xml)/$value", xml_path)


# ---------- NEW:  produce COARSE 23x23 GeoTIFF (5 km pixels) ----------
def compute_sga_coarse(product_id: str, tif_path: Path) -> None:
    """
    Save the un-interpolated 23 x 23 SGA grid (5 km pixels) as GeoTIFF
    with correct orientation for Earth Engine.
    """
    xml_path = tif_path.with_suffix(".xml")
    _download_xml(product_id, xml_path)
    root = etree.parse(str(xml_path))

    # -- helpers -----------------------------------------------------
    def _grid2array(values_list):
        rows = [np.fromstring(row, sep=" ") for row in values_list]
        return np.vstack(rows).astype(np.float32)

    def _mean_detector_grid(root_, band_id: str, xpath_suffix: str):
        expr = (
            f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']/{xpath_suffix}"
            "/Values_List/VALUES/text()"
        )
        blocks = root_.xpath(expr)
        arrays = [
            _grid2array(blocks[i * 23 : (i + 1) * 23]) for i in range(len(blocks) // 23)
        ]
        return np.nanmean(arrays, axis=0)

    # -- raw 23x23 values -------------------------------------------
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

    sga_grid = np.rot90(np.flipud(sga_grid.T), k=-1)

    # GeoTransform: 5 km pixels
    ulx = float(root.findtext(".//Geoposition/ULX"))
    uly = float(root.findtext(".//Geoposition/ULY"))
    tr = from_origin(ulx, uly, 5000.0, 5000.0)  # ← NEW pixel size

    profile = {
        "driver": "GTiff",
        "width": 23,
        "height": 23,
        "count": 1,
        "dtype": "float32",
        "crs": root.findtext(".//Tile_Geocoding/HORIZONTAL_CS_CODE"),
        "transform": tr,
        "compress": "deflate",
        "predictor": 2,
    }
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(sga_grid, 1)


# ---------------------------------------------------------------------
#  1.  EE helpers
# ---------------------------------------------------------------------
ee.Initialize()


def sentinel2_product_ids(point: ee.Geometry, start: str, end: str, cloud_pct: int):
    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(point)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    return sorted(set(coll.aggregate_array("PRODUCT_ID").getInfo()))


def ee_asset_exists(asset_id: str) -> bool:
    try:
        ee.Image(asset_id).getInfo()
        return True
    except Exception:
        return False


def gcs_stage(local_path: Path, bucket: str) -> str:
    dst = f"gs://{bucket}/sga/{local_path.name}"
    subprocess.run(
        ["gsutil", "cp", "-n", str(local_path.resolve()), dst],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return dst


def ensure_sga_asset(
    product_id: str,
    out_root: Path = Path("../data"),
    ee_asset: str = None,
    bucket: str = None,
) -> str:
    prefixed_product_id = f"SGA_{product_id}"
    asset_id = f"{ASSET_ROOT}/{prefixed_product_id}"
    if ee_asset_exists(asset_id):
        return asset_id

    tif_path = out_root / "sga" / f"{prefixed_product_id}.tif"
    if not tif_path.is_file():
        print(f"  ↻ computing *coarse* SGA grid for {product_id}")
        compute_sga_coarse(product_id, tif_path)

    print("  ↑ staging coarse GeoTIFF to GCS")
    gcs_url = gcs_stage(tif_path, GCS_BUCKET)

    if EXPORT_PARAM.get("ee_asset"):
        print(f"  ↑ ingesting as EE asset {asset_id}")
        subprocess.run(
            ["earthengine", "upload", "image", f"--asset_id={asset_id}", gcs_url],
            check=True,
        )

        while not ee_asset_exists(asset_id):
            time.sleep(1)

        subprocess.run(["gsutil", "rm", gcs_url], stdout=subprocess.DEVNULL)
    return asset_id


# ---------------------------------------------------------------------
#  2.  MBSP pipeline in Earth Engine  (unchanged math; coarse SGA upscale)
# ---------------------------------------------------------------------
def mbsp_ee(image: ee.Image, sga_coarse: ee.Image, centre: ee.Geometry) -> ee.Image:
    """Return a single-band EE Image with fractional MBSP R."""

    b03 = image.select("B3").divide(10000)
    b11 = image.select("B11").divide(10000)
    b12 = image.select("B12").divide(10000)

    b03 = b03.resample("bilinear").reproject(crs=b11.projection()).rename("B3_20m")

    denom = b12.add(b03).max(1e-4)
    sgi = b12.subtract(b03).divide(denom).clamp(-1, 1)

    mask_geom = centre.buffer(50_000)

    reg1_img = ee.Image.cat(ee.Image.constant(1), sgi, b11.rename("y"))
    lr1 = reg1_img.reduceRegion(
        ee.Reducer.linearRegression(numX=2, numY=1),
        geometry=mask_geom,
        scale=20,
        bestEffort=True,
    )
    coef1 = ee.Array(lr1.get("coefficients"))
    a0 = coef1.get([0, 0])
    a1 = coef1.get([1, 0])
    fit = ee.Image.constant(a0).add(ee.Image.constant(a1).multiply(sgi))
    b11_flat = b11.subtract(fit)

    ratio = b12.divide(b11.add(1e-6))
    b12_flat = b12.subtract(fit.multiply(ratio))

    x1 = b12_flat
    x2 = b12_flat.multiply(sgi)
    reg2_img = ee.Image.cat(x1, x2, b11_flat.rename("y"))
    lr2 = reg2_img.reduceRegion(
        ee.Reducer.linearRegression(numX=2, numY=1),
        geometry=mask_geom,
        scale=20,
        bestEffort=True,
    )
    coef2 = ee.Array(lr2.get("coefficients"))
    c0 = coef2.get([0, 0])
    c1 = coef2.get([1, 0])
    c = ee.Image.constant(c0).add(ee.Image.constant(c1).multiply(sgi))

    R = c.multiply(b12_flat).subtract(b11_flat).divide(b11_flat.add(1e-6))

    # ---------- NEW:  upscale coarse SGA to 20 m ---------------------
    sga_hi = sga_coarse.resample("bilinear").reproject(crs=b11.projection())

    R = R.updateMask(
        sga_hi.gt(GLINT_MASK_RANGE[0]).And(sga_hi.lt(GLINT_MASK_RANGE[1]))
    ).rename("MBSP")
    return ee.Image(R).copyProperties(image, ["PRODUCT_ID", "system:time_start"])


# ---------------------------------------------------------------------
#  2b.  Simpler fractional MBSP (Varon et al. 2021)
# ---------------------------------------------------------------------
def mbsp_simple_ee(image: ee.Image, centre: ee.Geometry) -> ee.Image:
    """Scene-wide zero-intercept regression of B11 on B12."""
    region = centre.buffer(50_000)  # same window used by the complex variant

    # c = Σ(R11·R12) / Σ(R12²)
    num = (
        image.select("B11")
        .multiply(image.select("B12"))
        .reduceRegion(ee.Reducer.sum(), region, 20, bestEffort=True)
    )
    den = (
        image.select("B12")
        .pow(2)
        .reduceRegion(ee.Reducer.sum(), region, 20, bestEffort=True)
    )
    slope = ee.Number(num.get("B11")).divide(ee.Number(den.get("B12")))

    R = (
        image.select("B12")
        .multiply(slope)
        .subtract(image.select("B11"))
        .divide(image.select("B11"))
        .rename("MBSP")
        .set({"slope": slope})
    )
    return R.copyProperties(image, ["PRODUCT_ID", "system:time_start"])


# ---------------------------------------------------------------------
#  3.  Export helper
# ---------------------------------------------------------------------


def export_image(
    image: ee.Image,
    description: str,
    region: ee.Geometry,
    bucket=None,
    ee_asset=None,
):
    image = ee.Image(image)
    roi = region.bounds().coordinates().getInfo()
    task = None

    if bucket:
        utm = image.select("MBSP").projection()
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=description,
            bucket=bucket,
            fileNamePrefix=f"MBSP/{description}",
            region=roi,
            scale=20,
            crs=utm,
            maxPixels=1 << 36,
        )
    elif ee_asset:
        utm = image.select("MBSP").projection()
        task = ee.batch.Export.image.toAsset(
            image=image,
            description=description,
            assetId=f"{ee_asset}/{description}",
            region=region.bounds().coordinates().getInfo(),
            scale=20,
            crs=utm,
            maxPixels=1 << 36,
            pyramidingPolicy={"MBSP": "sample"},
        )
    if task:
        task.start()
    return task


# ---------------------------------------------------------------------
#  4.  Main driver
# ---------------------------------------------------------------------
def main():
    centre_pt = ee.Geometry.Point([CENTRE_LON, CENTRE_LAT])
    products = sentinel2_product_ids(centre_pt, START, END, MAX_CLOUD)
    if not products:
        raise SystemExit("No matching Sentinel-2 products.")
    print(f"Found {len(products)} product(s)")

    active = []
    for pid in products:
        print(f"▶ {pid}")
        s2 = ee.Image(
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filter(ee.Filter.eq("PRODUCT_ID", pid))
            .first()
        )
        if s2 is None:
            print("  ⚠ EE image not found; skipped")
            continue

        if USE_SIMPLE_MBSP:
            # Simple fractional-slope MBSP (no glint masking)
            print("  ⚠ Simple MBSP selected")
            R_img = ee.Image(mbsp_simple_ee(s2, centre_pt))
        else:
            # Complex two-stage MBSP with coarse-SGA masking
            print("  ⚠ Complex MBSP selected")
            sga_asset = ensure_sga_asset(pid, **EXPORT_PARAM)
            sga_img = ee.Image(sga_asset)
            R_img = ee.Image(mbsp_ee(s2, sga_img, centre_pt))

        export_roi = centre_pt.buffer(AOI_RADIUS_M)
        R_img = R_img.clip(export_roi)
        R_img = R_img.toFloat()
        if SHOW_THUMB:
            png_url = R_img.visualize(
                min=-0.1, max=0.1, palette=["red", "white", "blue"]
            ).getThumbURL({"region": centre_pt.buffer(AOI_RADIUS_M), "dimensions": 512})
            print(f"  🖼  thumbnail → {png_url}")
        task = export_image(
            R_img,
            description=f"MBSP_{pid}",
            region=centre_pt.buffer(AOI_RADIUS_M),
            **EXPORT_PARAM,
        )
        if task:
            active.append(task)
            print(f"  ⧗ export {task.id} started")
        else:
            print("  ↪ export skipped")

        while (
            len([t for t in active if t.status()["state"] in ("READY", "RUNNING")])
            >= MAX_CONCURRENT_TASKS
        ):
            time.sleep(1)

    print("Waiting for EE exports to finish …")
    while any(t.status()["state"] in ("READY", "RUNNING") for t in active):
        statuses = {}
        for t in active:
            s = t.status()
            statuses[t.id] = f"{s['state']} ({s.get('progress', '?')}%)"
            if s.get("error_message"):
                statuses[t.id] += f"  ERROR: {s['error_message']}"
        print(statuses)
        time.sleep(10)  # poll every 10 s, not every second
    print("Done.")


if __name__ == "__main__":
    main()
# %%
