# %%
#!/usr/bin/env python3
"""
sga.py
~~~~~~~~~~~~~~~~~~~
End-to-end Sentinel-2 MBSP processing that leverages *pre-computed* 10 m
Sun-Glint-Angle (SGA) rasters.

  1.  Query Sentinel-2 L2A products in Google Earth Engine for a point,
      date range and cloud-cover threshold.
  2.  Ensure each granule has an on-disk SGA GeoTIFF
      («../data/sga/<product>.tif»); compute it once otherwise.
  3.  Download exactly B03,B11,B12 (20 m) JP2 files per granule
      - nothing more.
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
import time
from pathlib import Path

import ee
import numpy as np
import rasterio
from pyproj import Transformer

from offshore_methane.cdse import band_path, compute_sga_full, download_bands


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
def process_granule(
    prod_name: str,
    centre_lon: float,
    centre_lat: float,
    calc_sga: bool = True,
    out_root: Path = Path("../data"),
):
    mbsp_path = out_root / "mbsp" / f"{prod_name}_R.tif"
    sga_path = out_root / "sga" / f"{prod_name}.tif"

    if not mbsp_path.is_file():
        download_bands(prod_name, ["B03", "B11", "B12"], out_root)

    b03_jp2 = band_path(out_root, prod_name, "B03")
    b11_jp2 = band_path(out_root, prod_name, "B11")
    b12_jp2 = band_path(out_root, prod_name, "B12")

    # ----- SGA cache / compute -----
    if calc_sga:
        if not sga_path.is_file():
            try:
                sga_full = compute_sga_full(prod_name, sga_path)
                print(f"sga_full: {sga_full.shape}")
                print(f"✔ {prod_name} SGA saved")
            except Exception as exc:
                print(f"⚠ {prod_name} SGA fail :: {exc}")
        with rasterio.open(sga_path) as src:
            sga = src.read(1).astype(np.float32)

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

        # ----- centre in product CRS -----
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        cx, cy = transformer.transform(centre_lon, centre_lat)

        # ----- MBSP -----
        R = mbsp_numpy(b03, b11, b12, sga, transform, (cx, cy))
        if R is None:
            print(f"• {prod_name} skipped (low B12)")
            return

        mbsp_path.parent.mkdir(parents=True, exist_ok=True)
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
        with rasterio.open(mbsp_path, "w", **profile) as dst:
            dst.write(R, 1)

        print(f"✔ {prod_name} MBSP → {mbsp_path}")


# %% (main)
# ----- user parameters -----
lon, lat = -90.96802087968751, 27.29220815000002  # AOI centre (e.g. NYC test)
start, end = "2017-07-04", "2017-07-06"
max_cloud = 20
threads = 6

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

# ----- parallel processing -----
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=threads) as ex:
    list(
        ex.map(
            lambda p: process_granule(p, centre_lon=lon, centre_lat=lat),
            products,
        )
    )
print(f"Done in {time.time() - t0:.1f} s")

# %%
