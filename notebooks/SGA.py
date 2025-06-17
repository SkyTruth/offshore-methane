# %%
#!/usr/bin/env python3
"""
sga.py
~~~~~~~~~~~~~~~~~~~
End-to-end Sentinel-2 MBSP processing that leverages *pre-computed*
10 m Sun-Glint-Angle (SGA) rasters.

Changes 2025-06-15
------------------
* Added a CONFIG stanza - all magic numbers and algorithmic switches live here.
* `mbsp_numpy()` can now compute either:
      - the *simple* scene-wide constant `c`
      - the *polynomial* SGI-dependent `c = c0 + c1·SGI`
  based on CONFIG["SCALING_MODE"].
* Added optional QA60 cloud mask, radius and glint-mask limits to CONFIG.
* Safer handling of empty masks and low-signal scenes: always emit a raster.
* Minor typing / print-statement hygiene.

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
from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

import ee
import numpy as np
import rasterio
from pyproj import Transformer

from offshore_methane.cdse import (
    band_path,
    compute_sga_full,
    download_bands,
)

# ----------------------------  CONFIG  --------------------------------
CONFIG: dict[str, object] = {
    # ► Core AOI and temporal settings
    "LON": -90.96802087968751,
    "LAT": 27.29220815000002,
    "DATE_START": "2017-07-04",
    "DATE_END": "2017-07-06",
    "MAX_CLOUD_PERCENT": 20,
    # ► MBSP algorithm switches
    #     "simple"      - single least-squares constant 'c'
    #     "polynomial"  - c = c0 + c1·SGI   (original code path)
    "SCALING_MODE": "simple",
    # ► Numeric / mask parameters
    "AOI_RADIUS_M": 50_000,  # radius for local fitting [m]
    "MIN_B12_REFLECTANCE": 0.01,  # skip granules darker than this median
    "GLINT_MASK_RANGE": (1.0, 15.0),  # SGA degrees kept as "good"
    # ► Parallelism
    "THREADS": 6,
    # ► IO
    "DATA_ROOT": Path("../data"),
}

# ---------------------------------------------------------------------
#  Generic helpers
# ---------------------------------------------------------------------


def sentinel2_product_ids(point, start, end, cloud_pct):
    """Return a list of product-level IDs that satisfy the query."""
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
def mbsp_numpy(
    b03: np.ndarray,
    b11: np.ndarray,
    b12: np.ndarray,
    sga: np.ndarray,
    transform,
    centre_xy: tuple[float, float],
    *,
    cfg: dict[str, object],
) -> np.ndarray:
    """
    Return fractional MBSP raster R (float32) with glint/cloud mask applied.

    The scaling constant *c* is computed according to cfg["SCALING_MODE"]:
        • "simple"     - single least-squares slope
        • "polynomial" - c = c0 + c1·SGI           (default / previous behaviour)
    """
    # Down-sample SGA (10 m → 20 m) if needed
    if sga.shape != b11.shape:  # 10980 × 10980 vs 5490 × 5490
        factor = sga.shape[0] // b11.shape[0]  # =2 for Sentinel-2
        sga = sga[::factor, ::factor]

    # ---------- Sanity checks ----------
    if np.nanmedian(b12) <= cfg["MIN_B12_REFLECTANCE"]:
        return np.full_like(b11, np.nan, dtype=np.float32)

    # ---------- Sun-glint index ----------
    denom = np.maximum(b12 + b03, 1e-4)
    sgi = np.clip((b12 - b03) / denom, -1.0, 1.0)

    # ---------- AOI mask (disc around target) ----------
    rows, cols = b11.shape
    col_vec = np.arange(cols, dtype=np.float32)
    row_vec = np.arange(rows, dtype=np.float32)
    x_coords = transform[0] + (col_vec + 0.5) * transform[1]
    y_coords = transform[3] + (row_vec + 0.5) * transform[5]
    dx = x_coords - centre_xy[0]
    dy = y_coords - centre_xy[1]
    r2 = dx[None, :] ** 2 + dy[:, None] ** 2
    mask_aoi = r2 < cfg["AOI_RADIUS_M"] ** 2

    # ---------- Destriping (unchanged) ----------
    sgim = sgi[mask_aoi].ravel()
    b11m = b11[mask_aoi].ravel()
    if sgim.size == 0:
        return np.full_like(b11, np.nan, dtype=np.float32)
    A = np.column_stack((np.ones_like(sgim), sgim))
    a0, a1 = np.linalg.lstsq(A, b11m, rcond=None)[0]
    fit = a0 + a1 * sgi
    b11_flat = b11 - fit

    ratio = b12 / np.where(b11 == 0, np.nan, b11)
    b12_flat = b12 - fit * ratio

    # ---------- Scaling constant 'c' ----------
    mode = cfg["SCALING_MODE"]
    if mode == "simple":
        # scene-wide least-squares slope  b11_flat ≈ c * b12_flat
        X = b12_flat[mask_aoi].ravel()
        Y = b11_flat[mask_aoi].ravel()
        if np.all(np.isnan(X)) or np.all(np.isnan(Y)):
            return np.full_like(b11, np.nan, dtype=np.float32)
        c_simple = np.linalg.lstsq(X[:, None], Y, rcond=None)[0][0]
        c = np.full_like(b11, c_simple, dtype=np.float32)
    elif mode == "polynomial":
        x1 = b12_flat[mask_aoi].ravel()
        x2 = (b12_flat * sgi)[mask_aoi].ravel()
        X = np.column_stack((x1, x2))
        beta = np.linalg.lstsq(X, b11_flat[mask_aoi].ravel(), rcond=None)[0]
        c = beta[0] + beta[1] * sgi
    else:
        raise ValueError(f"Unknown SCALING_MODE: {mode}")

    # ---------- Fractional signal ----------
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (c * b12_flat - b11_flat) / b11_flat

    # ---------- Glint mask ----------
    sga_min, sga_max = cfg["GLINT_MASK_RANGE"]
    good = (sga > sga_min) & (sga < sga_max)
    R[~good] = np.nan
    return R.astype(np.float32)


# ---------------------------------------------------------------------
#  Granule driver
# ---------------------------------------------------------------------
def process_granule(
    prod_name: str,
    *,
    cfg: dict[str, object],
    calc_sga: bool = True,
) -> None:
    out_root: Path = cfg["DATA_ROOT"]
    mbsp_path = out_root / "mbsp" / f"{prod_name}_R.tif"
    sga_path = out_root / "sga" / f"{prod_name}.tif"

    # Ensure bands on disk
    download_bands(prod_name, ["B03", "B11", "B12"], out_root)

    b03_jp2 = band_path(out_root, prod_name, "B03")
    b11_jp2 = band_path(out_root, prod_name, "B11")
    b12_jp2 = band_path(out_root, prod_name, "B12")

    # SGA cache / compute
    if calc_sga and not sga_path.is_file():
        try:
            compute_sga_full(prod_name, sga_path)
            print(f"✔ {prod_name} SGA cached")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ {prod_name} SGA failure :: {exc}")

    with rasterio.open(sga_path) as src:
        sga = src.read(1).astype(np.float32)

    # Load radiance bands
    with rasterio.open(b11_jp2) as src:
        transform = src.transform
        crs = src.crs
        b11 = src.read(1).astype(np.float32) / 10_000
    with rasterio.open(b12_jp2) as src:
        b12 = src.read(1).astype(np.float32) / 10_000
    with rasterio.open(b03_jp2) as src:
        b03_full = src.read(1).astype(np.float32) / 10_000
        # Down-sample B03 (10 m → 20 m) if necessary
        if b03_full.shape != b11.shape:
            factor = b03_full.shape[0] // b11.shape[0]
            b03 = b03_full[::factor, ::factor]
        else:
            b03 = b03_full

        # ----- centre point in product CRS -----
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        cx, cy = transformer.transform(cfg["LON"], cfg["LAT"])

        # ----- MBSP -----
        R = mbsp_numpy(
            b03,
            b11,
            b12,
            sga,
            transform,
            (cx, cy),
            cfg=cfg,
        )

        # ----- Write GeoTIFF -----
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

        print(f"✔ {prod_name} MBSP ({CONFIG['SCALING_MODE']}) → {mbsp_path}")


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
# %%
# Earth Engine initialisation
try:
    ee.Initialize()
except Exception:  # noqa: BLE001
    ee.Authenticate()
    ee.Initialize()

point = ee.Geometry.Point((CONFIG["LON"], CONFIG["LAT"]))
products = sentinel2_product_ids(
    point,
    CONFIG["DATE_START"],
    CONFIG["DATE_END"],
    CONFIG["MAX_CLOUD_PERCENT"],
)
if not products:
    raise SystemExit("No matching Sentinel-2 products.")
print(f"Found {len(products)} product(s)")

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=CONFIG["THREADS"]) as ex:
    ex.map(lambda p: process_granule(p, cfg=CONFIG), products)
print(f"Done in {time.time() - t0:.1f} s")

# %%
