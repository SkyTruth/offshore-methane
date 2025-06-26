# masking.py
"""
Pixel-mask builders for C-factor and MBSP calculations.

All helpers return a single-band Boolean `ee.Image` named **"mask"**
where *1* keeps a pixel and *0* discards it.
"""

from __future__ import annotations

import math
from typing import Dict

import ee

# ---------------------------------------------------------------------
#  Module-level constants (avoid recomputing in map() lambdas)
# ---------------------------------------------------------------------
_DEG2RAD = math.pi / 180.0
_EARTH_R = 6_371_000  # m - WGS-84 authalic radius

# ---------------------------------------------------------------------
#  Default parameters
# ---------------------------------------------------------------------
DEFAULT_MASK_PARAMS: Dict[str, Dict] = {
    "disk": {"radius_m": 5_000},  # AOI around infrastructure
    "plume_disk": {"radius_m": 1_000},  # Upwind plume subset
    "wind_disk": {"radius_m": 5_000},  # Radius for wind test
    "saturation": {"bands": ["B11", "B12"], "max": 10_000},
    "cloud": {"qa_band": "cs_cdf", "threshold": 0.65},
    "outlier": {"bands": ["B11", "B12"], "p_low": 10, "p_high": 90},
    "ndwi": {"threshold": 0.0},
    "sga": {"max_deg": 20},
    "sgi": {"min": -0.30},
    "min_viable_mask_fraction": 0.005,
}


# ---------------------------------------------------------------------
#  Tiny utilities
# ---------------------------------------------------------------------
def _single_band(bmask: ee.Image) -> ee.Image:
    """Collapse a multi-band Boolean image to one band named 'mask'."""
    return bmask.reduce(ee.Reducer.min()).rename("mask")


def _geom_mask(centre: ee.Geometry, radius_m: float) -> ee.Image:
    """Disk mask = 1 inside radius, masked outside."""
    return ee.Image.constant(1).clip(centre.buffer(radius_m)).rename("mask")


# ---------------------------------------------------------------------
#  Sub-mask builders
# ---------------------------------------------------------------------
def geom_mask(img: ee.Image, centre: ee.Geometry, p: Dict) -> ee.Image:  # noqa: D401
    return _geom_mask(centre, p["disk"]["radius_m"])


def saturation_mask(img: ee.Image, p: Dict) -> ee.Image:
    good = img.select(p["saturation"]["bands"]).lt(p["saturation"]["max"])
    return _single_band(good)


def cloud_mask(img: ee.Image, p: Dict) -> ee.Image:
    # Cloud mask using **Cloud Score+** where available, else SimpleCloudScore.
    qa_band = p["cloud"].get("qa_band", "cs_cdf")
    thr = float(p["cloud"]["threshold"])
    thr = max(0.0, min(1.0, thr))  # clamp for safety

    # 1) Try Cloud Score+ (shares system:index with source S2 asset)
    sid = ee.String(img.get("system:index"))
    csplus = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", sid))
        .select(qa_band)
        .first()
    )

    has_csplus = csplus.bandNames().size().gt(0)

    # 2) Fallback → SimpleCloudScore (0-100, *lower* is clearer)
    simple_score = ee.Algorithms.SimpleCloudScore(img).select("score")
    simple_clear = simple_score.lt(thr * 100)  # keep where score < threshold%

    # 3) Choose mask depending on availability
    mask = ee.Image(ee.Algorithms.If(has_csplus, csplus.gte(thr), simple_clear))

    return mask.rename("mask")


def outlier_mask(img: ee.Image, centre: ee.Geometry, p: Dict) -> ee.Image:
    bands = p["outlier"]["bands"]
    stats = img.select(bands).reduceRegion(
        ee.Reducer.percentile([p["outlier"]["p_low"], p["outlier"]["p_high"]]),
        centre.buffer(p["disk"]["radius_m"]),
        scale=20,
        bestEffort=True,
    )

    def band_mask(b: str) -> ee.Image:
        lo = ee.Number(stats.get(f"{b}_p{p['outlier']['p_low']}"))
        hi = ee.Number(stats.get(f"{b}_p{p['outlier']['p_high']}"))
        return img.select(b).gte(lo).And(img.select(b).lte(hi))

    stacked = ee.ImageCollection([band_mask(b) for b in bands]).toBands()
    return _single_band(stacked)


def ndwi_mask(img: ee.Image, p: Dict) -> ee.Image:
    ndwi = img.normalizedDifference(["B3", "B8"])
    return ndwi.gt(p["ndwi"]["threshold"]).rename("mask")


def sga_mask(img: ee.Image, p: Dict) -> ee.Image:
    return img.select("SGA").lt(p["sga"]["max_deg"]).rename("mask")


def sgi_mask(img: ee.Image, p: Dict) -> ee.Image:
    return img.select("SGI").gt(p["sgi"]["min"]).rename("mask")


def wind_mask(img: ee.Image, centre: ee.Geometry, p: Dict) -> ee.Image:
    """
    1 for pixels *not* in the down-wind half—0 for pixels likely affected.
    """
    # --- wind vector -------------------------------------------------
    t0 = ee.Date(img.get("system:time_start"))
    met = (
        ee.ImageCollection("NOAA/CFSV2/FOR6H")
        .filterDate(t0.advance(-3, "hour"), t0.advance(3, "hour"))
        .first()
    )
    uv = met.sample(centre, 20_000).first()
    u = ee.Number(uv.get("u-component_of_wind_height_above_ground"))
    v = ee.Number(uv.get("v-component_of_wind_height_above_ground"))
    mag = u.hypot(v).max(1e-6)
    ux, uy = u.divide(mag), v.divide(mag)

    # --- per-pixel offsets (metres) ----------------------------------
    lonlat = ee.Image.pixelLonLat()
    lon0 = ee.Number(centre.coordinates().get(0))
    lat0 = ee.Number(centre.coordinates().get(1))
    dx = (
        lonlat.select("longitude")
        .subtract(lon0)
        .multiply(_DEG2RAD)
        .multiply(_EARTH_R)
        .multiply(lonlat.select("latitude").multiply(_DEG2RAD).cos())
    )
    dy = lonlat.select("latitude").subtract(lat0).multiply(_DEG2RAD).multiply(_EARTH_R)

    # --- predicates --------------------------------------------------
    r2 = p["wind_disk"]["radius_m"] ** 2
    inside = dx.pow(2).add(dy.pow(2)).lte(r2)
    downwind = dx.multiply(ux).add(dy.multiply(uy)).gte(0)

    return inside.And(downwind).Not().rename("mask")


# ---------------------------------------------------------------------
#  Compound builders
# ---------------------------------------------------------------------
def build_mask_for_C(
    img: ee.Image, centre: ee.Geometry, p: Dict = DEFAULT_MASK_PARAMS
) -> ee.Image:
    mask = (
        geom_mask(img, centre, p)
        .And(saturation_mask(img, p))
        .And(cloud_mask(img, p))
        .And(outlier_mask(img, centre, p))
        .And(ndwi_mask(img, p))
        .And(sga_mask(img, p))
    )
    return mask.rename("mask")


def build_mask_for_MBSP(
    img: ee.Image, centre: ee.Geometry, p: Dict = DEFAULT_MASK_PARAMS
) -> ee.Image:
    mask = (
        saturation_mask(img, p)
        .And(cloud_mask(img, p))
        .And(ndwi_mask(img, p))
        .And(sga_mask(img, p))
        .And(sgi_mask(img, p))
        .And(wind_mask(img, centre, p))
    )
    return mask.rename("mask")
