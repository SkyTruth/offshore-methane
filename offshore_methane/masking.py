# %%
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
import geemap
import numpy as np

# ---------------------------------------------------------------------
#  Module-level constants (avoid recomputing in map() lambdas)
# ---------------------------------------------------------------------
_DEG2RAD = math.pi / 180.0
_EARTH_R = 6_371_000  # m - WGS-84 authalic radius

DEFAULT_MASK_PARAMS: Dict[str, Dict] = {
    "dist": {"local_radius_m": 5_000, "plume_radius_m": 1_000},
    "cloud": {
        "scene_cloud_pct": 20,  # metadata
        "local_cloud_pct": 5,  # QA60 UNUSED?!
        "cs_thresh": 0.65,  # cloudy above
        "prob_thresh": 65,  # BACKUP: cloudy below
    },
    "wind": {"max_wind_10m": 9},  # m s‑1, ERA5 / CFSv2 upper limit
    "outlier": {
        "bands": ["B11", "B12"],
        "p_low": 2,
        "p_high": 98,
        "saturation": 10_000,
    },
    "ndwi": {"threshold": 0.0},
    "sunglint": {
        "scene_sga_range": (0.0, 40.0),  # deg
        "local_sga_range": (0.0, 30.0),  # deg
        "local_sgi_range": (-0.30, 1.0),  # NDI
    },
    "min_viable_mask_fraction": 0.005,  # UNUSED?!
}


# ---------------------------------------------------------------------
#  Scene filters
# ---------------------------------------------------------------------
def scene_cloud_filter(p: Dict) -> ee.Filter:
    return ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", p["cloud"]["scene_cloud_pct"])


def scene_sga_filter(img: ee.Image, p: Dict) -> ee.Image:
    """Add a Boolean property 'SGA_OK' based on scene-glint angle limits."""
    # metadata → Numbers
    sza = ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE"))
    saa = ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    vza = ee.Number(img.get("MEAN_INCIDENCE_ZENITH_ANGLE_B11"))
    vaa = ee.Number(img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B11"))

    # radians
    rad = ee.Number(np.pi).divide(180)
    sza_r = sza.multiply(rad)
    vza_r = vza.multiply(rad)
    dphi_r = saa.subtract(vaa).abs().multiply(rad)

    cos_sga = (
        sza_r.cos()
        .multiply(vza_r.cos())
        .subtract(sza_r.sin().multiply(vza_r.sin()).multiply(dphi_r.cos()))
    )
    sga_deg = cos_sga.acos().multiply(180 / np.pi)

    ok = sga_deg.gt(p["sunglint"]["scene_sga_range"][0]).And(
        sga_deg.lt(p["sunglint"]["scene_sga_range"][1])
    )

    # keep images that *lack* metadata
    return img.set(
        "SGA_OK",
        ee.Algorithms.If(
            img.propertyNames().contains("MEAN_SOLAR_ZENITH_ANGLE"), ok, True
        ),
    )


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
    return _geom_mask(centre, p["dist"]["local_radius_m"])


def saturation_mask(img: ee.Image, p: Dict) -> ee.Image:
    good = img.select(p["outlier"]["bands"]).lt(p["outlier"]["saturation"])
    return _single_band(good)


def cloud_mask(img: ee.Image, p: Dict) -> ee.Image:
    """
    Tiered cloud mask for Sentinel-2.

    1) Cloud Score+  (cloud + shadow)  → keep where cs ≥ cs_thresh
    2) s2cloudless   (cloud prob % )   → keep where prob < prob_thresh
    3) Fallback (rare)                 → keep everything
    """
    cs_thr = float(p["cloud"].get("cs_thresh", 0.65))
    prob_thr = int(p["cloud"].get("prob_thresh", 60))
    sid = ee.String(img.get("system:index"))

    # 1️⃣ Cloud Score+
    cs_col = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filter(ee.Filter.eq("system:index", sid))
        .select("cs_cdf")
    )
    has_cs = cs_col.size().gt(0)

    # 2️⃣ s2cloudless probability
    prob_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.eq("system:index", sid))
        .select("probability")
    )
    has_prob = prob_col.size().gt(0)

    mask = ee.Image(
        ee.Algorithms.If(
            has_cs,
            cs_col.first().gte(cs_thr),
            ee.Algorithms.If(
                has_prob,
                prob_col.first().lt(prob_thr),
                ee.Image.constant(1),  # fallback: no QA available
            ),
        )
    )

    return mask.rename("mask")


def outlier_mask(img: ee.Image, p: Dict) -> ee.Image:
    bands = p["outlier"]["bands"]
    stats = img.select(bands).reduceRegion(
        ee.Reducer.percentile([p["outlier"]["p_low"], p["outlier"]["p_high"]]),
        img.geometry(),
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
    return img.select("SGA").lt(p["sunglint"]["local_sga_range"][1]).rename("mask")


def sgi_mask(img: ee.Image, p: Dict) -> ee.Image:
    return img.select("SGI").gt(p["sunglint"]["local_sgi_range"][0]).rename("mask")


def wind_mask(
    img: ee.Image,
    centre: ee.Geometry,
    radius_m: float,
    p: Dict,
    inside: bool = True,
    downwind: bool = True,
    invert: bool = False,
    time_window: int = 3,
) -> ee.Image:
    """
    Returns 1 for pixels *not* in the down-wind half-disk, 0 otherwise.
    And where wind speed is less than max_wind_10m.

    Robust to:
      • No CFSv2 slice within ±time_window h
      • Reduction over the slice returning nulls

    Fallback is an all-ones mask so downstream ops never break.
    """
    t0 = ee.Date(img.get("system:time_start"))
    met = (
        ee.ImageCollection("NOAA/CFSV2/FOR6H")
        .filterDate(t0.advance(-time_window, "hour"), t0.advance(time_window, "hour"))
        .first()
    )

    # helper → compute mask given a valid met image
    def _make_mask(met_img):
        # Reduce over a disk
        met_hi = met_img.resample("bilinear").reproject(crs="EPSG:4326", scale=1000)
        stats = met_hi.select(
            [
                "u-component_of_wind_height_above_ground",
                "v-component_of_wind_height_above_ground",
            ]
        ).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=centre.buffer(radius_m),
            scale=20_000,
            bestEffort=True,
        )

        def _build(u, v):
            mag = u.hypot(v).max(1e-6)
            ux, uy = u.divide(mag), v.divide(mag)

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
            dy = (
                lonlat.select("latitude")
                .subtract(lat0)
                .multiply(_DEG2RAD)
                .multiply(_EARTH_R)
            )

            r2 = radius_m**2
            in_out = (
                dx.pow(2).add(dy.pow(2)).lte(r2)
                if inside
                else dx.pow(2).add(dy.pow(2)).gt(r2)
            )
            up_down = (
                dx.multiply(ux).add(dy.multiply(uy)).gte(0)
                if downwind
                else dx.multiply(ux).add(dy.multiply(uy)).lt(0)
            )

            mask = in_out.And(up_down)
            if invert:
                mask = mask.Not()

            return mask.And(mag.lt(p["wind"]["max_wind_10m"])).rename("mask")

        return ee.Image(
            _build(
                ee.Number(stats.get("u-component_of_wind_height_above_ground")),
                ee.Number(stats.get("v-component_of_wind_height_above_ground")),
            )
        )

    # ------------------------------------------------------------------
    # choose met-branch or fall back
    # ------------------------------------------------------------------
    return ee.Image(
        ee.Algorithms.If(met, _make_mask(met), ee.Image.constant(1).rename("mask"))
    )


# ---------------------------------------------------------------------
#  Compound builders
# ---------------------------------------------------------------------
def build_mask_for_C(
    img: ee.Image,
    centre: ee.Geometry,
    p: Dict = DEFAULT_MASK_PARAMS,
) -> ee.Image:
    mask = (
        geom_mask(img, centre, p)
        .And(saturation_mask(img, p))
        .And(cloud_mask(img, p))
        .And(outlier_mask(img, p))
        .And(ndwi_mask(img, p))
        .And(
            wind_mask(
                img,
                centre,
                p["dist"]["plume_radius_m"],
                p,
                inside=True,
                downwind=True,
                invert=True,
            )
        )
        .And(sga_mask(img, p))
        .And(sgi_mask(img, p))
    )
    return mask.rename("mask")


def build_mask_for_MBSP(
    img: ee.Image, centre: ee.Geometry, p: Dict = DEFAULT_MASK_PARAMS
) -> ee.Image:
    mask = (
        saturation_mask(img, p)
        .And(cloud_mask(img, p))
        .And(ndwi_mask(img, p))
        .And(
            wind_mask(
                img,
                centre,
                p["dist"]["local_radius_m"],
                p,
                inside=True,
                downwind=True,
            )
        )
    )
    return mask.rename("mask")


def view_mask(sid: str) -> geemap.Map:
    """
    Overlay `build_mask_for_C` (red) and `build_mask_for_MBSP` (green)
    on a Sentinel-2 true-colour image in an interactive geemap.Map.

    Parameters
    ----------
    sid   : Sentinel-2 system:index (e.g. ``"20210704T160919_20210704T162116_T16TGJ"``)

    Returns
    -------
    geemap.Map
        Interactive map; in notebooks just evaluate to render, or call
        ``m.to_html("outfile.html")`` to export.
    """
    ee.Initialize()

    # 1️⃣ Load image and derive centre geometry
    img = ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")
    if img is None:
        raise ValueError(f"Sentinel-2 image {sid!r} not found.")
    CENTRE_LON, CENTRE_LAT = -90.96802087968751, 27.29220815000002

    centre = ee.Geometry.Point([CENTRE_LON, CENTRE_LAT])

    # 2️⃣ True-colour backdrop (stretch a touch for contrast)
    rgb = (
        img.select(["B4", "B3", "B2"])
        .divide(10_000)  # convert DN→reflectance 0-1
        .visualize(min=0.05, max=0.35)
    )

    # Add SGA to image as a new band called "SGA"
    sga_img = ee.Image.loadGeoTIFF(f"gs://offshore_methane/{sid}/{sid}_SGA.tif")
    img = img.addBands(sga_img.rename("SGA"))

    # Add SGI to image as a new band called "SGI"
    b_vis = img.select("B2").add(img.select("B3")).add(img.select("B4")).divide(3)
    img = img.addBands(b_vis.rename("B_vis"))
    sgi = img.normalizedDifference(["B12", "B_vis"])
    img = img.addBands(sgi.rename("SGI"))

    # 3️⃣ Build masks, turn into single-colour rasters
    mask_c = build_mask_for_C(img, centre, DEFAULT_MASK_PARAMS).selfMask()
    mask_m = build_mask_for_MBSP(img, centre, DEFAULT_MASK_PARAMS).selfMask()

    vis_c = mask_c.visualize(palette=["#FF0000"])  # red
    vis_m = mask_m.visualize(palette=["#00FF00"])  # green

    # 4️⃣ Assemble map
    lon, lat = centre.coordinates().getInfo()
    m = geemap.Map(center=[lat, lon], zoom=11)
    m.addLayer(rgb, {}, "True colour")
    m.addLayer(vis_c, {"opacity": 0.6}, "Mask C (red)")
    m.addLayer(vis_m, {"opacity": 0.6}, "Mask MBSP (green)")
    m.addLayerControl()
    return m


# %%
view_mask("20170705T164319_20170705T165225_T15RXL")

# %%
