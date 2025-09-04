# %%
# masking.py
"""
Pixel-mask builders for C-factor and MBSP calculations.

All helpers return a single-band Boolean `ee.Image` named **"mask"**
where *1* keeps a pixel and *0* discards it.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import ee
import geemap
import numpy as np
import pandas as pd
from IPython.display import display

import offshore_methane.config as cfg
from offshore_methane.models.sgi_funcs import sgi_hat_img, sgi_std_img

# ---------------------------------------------------------------------
#  Module-level constants (avoid recomputing in map() lambdas)
# ---------------------------------------------------------------------
_DEG2RAD = math.pi / 180.0
_EARTH_R = 6_371_000  # m - WGS-84 authalic radius


# ---------------------------------------------------------------------
#  Scene filters
# ---------------------------------------------------------------------
def scene_cloud_filter(p: Dict) -> ee.Filter:
    return ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", p["cloud"]["scene_cloud_pct"])


def scene_sga_filter(img: ee.Image, p: Dict) -> ee.Number:
    """Compute the scene sun-glint angle (degrees) from image metadata.

    Returns an ee.Number (degrees). No filtering is applied here.
    """
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
    return sga_deg


# ---------------------------------------------------------------------
#  Tiny utilities
# ---------------------------------------------------------------------
def _single_band(bmask: ee.Image) -> ee.Image:
    """Collapse a multi-band Boolean image to one band named 'mask'."""
    return bmask.reduce(ee.Reducer.min()).rename("mask")


# helper to count ON-pixels (True == 1) inside region
def _count(img_bin, region, scale):
    return ee.Number(
        img_bin.unmask(0)
        .reduceRegion(ee.Reducer.sum(), region, scale=scale, maxPixels=1e9)
        .values()
        .get(0)
    )


# ---------------------------------------------------------------------
#  Sub-mask builders
# ---------------------------------------------------------------------
def geom_mask(centre: ee.Geometry, radius_m: float, inside: bool) -> ee.Image:
    """
    1 → pixels *inside* the radius (default) or *outside* when inside=False.
    Returns a single-band image called "mask".
    """
    dx_dy = _lonlat_dx_dy(centre)
    r2 = radius_m**2
    dist2 = dx_dy.select("dx").pow(2).add(dx_dy.select("dy").pow(2))
    mask = dist2.lte(r2) if inside else dist2.gt(r2)
    return mask.rename("mask")


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


def outlier_mask(
    img: ee.Image,
    p: Dict,
    valid_mask: Optional[ee.Image] = None,
) -> ee.Image:
    """
    Percentile-based outlier rejection that is *aware* of the pixels which have
    already passed previous filters.

    Parameters
    ----------
    img : ee.Image
        The original, *unmasked* image.
    p   : Dict
        Parameter dictionary (same as before).
    valid_mask : ee.Image, optional
        Boolean (0/1) mask representing the pixels that have survived all
        filters applied **prior** to this function.  When *None*, the function
        falls back to the image's native mask, preserving the old behaviour.

    Returns
    -------
    ee.Image
        Single-band mask ("mask") that is **both**
        • computed *from* the current valid pixels, *and*
        • restricted *to* those same pixels.
    """
    # Pixels considered valid so far
    current_mask = (
        valid_mask if valid_mask is not None else img.mask().reduce(ee.Reducer.min())
    )

    # Compute statistics only over those pixels
    masked_img = img.updateMask(current_mask)

    bands = p["outlier"]["bands"]
    stats = masked_img.select(bands).reduceRegion(
        reducer=ee.Reducer.percentile([p["outlier"]["p_low"], p["outlier"]["p_high"]]),
        geometry=masked_img.geometry(),
        scale=20,
        bestEffort=True,
    )

    def band_mask(b: str) -> ee.Image:
        lo = ee.Number(stats.get(f"{b}_p{p['outlier']['p_low']}"))
        hi = ee.Number(stats.get(f"{b}_p{p['outlier']['p_high']}"))
        return masked_img.select(b).gte(lo).And(masked_img.select(b).lte(hi))

    stacked = ee.ImageCollection([band_mask(b) for b in bands]).toBands()
    # Ensure we never *revive* pixels that were already invalid
    return _single_band(stacked).And(current_mask).rename("mask")


def ndwi_mask(img: ee.Image, p: Dict) -> ee.Image:
    ndwi = img.normalizedDifference(["B3", "B8"])
    return ndwi.gt(p["ndwi"]["threshold"]).rename("mask")


def sga_mask(img: ee.Image, p: Dict) -> ee.Image:
    return (
        img.select("SGA")
        .gt(p["sunglint"]["local_sga_range"][0])
        .And(img.select("SGA").lt(p["sunglint"]["local_sga_range"][1]))
        .rename("mask")
    )


def sgi_mask(img: ee.Image, p: Dict) -> ee.Image:
    return (
        img.select("SGI")
        .gt(p["sunglint"]["local_sgi_range"][0])
        .And(img.select("SGI").lt(p["sunglint"]["local_sgi_range"][1]))
        .rename("mask")
    )


def sgx_outlier(img, p):
    alpha = img.select("SGA")
    mean = sgi_hat_img(alpha)
    std = sgi_std_img(alpha)
    return (
        img.select("SGI")
        .gte(mean.add(std.multiply(p["sunglint"]["outlier_std_range"][0])))
        .And(
            img.select("SGI").lte(
                mean.add(std.multiply(p["sunglint"]["outlier_std_range"][1]))
            )
        )
    )


def _lonlat_dx_dy(centre: ee.Geometry) -> ee.Image:
    """Return per-pixel Δx, Δy [m] from *centre* in an EPSG:4326 grid."""
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
    return dx.addBands(dy).rename(["dx", "dy"])


def windspeed_mask(
    wind_layers: ee.Image,
    p: dict,
    invert: bool = False,
) -> ee.Image:
    """
    Mask pixels where wind speed is < p['wind']['max_wind_10m'] (default logic).

    Robust: fallback to all-ones image on missing data.
    """
    try:
        speed = wind_layers.select("wind_speed")
    except Exception:
        return ee.Image.constant(1).rename("mask")
    speed_ok = (
        speed.gt(p["wind"]["max_wind_10m"])
        if invert
        else speed.lt(p["wind"]["max_wind_10m"])
    )

    return speed_ok.rename("mask")


def downwind_mask(
    wind_layers: ee.Image,
    centre: ee.Geometry,
    invert: bool = False,
) -> ee.Image:
    """
    1 → pixel lies in the half-plane *downwind* of *centre*, as defined by the
    **velocity vector at that point** (no radius constraint).
    Set `invert=True` to select the *upwind* half-plane instead.

    Robustness: if `wind_dir` at the centre is missing/null, returns an
    all-ones mask so downstream operations continue gracefully.
    """
    # wind direction (degrees) exactly at the centre
    dir_stat = wind_layers.select("wind_dir").reduceRegion(
        reducer=ee.Reducer.first(),  # same as sampling a point
        geometry=centre,
        scale=20_000,
        bestEffort=True,
    )
    dir_deg = ee.Number(dir_stat.get("wind_dir"))

    def _build_mask(direction_deg: ee.Number) -> ee.Image:
        theta = direction_deg.multiply(_DEG2RAD)
        ux, uy = theta.cos(), theta.sin()  # unit vector of the wind

        dx_dy = _lonlat_dx_dy(centre)
        dot = dx_dy.select("dx").multiply(ux).add(dx_dy.select("dy").multiply(uy))

        mask = dot.lte(0)  # downwind: non-negative dot-product
        return mask.Not() if invert else mask

    return (
        ee.Image(
            ee.Algorithms.If(
                dir_deg,  # truthy only if a numeric value exists
                _build_mask(dir_deg),
                ee.Image.constant(1),
            )
        )
        .rename("mask")
        .copyProperties(wind_layers, wind_layers.propertyNames())
    )


# ---------------------------------------------------------------------
#  Compound builders
# ---------------------------------------------------------------------
def build_mask_for_C(
    img: ee.Image,
    centre: ee.Geometry,
    p: Dict = cfg.MASK_PARAMS,
    compute_stats: bool = False,  # ← set True to print %-removed per filter
    scale: int = 20,  # resolution (m) used for the per-pixel counts
) -> tuple[ee.Image, ee.Number]:
    """Build the composite quality mask and (optionally) print how much each
    individual filter removes relative to the geom_mask base.

    Setting `compute_stats=True` adds < 2 server calls: one for the base
    pixel-count and one per filter, so runtime impact is minimal for
    debugging but should be kept False in production.
    """
    from offshore_methane.ee_utils import get_wind_layers

    # ────────────────────────────────────────────────────
    wind_layers = ee.Image(get_wind_layers(img, p["wind"]["time_window"]))

    base = geom_mask(centre, p["dist"]["local_radius_m"], inside=True).And(
        (ee.Image(downwind_mask(wind_layers, centre, invert=True))).Or(
            geom_mask(centre, p["dist"]["plume_radius_m"], inside=False)
        )
    )
    data_ok = (
        img.select(p["outlier"]["bands"]).mask().reduce(ee.Reducer.min()).rename("mask")
    )
    base = base.And(data_ok)

    # keep filters in an OrderedDict so we can iterate deterministically
    filters = {
        # "outlier": outlier_mask(img, p, valid_mask=base),
        "saturation": saturation_mask(img, p),
        "cloud": cloud_mask(img, p),
        "ndwi": ndwi_mask(img, p),
        "windspeed": windspeed_mask(wind_layers, p),
        "sga": sga_mask(img, p),
        "sgi": sgi_mask(img, p),
        "sgx_outlier": sgx_outlier(img, p),
    }

    # ── build the final mask (logical AND of everything) ─────────────────────────
    mask = base
    for m in filters.values():
        mask = mask.And(m)
    mask = mask.rename("mask")

    # ── optional debug stats ─────────────────────────────────────────────────────
    region = centre.buffer(p["dist"]["local_radius_m"]).bounds()
    total = _count(base, region, scale)  # pixels allowed by geom_mask
    if compute_stats:
        stats_reducer = ee.Reducer.percentile([10, 50, 90], ["p10", "p50", "p90"])
        sga_stats = (
            img.select("SGA").reduceRegion(stats_reducer, region, scale).getInfo()
        )
        sgi_stats = (
            img.select("SGI").reduceRegion(stats_reducer, region, scale).getInfo()
        )
        print(
            f"SGA 10, 50, 90%: \n{sga_stats['SGA_p10']:.1f}, {sga_stats['SGA_p50']:.1f}, {sga_stats['SGA_p90']:.1f}\n",
            f"SGI 10, 50, 90%: \n{sgi_stats['SGI_p10']:.2f}, {sgi_stats['SGI_p50']:.2f}, {sgi_stats['SGI_p90']:.2f}",
        )

        print("Percentage of pixels removed w.r.t. geom_mask:")
        for name, f in filters.items():
            remaining = _count(base.And(f), region, scale)
            removed_pct = (
                total.subtract(remaining).divide(total).multiply(100).getInfo()
            )
            print(f"  {name:<30s}: {removed_pct:6.2f}%")

    kept_frac = _count(mask, region, scale).divide(total)
    return mask, kept_frac


def build_mask_for_MBSP(
    img: ee.Image,
    centre: ee.Geometry,
    p: Dict = cfg.MASK_PARAMS,
    compute_stats: bool = False,  # print per-filter percentages if True
    scale: int = 20,  # pixel resolution (m) for the stats
) -> tuple[ee.Image, ee.Number]:
    """MBSP composite mask, now mirroring build_mask_for_C.

    * `base` geometry: export-radius intersected with the *actual*
      down-wind sector (no inversion like build_mask_for_C).
    * Pixel-quality filters applied in a deterministic order.
    * Optional per-filter removal statistics vs. the geom-mask base.
    """
    from offshore_methane.ee_utils import get_wind_layers

    # ────────────────────────────────────────────────────
    wind_layers = ee.Image(get_wind_layers(img, p["wind"]["time_window"]))

    # base = geometry ⨅ down-wind sector
    base = geom_mask(centre, p["dist"]["export_radius_m"], inside=True)
    # .And(
    #     downwind_mask(wind_layers, centre)
    # )
    data_ok = (
        img.select(p["outlier"]["bands"]).mask().reduce(ee.Reducer.min()).rename("mask")
    )
    base = base.And(data_ok)

    # Ordered list of pixel-quality filters (keep MBSP-specific subset)
    filters = {
        "saturation": saturation_mask(img, p),
        "cloud": cloud_mask(img, p),
        "ndwi": ndwi_mask(img, p),
        "windspeed": windspeed_mask(wind_layers, p),
        # "sgi": sgi_mask(img, p),
        "sgx_outlier": sgx_outlier(img, p),
    }

    # ── build the final mask ───────────────────────────────────────────
    mask = base
    for f in filters.values():
        mask = mask.And(f)
    mask = mask.rename("mask")

    # ── optional per-filter statistics ────────────────────────────────
    region = centre.buffer(p["dist"]["local_radius_m"]).bounds()
    total = _count(base, region, scale)  # pixels allowed by geom_mask
    if compute_stats:
        print("Percentage of pixels removed w.r.t. geom_mask:")
        for name, f in filters.items():
            remaining = _count(base.And(f), region, scale)
            removed_pct = (
                total.subtract(remaining).divide(total).multiply(100).getInfo()
            )
            print(f"  {name:<30s}: {removed_pct:6.2f}%")

    kept_frac = _count(mask, region, scale).divide(total)
    return mask, kept_frac


def view_mask(
    sid: str, centre_lon: float, centre_lat: float, compute_stats: bool = False
) -> None:
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

    centre = ee.Geometry.Point([centre_lon, centre_lat])

    # 2️⃣ True-colour backdrop (stretch a touch for contrast)
    rgb = (
        img.select(["B4", "B3", "B2"])
        .divide(10_000)  # convert DN→reflectance 0-1
        .visualize(min=0.05, max=0.35)
    )

    # Add SGA to image as a new band called "SGA"
    sga_path = f"gs://{cfg.EXPORT_PARAMS['bucket']}/{sid}/{sid}_SGA.tif"
    sga_img = ee.Image.loadGeoTIFF(sga_path)
    img = img.addBands(sga_img.rename("SGA"))

    # Add SGI to image as a new band called "SGI"
    b_vis = img.select("B2").add(img.select("B3")).add(img.select("B4")).divide(3)
    img = img.addBands(b_vis.rename("B_vis"))
    sgi = img.normalizedDifference(["B8A", "B_vis"])
    img = img.addBands(sgi.rename("SGI"))

    # 3️⃣ Build masks, turn into single-colour rasters
    mask_c, kept_c = build_mask_for_C(img, centre, cfg.MASK_PARAMS, compute_stats)
    mask_m, kept_m = build_mask_for_MBSP(img, centre, cfg.MASK_PARAMS, compute_stats)

    print(f"Mask C kept {kept_c.multiply(100).getInfo():.2f}%")
    print(f"Mask MBSP kept {kept_m.multiply(100).getInfo():.2f}%")

    mask_c = mask_c.selfMask()
    mask_m = mask_m.selfMask()

    # 4️⃣ Assemble map
    coords = centre.coordinates().getInfo()
    if coords is None:
        raise RuntimeError("Could not retrieve centre coordinates")
    lon, lat = coords
    m = geemap.Map(center=[lat, lon], zoom=14)
    m.addLayer(rgb, {}, "True colour")

    sid_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", sid)
    )
    mbsp_fp = os.path.join(sid_dir, f"{sid}_MBSP.tif")
    sga_fp = os.path.join(sid_dir, f"{sid}_SGA.tif")

    if os.path.exists(mbsp_fp):
        # red = -.1, white = 0, blue = .1
        m.add_raster(
            mbsp_fp,
            layer_name="MBSP",
            vmin=-0.1,
            vmax=0.1,
            palette=["#FF0000", "#FFFFFF", "#0000FF"],
            zoom_to_layer=False,
            nodata=-np.inf,
        )
    if os.path.exists(sga_fp):
        m.add_raster(
            sga_fp,
            layer_name="SGA",
            vmin=0,
            vmax=30,
            palette=["#FFFFFF", "#000000"],
            visible=False,
            zoom_to_layer=False,
        )
    # lon_fmt = f"{lon:.3f}"
    # lat_fmt = f"{lat:.3f}"
    # vec_fp = os.path.join(sid_dir, f"{sid}_VEC_{lon_fmt}_{lat_fmt}.geojson")

    # if os.path.exists(vec_fp):
    #     m.add_geojson(vec_fp, "Vector", {"color": "black"})

    # addLayer(object, visparams, name, display, opacity)
    m.addLayer(mask_c, {"palette": ["#FF0000"]}, "Mask C (red)", False)
    m.addLayer(mask_m, {"palette": ["#00FF00"]}, "Mask MBSP (green)", False)
    m.addLayer(centre, {"color": "yellow"}, "Target")

    m.addLayerControl()
    display(m)


def show_me(
    eid: int | None = None,
    sid: str | None = None,
    stats: bool = True,
    **filters,
) -> None:
    """
    Interactive viewer using the CSV virtual database (csv_utils).

    Parameters
    ----------
    eid : int | None
        Event id to visualize. If both `eid` and `sid` are provided, filters to
        that specific mapping.
    sid : str | None
        Sentinel-2 system:index (granule id). If provided, chooses the first
        joined row for this granule (optionally filtered by `eid`).
    stats : bool
        When True, `view_mask` prints per-filter stats.
    **filters : Any
        Optional quality filters passed through to csv_utils (e.g.
        scene_cloud_pct=..., scene_sga_range=(...), local_sga_range=(...),
        local_sgi_range=(...)).
    """
    from offshore_methane.csv_utils import df_for_event, df_for_granule, virtual_db

    target_sid: str | None = None
    target_eid: int | None = None
    lon: float | None = None
    lat: float | None = None

    if sid is not None:
        # For explicit SID lookups, do NOT apply default scene filters by default.
        # This avoids accidentally filtering out the correct mapping due to
        # missing scene metadata in granules.csv.
        f = dict(filters)
        f.setdefault("scene_cloud_pct", None)
        f.setdefault("scene_sga_range", None)
        df = df_for_granule(sid, **f)
        if eid is not None:
            df = df[df["event_id"] == int(eid)]
        if df.empty:
            print("No matching rows for given sid/eid with current filters.")
            return
        # Prefer earliest timestamp when available
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        row = df.iloc[0]
        target_sid = str(row["system_index"]).strip()
        target_eid = int(row["event_id"]) if pd.notna(row.get("event_id")) else None  # type: ignore[arg-type]
        lon, lat = float(row["lon"]), float(row["lat"])  # type: ignore[arg-type]

    elif eid is not None:
        df = df_for_event(int(eid), **filters)
        if df.empty:
            print("No matching rows for given eid with current filters.")
            return
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        row = df.iloc[0]
        target_sid = str(row["system_index"]).strip()
        target_eid = int(row["event_id"])  # type: ignore[arg-type]
        lon, lat = float(row["lon"]), float(row["lat"])  # type: ignore[arg-type]

    else:
        # fallback to first available joined row using default filters
        df = virtual_db(**filters)
        if df.empty:
            print("Virtual database is empty (no events/granules join).")
            return
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        row = df.iloc[0]
        target_sid = str(row["system_index"]).strip()
        target_eid = int(row["event_id"]) if pd.notna(row.get("event_id")) else None  # type: ignore[arg-type]
        lon, lat = float(row["lon"]), float(row["lat"])  # type: ignore[arg-type]

    if target_sid is None or lon is None or lat is None:
        print("Could not resolve a target scene and location.")
        return

    view_mask(target_sid, lon, lat, stats)
    print(f"eid: {target_eid}, sid: {target_sid}, lon: {lon}, lat: {lat}")


# %%
if __name__ == "__main__":
    show_me(sid="20170705T164319_20170705T165225_T15RXL")

# %%
