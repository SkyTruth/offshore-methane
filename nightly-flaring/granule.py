"""Process one VIIRS granule: one record per site the granule observes.

Stage 1 reads geolocation and M10 for every granule. Stage 2 reads M11-M13 only
when M10 shows a hot pixel near a site. The cloud mask is read at site pixels.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import h5py
import numpy as np
import requests
from scipy.spatial import cKDTree

import nightfire as nf
import nodd

WINDOW = 2  # pixels either side of the site pixel searched for hot pixels
MAX_SITE_PIXEL_M = 1200.0  # site must be this close to a valid pixel centre
ASSOC_M = 2000.0  # hot pixels farther than this from every site are ignored
SZA_NIGHT = 95.0  # detection only where solar zenith >= this (Elvidge 2013)
GRANULE_SECONDS = 86.0
SWIR_T_MIN, SWIR_T_MAX = 1200.0, 3000.0  # accepted range for M10+M11-only fits
RECORD_COLUMNS = [
    "satellite",
    "granule",
    "time_utc",
    "site_id",
    "row",
    "col",
    "dist_m",
    "sza",
    "cloud_mask",
    "n_hot_m10",
    "n_fitted",
    "detected",
    "temp_k",
    "area_m2",
    "rh_mw",
    "bands",
    "bytes_granule",
]

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def _read(url: str, paths: list[str]) -> tuple[list[np.ndarray], int]:
    f = nodd.HTTPRangeFile(url, _session())
    with h5py.File(f, "r") as h:
        out = [h[p][()] for p in paths]
    return out, f.bytes_fetched


def _band(url: str, band: str) -> tuple[np.ndarray, int]:
    grp = f"All_Data/VIIRS-{band}-SDR_All"
    f = nodd.HTTPRangeFile(url, _session())
    with h5py.File(f, "r") as h:
        raw = h[f"{grp}/Radiance"][()]
        factors = h[f"{grp}/RadianceFactors"][()] if raw.dtype.kind == "u" else None
    return nf.sdr_radiance(raw, factors), f.bytes_fetched


def _unit_xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    )


def process_granule(job: dict) -> list[dict]:
    """Return one dict per site observed by the granule described in ``job``.

    ``job`` holds ``satellite``, ``bucket``, ``gid``, ``start`` (datetime),
    ``keys`` (product -> S3 key) and ``sites`` (list of (site_id, lat, lon)).
    """
    base = nodd.bucket_url(job["bucket"])
    keys = job["keys"]
    nbytes = 0

    geo = "All_Data/VIIRS-MOD-GEO-TC_All"
    (lat, lon), b = _read(
        f"{base}/{keys['geo']}", [f"{geo}/Latitude", f"{geo}/Longitude"]
    )
    nbytes += b
    valid = (lat > nf.FLOAT_FILL_MAX) & (lon > nf.FLOAT_FILL_MAX)
    nrow, ncol = lat.shape

    # Site -> nearest valid pixel.
    vr, vc = np.nonzero(valid)
    tree = cKDTree(_unit_xyz(lat[vr, vc], lon[vr, vc]))
    sites = np.array(job["sites"], dtype=object)
    s_lat, s_lon = sites[:, 1].astype(float), sites[:, 2].astype(float)
    chord, idx = tree.query(_unit_xyz(s_lat, s_lon))
    dist_m = 2 * nf.EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2, 0, 1))
    rows, cols = vr[idx], vc[idx]
    inside = (
        (dist_m < MAX_SITE_PIXEL_M)
        & (rows > 0)
        & (rows < nrow - 1)
        & (cols > 0)
        & (cols < ncol - 1)
    )
    if not inside.any():
        return []
    sites, s_lat, s_lon = sites[inside], s_lat[inside], s_lon[inside]
    rows, cols, dist_m = rows[inside], cols[inside], dist_m[inside]

    mid = job["start"] + timedelta(seconds=GRANULE_SECONDS / 2)
    sza = np.full(lat.shape, np.nan)
    sza[valid] = nf.solar_zenith_deg(lat[valid], lon[valid], mid)
    night = valid & (sza >= SZA_NIGHT)
    if not night[rows, cols].any():
        return []

    # Stage 1: M10 detection, thresholds per aggregation zone.
    l10, b = _band(f"{base}/{keys['M10']}", "M10")
    nbytes += b
    usable = night & np.isfinite(l10)
    hot10 = usable & (l10 > nf.zone_thresholds(l10, usable)[None, :])

    cand = set()
    for r, c in zip(rows, cols):
        rr, cc = np.nonzero(
            hot10[r - WINDOW : r + WINDOW + 1, c - WINDOW : c + WINDOW + 1]
        )
        cand.update(zip((rr + r - WINDOW).tolist(), (cc + c - WINDOW).tolist()))

    # Stage 2: confirmation bands and Planck fit for candidate pixels.
    pixels = []
    if cand:
        l11, b11 = _band(f"{base}/{keys['M11']}", "M11")
        l12, b12 = _band(f"{base}/{keys['M12']}", "M12")
        l13, b13 = _band(f"{base}/{keys['M13']}", "M13")
        nbytes += b11 + b12 + b13
        use11 = night & np.isfinite(l11)
        hot11 = use11 & (l11 > nf.zone_thresholds(l11, use11)[None, :])
        site_tree = cKDTree(_unit_xyz(s_lat, s_lon))
        for r, c in sorted(cand):
            obs = {"M10": l10[r, c]}
            if hot11[r, c]:
                obs["M11"] = l11[r, c]
            for name, arr in (("M12", l12), ("M13", l13)):
                mu, sd = nf.local_background(arr, hot10, r, c)
                if np.isfinite(arr[r, c]) and arr[r, c] > mu + 3 * sd:
                    obs[name] = arr[r, c] - mu
            obs = {k: v for k, v in obs.items() if v < 0.98 * nf.SATURATION[k]}
            chord, j = site_tree.query(_unit_xyz([lat[r, c]], [lon[r, c]])[0])
            d = 2 * nf.EARTH_RADIUS_M * np.arcsin(min(chord / 2, 1.0))
            if d > ASSOC_M:
                continue
            px = {"site": int(j), "row": r, "col": c, "bands": sorted(obs)}
            if len(obs) >= 2 and "M10" in obs:
                t, esf, ssr, used = nf.fit_planck(list(obs), list(obs.values()))
                swir_only = not {"M12", "M13"} & set(used)
                if swir_only and not (SWIR_T_MIN <= t <= SWIR_T_MAX):
                    # M10+M11-only fits are unstable (Elvidge 2023 s5): pin T.
                    t = nf.FLARE_T_DEFAULT
                    esf = obs["M10"] / float(nf.planck(nf.BAND_UM["M10"], t))
                    used = [*used, "T1810"]
                area = nf.pixel_area_m2(lat, lon, r, c)
                px.update(
                    temp_k=t,
                    esf=esf,
                    ssr=ssr,
                    bands=used,
                    area_m2=area,
                    rh_mw=nf.radiant_heat_mw(t, esf, area),
                )
            pixels.append(px)

    # Cloud mask at site pixels (0 clear, 1 probably clear, 2/3 cloudy).
    cloud = np.full(len(sites), -1, dtype=int)
    if keys.get("cloud"):
        f = nodd.HTTPRangeFile(f"{base}/{keys['cloud']}", _session())
        with h5py.File(f, "r") as h:
            ds = h["CloudMask"]
            if ds.shape == lat.shape:
                cloud = np.array([int(ds[r, c]) for r, c in zip(rows, cols)])
        nbytes += f.bytes_fetched

    records = []
    for k in range(len(sites)):
        r, c = int(rows[k]), int(cols[k])
        mine = [p for p in pixels if p["site"] == k]
        fitted = [p for p in mine if "rh_mw" in p]
        rec = {
            "satellite": job["satellite"],
            "granule": job["gid"],
            "time_utc": (
                job["start"] + timedelta(seconds=GRANULE_SECONDS * r / nrow)
            ).isoformat(timespec="seconds"),
            "site_id": int(sites[k][0]),
            "row": r,
            "col": c,
            "dist_m": round(float(dist_m[k]), 1),
            "sza": round(float(sza[r, c]), 2),
            "cloud_mask": int(cloud[k]),
            "n_hot_m10": len(mine),
            "n_fitted": len(fitted),
            "detected": bool(fitted),
            "temp_k": np.nan,
            "area_m2": np.nan,
            "rh_mw": 0.0,
            "bands": "",
            "bytes_granule": nbytes,
        }
        if fitted:
            top = max(fitted, key=lambda p: p["rh_mw"])
            scan = top["row"] // nf.M_SCAN_LINES
            keep = [
                p
                for p in fitted
                if p["row"] // nf.M_SCAN_LINES == scan
                and p["rh_mw"] >= 0.75 * top["rh_mw"]
            ]
            rec.update(
                temp_k=round(top["temp_k"], 1),
                area_m2=round(top["esf"] * top["area_m2"], 3),
                rh_mw=round(sum(p["rh_mw"] for p in keep), 4),
                bands="+".join(top["bands"]),
            )
        records.append(rec)
    return records


def granule_start(gid: str) -> datetime:
    """'d20261101_t0200577' -> datetime."""
    return nodd._hhmmsss(gid[1:9], gid[11:18])
