"""Process one VIIRS granule: one record per site the granule observes.

Stage 1 reads geolocation and M10 for every granule. Stage 2 reads M7/M8/M11-M13
only when M10 shows a hot pixel near a site. The cloud mask is read at site pixels.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import h5py
import numpy as np
import requests
from scipy.ndimage import label
from scipy.spatial import cKDTree

import nightfire as nf
import nodd

WINDOW = 2  # pixels either side of the site pixel searched for hot pixels
MAX_SITE_PIXEL_M = 1200.0  # site must be this close to a valid pixel centre
ASSOC_M = 2000.0  # hot pixels farther than this from every site are ignored
SZA_NIGHT = 95.0  # detection only where solar zenith >= this (Elvidge 2013)
GRANULE_SECONDS = 86.0
# JPSS VIIRS SDR DD Part 6, QF1: bits 2-3, 0 none / 1 some / 2 all.
QF1_SATURATION_MASK = 0b00001100
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
    "cloud_clear_neighbors",
    "n_hot_m10",
    "n_fitted",
    "n_clusters",
    "detected",
    "temp_k",
    "area_m2",
    "rh_mw",
    "bands",
    "fit_flag",
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


def _band(url: str, band: str) -> tuple[np.ndarray, np.ndarray, int]:
    grp = f"All_Data/VIIRS-{band}-SDR_All"
    f = nodd.HTTPRangeFile(url, _session())
    with h5py.File(f, "r") as h:
        raw = h[f"{grp}/Radiance"][()]
        factors = h[f"{grp}/RadianceFactors"][()] if raw.dtype.kind == "u" else None
        qf = h[f"{grp}/QF1_VIIRSMBANDSDR"][()]
    return nf.sdr_radiance(raw, factors), qf, f.bytes_fetched


def _unit_xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    )


def site_pixels(
    lat: np.ndarray, lon: np.ndarray, l10: np.ndarray, sites: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Nearest finite-radiance geolocated pixel for each site, or None."""
    valid = np.isfinite(l10) & np.isfinite(lat) & np.isfinite(lon)
    valid &= (lat > nf.FLOAT_FILL_MAX) & (lon > nf.FLOAT_FILL_MAX)
    vr, vc = np.nonzero(valid)
    if not len(vr):
        return None
    tree = cKDTree(_unit_xyz(lat[vr, vc], lon[vr, vc]))
    chord, idx = tree.query(
        _unit_xyz(sites[:, 1].astype(float), sites[:, 2].astype(float))
    )
    dist = 2 * nf.EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2, 0, 1))
    return vr[idx], vc[idx], dist


def candidate_clusters(
    hot10: np.ndarray, l10: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> list[dict]:
    """Scan-separated 8-connected clusters touching clipped site windows.

    Reject a cluster if any pixel within two of its M10 peak in the same scan
    is brighter, including pixels outside the hot mask.
    """
    labels = np.zeros(hot10.shape, dtype=np.int32)
    offset = 0
    for r0 in range(0, len(hot10), nf.M_SCAN_LINES):
        block, count = label(hot10[r0 : r0 + nf.M_SCAN_LINES], np.ones((3, 3)))
        labels[r0 : r0 + nf.M_SCAN_LINES] = np.where(block, block + offset, 0)
        offset += count
    selected = set()
    for r, c in zip(rows, cols):
        selected.update(
            np.unique(
                labels[
                    max(0, r - WINDOW) : r + WINDOW + 1,
                    max(0, c - WINDOW) : c + WINDOW + 1,
                ]
            )
        )
    clusters = []
    # Group labelled coordinates once rather than scanning the array per cluster.
    rr, cc = np.nonzero(labels)
    ids = labels[rr, cc]
    order = np.argsort(ids, kind="stable")
    for group in np.split(order, np.flatnonzero(np.diff(ids[order])) + 1):
        if not len(group) or ids[group[0]] not in selected:
            continue
        pixels = list(zip(rr[group].tolist(), cc[group].tolist()))
        peak = max(pixels, key=lambda rc: l10[rc])
        r, c = peak
        scan0 = r // nf.M_SCAN_LINES * nf.M_SCAN_LINES
        window = l10[
            max(scan0, r - WINDOW) : min(scan0 + nf.M_SCAN_LINES, r + WINDOW + 1),
            max(0, c - WINDOW) : c + WINDOW + 1,
        ]
        if np.any(window > l10[peak]):
            continue
        clusters.append({"pixels": pixels, "peak": peak})
    return clusters


def summarize_clusters(
    clusters: list[dict],
    pixels: dict,
    lat: np.ndarray,
    lon: np.ndarray,
    sites: np.ndarray,
) -> list[dict]:
    """Assign confirmed clusters once by peak location; compute cluster RH75%."""
    tree = cKDTree(_unit_xyz(sites[:, 1].astype(float), sites[:, 2].astype(float)))
    result = []
    for cluster in clusters:
        peak = pixels.get(cluster["peak"], {})
        if "rh_mw" not in peak:
            continue
        r, c = cluster["peak"]
        chord, j = tree.query(_unit_xyz([lat[r, c]], [lon[r, c]])[0])
        dist = 2 * nf.EARTH_RADIUS_M * np.arcsin(min(chord / 2, 1.0))
        if dist > ASSOC_M:
            continue
        fitted = [pixels[rc] for rc in cluster["pixels"] if "rh_mw" in pixels[rc]]
        maximum = max(p["rh_mw"] for p in fitted)
        result.append(
            dict(
                site=int(j),
                peak=peak,
                n_hot=len(cluster["pixels"]),
                n_fitted=len(fitted),
                rh_mw=sum(p["rh_mw"] for p in fitted if p["rh_mw"] >= 0.75 * maximum),
            )
        )
    return result


def site_cluster_summary(clusters: list[dict], site: int) -> dict:
    """Use the maximum cluster RH75%, absorbing copies in adjacent scans."""
    mine = [p for p in clusters if p["site"] == site]
    out = dict(
        n_clusters=len(mine),
        n_hot_m10=sum(p["n_hot"] for p in mine),
        n_fitted=sum(p["n_fitted"] for p in mine),
        detected=bool(mine),
    )
    if mine:
        best = max(mine, key=lambda p: p["rh_mw"])
        peak = best["peak"]
        out.update(
            temp_k=round(peak["temp_k"], 1),
            area_m2=round(peak["esf"] * peak["area_m2"], 3),
            rh_mw=round(best["rh_mw"], 4),
            bands="+".join(peak["bands"]),
            fit_flag=peak.get("fit_flag", "ok"),
        )
    return out


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
    l10, q10, b = _band(f"{base}/{keys['M10']}", "M10")
    nbytes += b
    valid = np.isfinite(l10) & (lat > nf.FLOAT_FILL_MAX) & (lon > nf.FLOAT_FILL_MAX)
    nrow, ncol = lat.shape

    # Site -> nearest valid pixel.
    if not valid.any():
        return []
    sites = np.array(job["sites"], dtype=object)
    rows, cols, dist_m = site_pixels(lat, lon, l10, sites)
    inside = (
        (dist_m < MAX_SITE_PIXEL_M)
        & (rows > 0)
        & (rows < nrow - 1)
        & (cols > 0)
        & (cols < ncol - 1)
    )
    if not inside.any():
        return []
    sites = sites[inside]
    rows, cols, dist_m = rows[inside], cols[inside], dist_m[inside]

    mid = job["start"] + timedelta(seconds=GRANULE_SECONDS / 2)
    sza = np.full(lat.shape, np.nan)
    sza[valid] = nf.solar_zenith_deg(lat[valid], lon[valid], mid)
    night = valid & (sza >= SZA_NIGHT)
    if not night[rows, cols].any():
        return []

    # Stage 1: M10 detection, thresholds per aggregation zone.
    usable = night & np.isfinite(l10)
    thr10, sd10 = nf.zone_statistics(l10, usable)
    hot10 = usable & (l10 > thr10[None, :])

    clusters = candidate_clusters(hot10, l10, rows, cols)
    cand = {rc for cluster in clusters for rc in cluster["pixels"]}

    # Stage 2: confirmation bands and Planck fit for candidate pixels.
    pixels = {}
    if cand:
        arrays, flags = {"M10": l10}, {"M10": q10}
        thresholds, noise = {}, {"M10": sd10}
        for name in ("M7", "M8", "M11", "M12", "M13"):
            if not keys.get(name):
                continue
            try:
                arr, qf, count = _band(f"{base}/{keys[name]}", name)
            except requests.HTTPError as exc:
                if name in ("M7", "M8") and exc.response.status_code == 404:
                    continue
                raise
            arrays[name], flags[name] = arr, qf
            nbytes += count
            if name in ("M7", "M8", "M11"):
                thresholds[name], noise[name] = nf.zone_statistics(
                    arr, night & np.isfinite(arr)
                )
        for r, c in sorted(cand):
            obs, sigma = {"M10": l10[r, c]}, {"M10": sd10[c]}
            for name, arr in arrays.items():
                if name in thresholds and arr[r, c] > thresholds[name][c]:
                    obs[name], sigma[name] = arr[r, c], noise[name][c]
                elif name in ("M12", "M13"):
                    mu, sd = nf.local_background(arr, hot10, r, c)
                    if np.isfinite(arr[r, c]) and arr[r, c] > mu + 3 * sd:
                        obs[name], sigma[name] = arr[r, c] - mu, sd
            obs = {
                k: v
                for k, v in obs.items()
                if arrays[k][r, c] < 0.98 * nf.SATURATION[k]
                and not (flags[k][r, c] & QF1_SATURATION_MASK)
            }
            px = {"row": r, "col": c, "bands": sorted(obs)}
            if len(obs) >= 2:
                t, esf, chi, used, flag = nf.fit_planck(
                    list(obs), list(obs.values()), [sigma[k] for k in obs]
                )
                if flag != "unfitted":
                    area = nf.pixel_area_m2(lat, lon, r, c)
                    px.update(
                        temp_k=t,
                        esf=esf,
                        chi2=chi,
                        bands=used,
                        fit_flag=flag,
                        area_m2=area,
                        rh_mw=nf.radiant_heat_mw(t, esf, area),
                    )
            pixels[r, c] = px

    assigned = summarize_clusters(clusters, pixels, lat, lon, sites)

    # Cloud mask at site pixels (0 clear, 1 probably clear, 2/3 cloudy).
    clear_neighbors = np.zeros(len(sites), dtype=int)
    cloud = np.full(len(sites), -1, dtype=int)
    if keys.get("cloud"):
        f = nodd.HTTPRangeFile(f"{base}/{keys['cloud']}", _session())
        with h5py.File(f, "r") as h:
            ds = h["CloudMask"]
            if ds.shape == lat.shape:
                cloud = np.array([int(ds[r, c]) for r, c in zip(rows, cols)])
                for k, (r, c) in enumerate(zip(rows, cols)):
                    window = ds[r - 1 : r + 2, c - 1 : c + 2]
                    clear_neighbors[k] = np.isin(window, [0, 1]).sum() - int(
                        cloud[k] in (0, 1)
                    )
        nbytes += f.bytes_fetched

    records = []
    for k in range(len(sites)):
        r, c = int(rows[k]), int(cols[k])
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
            "cloud_clear_neighbors": int(clear_neighbors[k]),
            "temp_k": np.nan,
            "area_m2": np.nan,
            "rh_mw": 0.0,
            "bands": "",
            "fit_flag": "",
            "bytes_granule": nbytes,
        }
        rec.update(site_cluster_summary(assigned, k))
        records.append(rec)
    return records


def granule_start(gid: str) -> datetime:
    """'d20261101_t0200577' -> datetime."""
    return nodd._hhmmsss(gid[1:9], gid[11:18])
