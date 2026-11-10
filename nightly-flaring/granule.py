"""Process one VIIRS granule: one record per site the granule observes.

Stage 1 reads geolocation and M10 for every granule. Stage 2 reads M7/M8/M11-M13
only when M10 shows a hot pixel near a site. The cloud mask is read at site pixels.
"""

from __future__ import annotations

import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timedelta

import h5py
import numpy as np
import requests
from scipy.ndimage import label
from scipy.spatial import cKDTree

import nightfire as nf
import nodd
import orbit
import speedup

WINDOW = 2  # pixels either side of the site pixel searched for hot pixels
MAX_SITE_PIXEL_M = 1200.0  # site must be this close to a valid pixel centre
ASSOC_M = 2000.0  # hot pixels farther than this from every site are ignored
BOWTIE_MERGE_PIXELS = (
    1.5  # adjacent-scan distance limit, with column/RH/group guards (D36)
)
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
    "n_band_rule_changed",
    "n_clusters",
    "n_sources",
    "rh_peak_mw",
    "n_rh75",
    "n_rh_cap_excluded",
    "n_components_split",
    "split_component_ids",
    "n_copies_merged",
    "peak_row",
    "peak_col",
    "peak_lat",
    "peak_lon",
    "peak_dist_m",
    "detected",
    "temp_k",
    "area_m2",
    "rh_mw",
    "bands",
    "fit_flag",
    "bytes_granule",
]

COPY_COLUMNS = """satellite granule time_utc scan component_id peak_row peak_col
peak_lat peak_lon footprint_m2 m10_peak prominence_ratio rh75_mw rh_peak_mw
n_rh75 n_rh_cap_excluded temp_k area_m2 bands fit_flag t_fit_rejected
cloud_mask_peak clear_neighbors_peak group_id group_size pair_distance_px
pair_dcol pair_drow pair_rh_ratio group_rh_mw group_max_rh_mw centroid_lat
centroid_lon assigned_site site_dist_m band_rule_changed n_hot
n_suppressed_maxima suppressed_max_m10 suppressed_saddle_ratio""".split()


def peak_saddle(
    pixels: list[tuple], values: np.ndarray, start: tuple, targets: set[tuple]
) -> float:
    """Highest bottleneck level on an 8-connected path to a stronger peak."""
    import heapq

    allowed = set(pixels)
    queue = [(-float(values[start]), start)]
    seen = set()
    while queue:
        negative, rc = heapq.heappop(queue)
        if rc in seen:
            continue
        seen.add(rc)
        if rc in targets:
            return -negative
        r, c = rc
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nxt = r + dr, c + dc
                if nxt in allowed and nxt not in seen:
                    heapq.heappush(queue, (-min(-negative, float(values[nxt])), nxt))
    return 0.0


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
        peaks = []
        for r, c in sorted(pixels, key=lambda rc: (-l10[rc], rc)):
            scan0 = r // nf.M_SCAN_LINES * nf.M_SCAN_LINES
            window = l10[
                max(scan0, r - WINDOW) : min(scan0 + nf.M_SCAN_LINES, r + WINDOW + 1),
                max(0, c - WINDOW) : c + WINDOW + 1,
            ]
            if np.any(window > l10[r, c]):
                continue
            # Collapse a flat summit deterministically within the glow window.
            if any(max(abs(r - pr), abs(c - pc)) <= WINDOW for pr, pc in peaks):
                continue
            peaks.append((r, c))
        if not peaks:
            continue
        retained, prominence, suppressed = [peaks[0]], [0.0], []
        for peak in peaks[1:]:
            ratio = peak_saddle(pixels, l10, peak, set(retained)) / l10[peak]
            if ratio <= 0.5:
                retained.append(peak)
                prominence.append(float(ratio))
            else:
                suppressed.append((float(l10[peak]), float(ratio)))
        peaks = retained
        # Suppressed maxima are exported per component so the saddle rule can
        # be audited offline (D45).
        top = max(suppressed) if suppressed else (np.nan, np.nan)
        groups = [[] for _ in peaks]
        for rc in pixels:
            j = min(
                range(len(peaks)),
                key=lambda j: sum((a - b) ** 2 for a, b in zip(rc, peaks[j])),
            )
            groups[j].append(rc)
        for j, peak in enumerate(peaks):
            clusters.append(
                {
                    "component_id": int(ids[group[0]]),
                    "prominence_ratio": prominence[j],
                    "m10_peak": float(l10[peak]),
                    "n_suppressed_maxima": len(suppressed),
                    "suppressed_max_m10": top[0],
                    "suppressed_saddle_ratio": top[1],
                    "pixels": groups[j],
                    "peak": peak,
                    "split_component": int(ids[group[0]]) if len(peaks) > 1 else None,
                }
            )
    return clusters


def summarize_clusters(
    clusters: list[dict],
    pixels: dict,
    lat: np.ndarray,
    lon: np.ndarray,
    sites: np.ndarray,
    copy_rows: list[dict] | None = None,
) -> list[dict]:
    """Compute peak-anchored RH75%, merge scan copies, then assign sources."""
    tree = cKDTree(_unit_xyz(sites[:, 1].astype(float), sites[:, 2].astype(float)))
    copies = []
    for cluster in clusters:
        peak = pixels.get(cluster["peak"], {})
        if "rh_mw" not in peak or peak.get("fit_flag") == "unfitted":
            continue
        r, c = cluster["peak"]
        fitted = [
            pixels[rc]
            for rc in cluster["pixels"]
            if "rh_mw" in pixels.get(rc, {})
            and pixels[rc].get("fit_flag") != "unfitted"
        ]
        peak_rh = peak["rh_mw"]
        adjacent = [
            pixels[rc]
            for rc in cluster["pixels"]
            if max(abs(rc[0] - r), abs(rc[1] - c)) <= 1
            and rc[0] // nf.M_SCAN_LINES == r // nf.M_SCAN_LINES
            and "rh_mw" in pixels.get(rc, {})
            and pixels[rc].get("fit_flag") != "unfitted"
        ]
        selected = [
            p for p in adjacent if 0.75 * peak_rh <= p["rh_mw"] <= 1.5 * peak_rh
        ]
        copies.append(
            dict(
                component_id=cluster.get("component_id"),
                prominence_ratio=cluster.get("prominence_ratio", 0.0),
                m10_peak=cluster.get("m10_peak", np.nan),
                n_suppressed_maxima=cluster.get("n_suppressed_maxima", 0),
                suppressed_max_m10=cluster.get("suppressed_max_m10", np.nan),
                suppressed_saddle_ratio=cluster.get("suppressed_saddle_ratio", np.nan),
                peak=peak,
                peak_row=r,
                peak_col=c,
                peak_lat=float(lat[r, c]),
                peak_lon=float(lon[r, c]),
                footprint=nf.pixel_area_m2(lat, lon, r, c),
                n_hot=len(cluster["pixels"]),
                n_fitted=len(fitted),
                rh_mw=sum(p["rh_mw"] for p in selected),
                rh_peak_mw=peak_rh,
                n_rh75=len(selected),
                n_rh_cap_excluded=sum(p["rh_mw"] > 1.5 * peak_rh for p in adjacent),
                split_components={cluster["split_component"]}
                if cluster.get("split_component") is not None
                else set(),
            )
        )
    result = merge_scan_copies(copies, tree)
    if copy_rows is not None:
        copy_rows.extend(copies)
    return result


def merge_scan_copies(copies: list[dict], tree: cKDTree) -> list[dict]:
    """Match closest qualifying pairs; forbid duplicate scans within a group."""
    parent = list(range(len(copies)))
    scans = [{p["peak_row"] // nf.M_SCAN_LINES} for p in copies]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i, a in enumerate(copies):
        a["pairs"] = []
        for j in range(i):
            b = copies[j]
            if abs(a["peak_row"] // 16 - b["peak_row"] // 16) != 1:
                continue
            dcol = abs(a["peak_col"] - b["peak_col"])
            ratio = max(a["rh_mw"], b["rh_mw"]) / min(a["rh_mw"], b["rh_mw"])
            distance = float(
                nf.haversine_m(
                    a["peak_lat"], a["peak_lon"], b["peak_lat"], b["peak_lon"]
                )
            )
            scale = np.sqrt(min(a["footprint"], b["footprint"]))
            px = distance / scale if scale > 0 else np.inf
            if dcol <= 1 and px < BOWTIE_MERGE_PIXELS and ratio < 3:
                pairs.append(
                    (
                        distance,
                        i,
                        j,
                        px,
                        dcol,
                        abs(a["peak_row"] - b["peak_row"]),
                        ratio,
                    )
                )
    for _, i, j, px, dc, dr, ratio in sorted(pairs):
        x, y = find(i), find(j)
        if scans[x] & scans[y]:
            continue
        parent[x] = y
        scans[y] |= scans[x]
        for k in (i, j):
            copies[k]["pairs"].append((px, dc, dr, ratio))
    groups = {}
    for i, copy in enumerate(copies):
        groups.setdefault(find(i), []).append(copy)
    result = []
    for group_id, group in enumerate(groups.values()):
        best = max(group, key=lambda p: p["rh_mw"]).copy()
        weights = np.array([p["rh_mw"] for p in group])
        xyz = np.average(
            _unit_xyz([p["peak_lat"] for p in group], [p["peak_lon"] for p in group]),
            axis=0,
            weights=weights,
        )
        xyz /= np.linalg.norm(xyz)
        la = float(np.degrees(np.arcsin(xyz[2])))
        lo = float(np.degrees(np.arctan2(xyz[1], xyz[0])))
        chord, j = tree.query(xyz)
        dist = float(2 * nf.EARTH_RADIUS_M * np.arcsin(min(chord / 2, 1.0)))
        rh = float(weights.mean())
        for copy in group:
            copy.update(
                group_id=group_id,
                group_size=len(group),
                group_rh_mw=rh,
                group_max_rh_mw=float(weights.max()),
                centroid_lat=la,
                centroid_lon=lo,
                site=int(j) if dist <= ASSOC_M else None,
                site_dist_m=dist,
            )
        if dist > ASSOC_M:
            continue
        best.update(
            site=int(j),
            peak_dist_m=dist,
            n_clusters=len(group),
            n_copies_merged=len(group) - 1,
            rh_mw=rh,
            centroid_lat=la,
            centroid_lon=lo,
        )
        best["split_components"] = set().union(
            *(p.get("split_components", set()) for p in group)
        )
        for key in ("n_hot", "n_fitted", "n_rh_cap_excluded"):
            best[key] = sum(p[key] for p in group)
        result.append(best)
    return result


def site_cluster_summary(clusters: list[dict], site: int) -> dict:
    """Sum distinct sources; expose the strongest source's fit and location."""
    mine = [p for p in clusters if p["site"] == site]
    out = dict(
        n_clusters=sum(p.get("n_clusters", 1) for p in mine),
        n_sources=len(mine),
        n_hot_m10=sum(p["n_hot"] for p in mine),
        n_fitted=sum(p["n_fitted"] for p in mine),
        detected=bool(mine),
        n_rh75=sum(p.get("n_rh75", 0) for p in mine),
        n_rh_cap_excluded=sum(p.get("n_rh_cap_excluded", 0) for p in mine),
        n_components_split=len(
            set().union(*(p.get("split_components", set()) for p in mine))
        ),
        split_component_ids=",".join(
            f"c{i}"
            for i in sorted(
                set().union(*(p.get("split_components", set()) for p in mine))
            )
        ),
        n_copies_merged=sum(p.get("n_copies_merged", 0) for p in mine),
    )
    if mine:
        best = max(mine, key=lambda p: p["rh_mw"])
        peak = best["peak"]
        out.update(
            temp_k=round(peak["temp_k"], 1),
            area_m2=round(peak["esf"] * peak["area_m2"], 3),
            rh_mw=round(sum(p["rh_mw"] for p in mine), 4),
            rh_peak_mw=best.get("rh_peak_mw", peak["rh_mw"]),
            bands="+".join(peak["bands"]),
            fit_flag=peak.get("fit_flag", "ok"),
        )
        for key in ("peak_row", "peak_col", "peak_lat", "peak_lon", "peak_dist_m"):
            out[key] = best.get(key, np.nan)
    return out


def process_granule(job: dict) -> tuple[list[dict], list[dict]]:
    """Return site records and confirmed-copy records for one granule.

    ``job`` holds ``satellite``, ``bucket``, ``gid``, ``start`` (datetime),
    ``keys`` (product -> S3 key) and ``sites`` (list of (site_id, lat, lon)).
    """
    with ExitStack() as stack:
        return _process_granule(job, stack)


def process_job(job: dict) -> dict:
    """Worker envelope: account for empty results and failed attempts (T0)."""
    global _SESSION
    original = _session()
    counter = nodd.CountingSession(original)
    _SESSION = counter
    metrics = {}
    job = dict(job, _metrics=metrics)
    wall, cpu = time.perf_counter(), time.process_time()
    result = dict(records=[], copies=[], error=None)
    try:
        result["records"], result["copies"] = process_granule(job)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        _SESSION = original
    result["stats"] = dict(
        bytes_fetched=counter.bytes_fetched,
        requests=counter.requests,
        heads=counter.heads,
        wall_s=time.perf_counter() - wall,
        cpu_s=time.process_time() - cpu,
        accounting_complete=True,
        **metrics,
    )
    return result


def _process_granule(job: dict, stack: ExitStack) -> tuple[list[dict], list[dict]]:
    base = nodd.bucket_url(job["bucket"])
    keys = job["keys"]
    nbytes = 0
    opts = speedup.options(job)
    metrics = job.get("_metrics", {})

    def full_fallback(reason):
        metrics["geo_mode"] = "fallback_" + reason
        config = dict(
            job.get("speedup", {}),
            windowed_geo=False,
            header_prefilter=False,
            orbit_whitelist=False,
        )
        return _process_granule(dict(job, speedup=config), stack)

    prediction = None
    if opts.orbit_whitelist or opts.windowed_geo:
        try:
            if not opts.tle_dir:
                raise ValueError("no local TLE directory")
            prediction = orbit.predict(job, opts.tle_dir)
            metrics["orbit"] = "predicted"
            if opts.orbit_whitelist and not prediction["eligible"].any():
                metrics.update(skipped=True, orbit="outside_swath")
                return [], []
        except (ValueError, OSError, KeyError, RuntimeError):
            metrics["orbit"] = "fallback_tle"

    geo_file = geo_h = None
    attrs = {}
    if opts.header_prefilter and speedup.solar_skip(job):
        metrics.update(prefilter="site_solar_bound", skipped=True)
        return [], []
    if opts.header_prefilter or opts.windowed_geo:
        geo_file = stack.enter_context(
            nodd.HTTPRangeFile(f"{base}/{keys['geo']}", _session(), opts.block_size)
        )
        geo_h = stack.enter_context(h5py.File(geo_file, "r"))
        attrs = speedup.header(geo_h)
    if opts.header_prefilter:
        skip, reason = speedup.prefilter(job, opts, attrs)
        metrics["prefilter"] = reason
        if skip:
            metrics["skipped"] = True
            return [], []

    l10 = q10 = None
    sparse = sparse_file = sparse_h = nearest = None
    if opts.windowed_geo:
        l10, q10, b = _band(f"{base}/{keys['M10']}", "M10")
        nbytes += b
        if prediction is not None and l10.shape != (768, 3200):
            try:
                prediction = orbit.predict(job, opts.tle_dir, l10.shape)
            except (ValueError, OSError, KeyError, RuntimeError):
                prediction = None
        certified, edge_sza = speedup.night_certificate(
            attrs, job["start"], opts.night_margin_deg
        )
        metrics["night_edge_min_sza"] = edge_sza
        metrics["geo_mode"] = "fallback_prediction"
        if not keys.get("cloud"):
            metrics["geo_mode"] = "fallback_missing_jrr"
        elif job["start"].year < 2026:
            # Historic sample has different GEO/JRR fill sentinels; keep full GEO
            # until that product generation has broader valid-pixel evidence.
            metrics["geo_mode"] = "fallback_unvalidated_jrr_epoch"
        elif not certified:
            metrics["geo_mode"] = "fallback_night_mask"
        elif prediction is not None:
            distances = speedup.footprint_distances(job, attrs)
            excluded = (
                distances > opts.footprint_buffer_km if distances is not None else None
            )
            # Sites predicted beyond the image by more than the margins cannot
            # have a pixel within MAX_SITE_PIXEL_M; leave them out (D64).
            far = speedup.far_outside(prediction, l10.shape, opts)
            if not speedup.margins_cover_error(opts):
                far[:] = False
            excluded = far if excluded is None else (excluded | far)
            metrics["geo_far_outside"] = int(far.sum())
            if excluded.all():
                # No site can be observed: the full read would return no rows.
                metrics.update(geo_mode="all_sites_excluded", skipped=True)
                return [], []
            else:
                try:
                    sparse_file = stack.enter_context(
                        nodd.HTTPRangeFile(
                            f"{base}/{keys['cloud']}", _session(), opts.block_size
                        )
                    )
                    sparse_h = stack.enter_context(h5py.File(sparse_file, "r"))
                    sparse = speedup.SparseGeo(sparse_h, l10.shape)
                    nearest = sparse.nearest(
                        l10,
                        np.array(job["sites"], dtype=object),
                        prediction,
                        opts,
                        excluded,
                    )
                    if nearest is None:
                        metrics["geo_mode"] = "fallback_" + sparse.reason
                        sparse = None
                    else:
                        metrics["geo_mode"] = "windowed"
                        metrics["geo_widened"] = sparse.widened
                        metrics["geo_certified_misses"] = sparse.misses
                except (KeyError, ValueError, OSError, requests.HTTPError):
                    metrics["geo_mode"] = "fallback_jrr_layout_or_read"
                    sparse = None

    geo = "All_Data/VIIRS-MOD-GEO-TC_All"
    if sparse is None:
        if geo_h is not None:
            lat, lon = geo_h[f"{geo}/Latitude"][()], geo_h[f"{geo}/Longitude"][()]
            nbytes += geo_file.bytes_fetched
        else:
            (lat, lon), b = _read(
                f"{base}/{keys['geo']}", [f"{geo}/Latitude", f"{geo}/Longitude"]
            )
            nbytes += b
    else:
        lat, lon = sparse.lat, sparse.lon
    if l10 is None:
        l10, q10, b = _band(f"{base}/{keys['M10']}", "M10")
        nbytes += b
    valid = np.isfinite(l10) & (lat > nf.FLOAT_FILL_MAX) & (lon > nf.FLOAT_FILL_MAX)
    if sparse is not None:
        valid = np.isfinite(l10) & sparse.valid
    nrow, ncol = lat.shape

    # Site -> nearest valid pixel.
    if not valid.any():
        return [], []
    sites = np.array(job["sites"], dtype=object)
    rows, cols, dist_m = (
        nearest if sparse is not None else site_pixels(lat, lon, l10, sites)
    )
    inside = (
        (dist_m < MAX_SITE_PIXEL_M)
        & (rows > 0)
        & (rows < nrow - 1)
        & (cols > 0)
        & (cols < ncol - 1)
    )
    if not inside.any():
        return [], []
    sites = sites[inside]
    rows, cols, dist_m = rows[inside], cols[inside], dist_m[inside]
    if sparse is not None:
        sparse.ensure(zip(rows, cols))
        failure = sparse.check(
            l10, job["start"] + timedelta(seconds=GRANULE_SECONDS / 2)
        )
        if failure:
            return full_fallback(failure)

    mid = job["start"] + timedelta(seconds=GRANULE_SECONDS / 2)
    sza = np.full(lat.shape, np.nan)
    located = valid if sparse is None else valid & np.isfinite(lat) & np.isfinite(lon)
    sza[located] = nf.solar_zenith_deg(lat[located], lon[located], mid)
    night = valid & (sza >= SZA_NIGHT) if sparse is None else valid.copy()
    if not night[rows, cols].any():
        return [], []

    # Stage 1: M10 detection, thresholds per aggregation zone.
    usable = night & np.isfinite(l10)
    thr10, sd10 = nf.zone_statistics(l10, usable)
    hot10 = usable & (l10 > thr10[None, :])

    clusters = candidate_clusters(hot10, l10, rows, cols)
    cand = {rc for cluster in clusters for rc in cluster["pixels"]}
    if sparse is not None:
        # Covers Planck footprint neighbours, cluster/peak positions, merge
        # distances, RH centroids and all downstream lat/lon uses.
        sparse.ensure(cand)
        failure = sparse.check(l10, mid)
        if failure:
            return full_fallback(failure)

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
                diagnostic = {}
                t, esf, chi, used, flag = nf.fit_planck(
                    list(obs),
                    list(obs.values()),
                    [sigma[k] for k in obs],
                    diagnostics=diagnostic,
                )
                if flag != "unfitted":
                    area = nf.pixel_area_m2(lat, lon, r, c)
                    px.update(
                        band_rule_changed=diagnostic.get("band_rule_changed", False),
                        t_fit_rejected=diagnostic.get("t_fit_rejected", np.nan),
                        temp_k=t,
                        esf=esf,
                        chi2=chi,
                        bands=used,
                        fit_flag=flag,
                        area_m2=area,
                        rh_mw=nf.radiant_heat_mw(t, esf, area),
                    )
            pixels[r, c] = px

    copies = []
    assigned = summarize_clusters(clusters, pixels, lat, lon, sites, copies)
    copy_cloud = {}

    # Cloud mask at site pixels (0 clear, 1 probably clear, 2/3 cloudy).
    clear_neighbors = np.zeros(len(sites), dtype=int)
    cloud = np.full(len(sites), -1, dtype=int)
    reduced = opts.tier2_detections_only and not speedup.tier1(job, opts)
    metrics["reduced"] = reduced
    if keys.get("cloud") and (not reduced or copies):
        if sparse_h is not None:
            h = sparse_h
            f = None
        else:
            f = stack.enter_context(
                nodd.HTTPRangeFile(f"{base}/{keys['cloud']}", _session())
            )
            h = stack.enter_context(h5py.File(f, "r"))
        if h is not None:
            ds = h["CloudMask"]
            if ds.shape == lat.shape:
                for copy in copies:
                    r, c = copy["peak_row"], copy["peak_col"]
                    cm = int(ds[r, c])
                    window = ds[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
                    copy_cloud[r, c] = (
                        cm,
                        int(np.isin(window, [0, 1]).sum()) - int(cm in (0, 1)),
                    )
                if not reduced:
                    cloud = np.array([int(ds[r, c]) for r, c in zip(rows, cols)])
                    for k, (r, c) in enumerate(zip(rows, cols)):
                        window = ds[r - 1 : r + 2, c - 1 : c + 2]
                        clear_neighbors[k] = np.isin(window, [0, 1]).sum() - int(
                            cloud[k] in (0, 1)
                        )
        if f is not None:
            nbytes += f.bytes_fetched

    if sparse_file is not None:
        nbytes += sparse_file.bytes_fetched
    if sparse is not None and geo_file is not None:
        nbytes += geo_file.bytes_fetched
    if sparse is not None:
        metrics["geo_chunks"] = len(sparse.loaded)

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
            "n_band_rule_changed": sum(
                p.get("band_rule_changed", False) for p in pixels.values()
            ),
        }
        rec.update(site_cluster_summary(assigned, k))
        records.append(rec)
    copy_records = []
    for copy in copies:
        peak = copy["peak"]
        r, c = copy["peak_row"], copy["peak_col"]
        row = {k: copy.get(k, np.nan) for k in COPY_COLUMNS}
        row.update(
            satellite=job["satellite"],
            granule=job["gid"],
            time_utc=(
                job["start"] + timedelta(seconds=GRANULE_SECONDS * r / nrow)
            ).isoformat(timespec="seconds"),
            scan=r // 16,
            footprint_m2=copy["footprint"],
            rh75_mw=copy["rh_mw"],
            temp_k=peak["temp_k"],
            area_m2=peak["esf"] * peak["area_m2"],
            bands="+".join(peak["bands"]),
            fit_flag=peak["fit_flag"],
            band_rule_changed=peak.get("band_rule_changed", False),
            t_fit_rejected=peak.get("t_fit_rejected", np.nan),
            cloud_mask_peak=copy_cloud.get((r, c), (-1, 0))[0],
            clear_neighbors_peak=copy_cloud.get((r, c), (-1, 0))[1],
            assigned_site=int(sites[copy["site"], 0])
            if copy["site"] is not None
            else "",
        )
        for i, key in enumerate(
            ("pair_distance_px", "pair_dcol", "pair_drow", "pair_rh_ratio")
        ):
            row[key] = ";".join(str(pair[i]) for pair in copy["pairs"])
        copy_records.append(row)
    if reduced:
        had_nondetections = any(not r["detected"] for r in records)
        records = [r for r in records if r["detected"]]
        if had_nondetections:
            # The full table serializes these optional integer coordinates as
            # floats when non-detections introduce NaNs. Preserve those exact
            # detection field strings after removing the other rows.
            for rec in records:
                for key in ("peak_row", "peak_col", "temp_k", "area_m2", "rh_peak_mw"):
                    rec[key] = float(rec[key])
    return records, copy_records


def granule_start(gid: str) -> datetime:
    """'d20261101_t0200577' -> datetime."""
    return nodd._hhmmsss(gid[1:9], gid[11:18])
