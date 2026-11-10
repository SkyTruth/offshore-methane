"""Opt-in cold VIIRS certificates and chunk reads (D64).

Margins are empirical bounds validated offline, not per-object equality proofs.
Uncertain predictions, night masks, borders and ties retain whole GMTCO reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

import nightfire as nf
import nodd

GEO = "All_Data/VIIRS-MOD-GEO-TC_All"


@dataclass(frozen=True)
class Options:
    header_prefilter: bool = False  # T1 v2
    windowed_geo: bool = False  # T2 v2
    orbit_whitelist: bool = False  # T7; also supplies T2 predictions
    tle_dir: str | None = None
    footprint_buffer_km: float = 25.0
    night_margin_deg: float = 1.0
    row_margin: int = 64  # includes adjacent-scan +/-16-line bow-tie overlap
    col_margin: int = 96
    tier_order: bool = False  # T4a
    tier2_detections_only: bool = False  # T4b
    satellite_first: str | None = None  # T5; T6 is workers
    tier1_ids: tuple[int, ...] = ()
    block_size: int = 16384

    def __post_init__(self):
        if self.block_size < 1024:
            raise ValueError("block_size must be at least 1024")
        if self.footprint_buffer_km < 0 or self.night_margin_deg < 0:
            raise ValueError("certificate margins must be nonnegative")
        if self.row_margin < 32 or self.col_margin < 2:
            raise ValueError("window margins must include bow-tie overlap")
        if self.satellite_first not in (None, *nodd.BUCKETS):
            raise ValueError("unknown satellite_first")
        if (self.tier_order or self.tier2_detections_only) and not self.tier1_ids:
            raise ValueError("site tiers require explicit nonempty tier1_ids")


def options(job: dict) -> Options:
    return Options(**job.get("speedup", {}))


def tier1(job: dict, opts: Options) -> bool:
    return bool(set(opts.tier1_ids).intersection(int(s[0]) for s in job["sites"]))


def configure(jobs: list[dict], opts: Options) -> list[dict]:
    """Copy jobs; preserve default chronological order unless explicitly changed."""
    result = [dict(j, speedup=asdict(opts)) for j in jobs]
    if opts.tier_order or opts.satellite_first:
        result.sort(
            key=lambda j: (
                bool(opts.satellite_first and j["satellite"] != opts.satellite_first),
                bool(opts.tier_order and not tier1(j, opts)),
                j["start"],
                j["satellite"],
            )
        )
    return result


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    )


def geo_valid(lat, lon):
    return (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat > nf.FLOAT_FILL_MAX)
        & (lon > nf.FLOAT_FILL_MAX)
    )


def header(h) -> dict:
    """Read only Data_Products granule attributes, never science datasets."""
    root = h.get("Data_Products/VIIRS-MOD-GEO-TC")
    if root is None:
        return {}
    ds = root.get("VIIRS-MOD-GEO-TC_Gran_0")
    if ds is None:
        return {}
    out = {}
    for key in ("G-Ring_Latitude", "G-Ring_Longitude", "N_Day_Night_Flag"):
        if key in ds.attrs:
            value = np.asarray(ds.attrs[key])
            out[key] = (
                value.astype(str).tolist()
                if value.dtype.kind in "SO"
                else value.tolist()
            )
    return out


def _ring(attrs):
    lat = np.asarray(attrs.get("G-Ring_Latitude", []), dtype=float).ravel()
    lon = np.asarray(attrs.get("G-Ring_Longitude", []), dtype=float).ravel()
    if len(lat) < 4 or lat.shape != lon.shape or not geo_valid(lat, lon).all():
        raise ValueError("missing or invalid G-Ring")
    if (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any():
        raise ValueError("out-of-range G-Ring")
    points = xyz(lat, lon)
    center = points.mean(axis=0)
    norm = np.linalg.norm(center)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("uncertain G-Ring center")
    center /= norm
    east = np.cross([0, 0, 1], center)
    if np.linalg.norm(east) < 1e-6:
        east = np.cross([1, 0, 0], center)
    east /= np.linalg.norm(east)
    north = np.cross(center, east)

    def project(vectors):
        cosine = vectors @ center
        return np.column_stack([vectors @ east, vectors @ north]) / cosine[
            :, None
        ], cosine

    projected, cosine = project(points)
    if not np.isfinite(projected).all() or cosine.min() <= 0.5:
        raise ValueError("uncertain G-Ring hemisphere")
    polygon = Polygon(projected * nf.EARTH_RADIUS_M / 1000)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("uncertain G-Ring projection")
    return points, polygon, project


def footprint_distances(job, attrs):
    """Signed distance in km in a local gnomonic plane; None is uncertain."""
    try:
        _, polygon, project = _ring(attrs)
        sites = np.asarray(job["sites"], dtype=float).reshape(-1, 3)
        projected, cosine = project(xyz(sites[:, 1], sites[:, 2]))
        distances = []
        for xy, co in zip(projected, cosine):
            if co <= 0:
                distances.append(np.inf)
                continue
            point = Point(xy * nf.EARTH_RADIUS_M / 1000)
            distances.append(
                polygon.exterior.distance(point) * (-1 if polygon.covers(point) else 1)
            )
        return np.asarray(distances)
    except (ValueError, TypeError, KeyError):
        return None


def footprint_skip(job, attrs, buffer_km=25.0):
    distance = footprint_distances(job, attrs)
    return bool(distance is not None and np.all(distance > buffer_km))


def solar_skip(job):
    """SZA is Lipschitz in angular distance; never use the Day flag to skip."""
    from granule import GRANULE_SECONDS, MAX_SITE_PIXEL_M, SZA_NIGHT

    sites = np.asarray(job["sites"], dtype=float).reshape(-1, 3)
    if not len(sites):
        return True
    mid = job["start"] + timedelta(seconds=GRANULE_SECONDS / 2)
    radius = np.degrees(MAX_SITE_PIXEL_M / nf.EARTH_RADIUS_M)
    angles = nf.solar_zenith_deg(sites[:, 1], sites[:, 2], mid)
    return bool(np.all(angles + radius + 0.001 < SZA_NIGHT))


def prefilter(job, opts, attrs):
    if solar_skip(job):
        return True, "site_solar_bound"
    if footprint_skip(job, attrs, opts.footprint_buffer_km):
        return True, "header_footprint"
    return False, "retain_header"


def night_certificate(attrs, start, margin=1.0):
    """Sample every G-Ring edge at 1/32 intervals; pad by 1 degree by default.

    Reject the solar subpoint inside the projected polygon as well as edges
    near the terminator. The margin covers edge curvature, off-edge detectors
    and sampling; whole-array minima on seven real granules validate it.
    """
    from granule import GRANULE_SECONDS, SZA_NIGHT

    try:
        points, polygon, project = _ring(attrs)
    except (ValueError, TypeError):
        return False, None
    weight = np.linspace(0, 1, 33)
    edges = (
        (1 - weight[None, :, None]) * points[:, None, :]
        + weight[None, :, None] * np.roll(points, -1, axis=0)[:, None, :]
    ).reshape(-1, 3)
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    lat = np.degrees(np.arcsin(edges[:, 2]))
    lon = np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))
    mid = start + timedelta(seconds=GRANULE_SECONDS / 2)
    # Recover the sun unit vector from cos(SZA) at three orthogonal locations.
    sun = np.cos(
        np.radians(nf.solar_zenith_deg(np.array([0, 0, 90]), np.array([0, 90, 0]), mid))
    )
    xy, cosine = project(sun[None, :])
    if cosine[0] > 0 and polygon.covers(Point(xy[0] * nf.EARTH_RADIUS_M / 1000)):
        return False, 0.0
    minimum = float(nf.solar_zenith_deg(lat, lon, mid).min())
    return minimum >= SZA_NIGHT + margin, minimum


# Largest orbit-prediction errors over 170,715 reference site rows (D64), plus
# room for the +/-16-row bow-tie overlap and the 1.2 km radius (<= 3 pixels).
MAX_PRED_ERROR_ROWS = 26.68 + 16 + 3
MAX_PRED_ERROR_COLS = 14.05 + 3


def far_outside(prediction, shape, opts) -> np.ndarray:
    """Sites predicted beyond the image by more than the window margins.

    With margins above the measured prediction error, such a site has no pixel
    within ``MAX_SITE_PIXEL_M``, so the full read would drop it as well (D64).
    """
    row, col = np.asarray(prediction["row"]), np.asarray(prediction["col"])
    return (
        (row < -opts.row_margin)
        | (row >= shape[0] + opts.row_margin)
        | (col < -opts.col_margin)
        | (col >= shape[1] + opts.col_margin)
    )


def margins_cover_error(opts) -> bool:
    return (
        opts.row_margin >= MAX_PRED_ERROR_ROWS
        and opts.col_margin >= MAX_PRED_ERROR_COLS
    )


class SparseGeo:
    """Chunked JRR reads with prediction, border and tie certificates."""

    def __init__(self, h, shape):
        a, b = h["Latitude"], h["Longitude"]
        if (
            a.shape != tuple(shape)
            or b.shape != tuple(shape)
            or not a.chunks
            or a.chunks != b.chunks
            or a.dtype != np.dtype("float32")
            or b.dtype != a.dtype
        ):
            raise ValueError("JRR layout differs from validated float32 pairs")
        self.h = h
        self.misses = 0
        self.lat = np.full(shape, np.nan, dtype=a.dtype)
        self.lon = np.full_like(self.lat, np.nan)
        self.loaded = set()
        self.checked = set()
        self.dr, self.dc = a.chunks
        # SDR fill/bow-tie pixels have nonfinite M10: checked on all samples.
        self.valid = np.ones(shape, dtype=bool)
        self.reason = "prediction"
        self.widened = False

    def load(self, r, c):
        r, c = r // self.dr * self.dr, c // self.dc * self.dc
        if (r, c) not in self.loaded:
            sl = np.s_[r : r + self.dr, c : c + self.dc]
            self.lat[sl], self.lon[sl] = self.h["Latitude"][sl], self.h["Longitude"][sl]
            self.loaded.add((r, c))

    def rectangle(self, r0, r1, c0, c1):
        for r in range(r0 // self.dr * self.dr, r1, self.dr):
            for c in range(c0 // self.dc * self.dc, c1, self.dc):
                self.load(r, c)

    def ensure(self, pixels, halo=1):
        for r, c in pixels:
            self.rectangle(
                max(0, r - halo),
                min(len(self.lat), r + halo + 1),
                max(0, c - halo),
                min(self.lat.shape[1], c + halo + 1),
            )

    def check(self, l10, when):
        """Invalidate a certificate when newly loaded chunks contradict it."""
        from granule import SZA_NIGHT

        for r, c in self.loaded - self.checked:
            sl = np.s_[r : r + self.dr, c : c + self.dc]
            finite = np.isfinite(l10[sl])
            valid = geo_valid(self.lat[sl], self.lon[sl])
            if np.any(finite & ~valid):
                return "geo_fill"
            if np.any(
                nf.solar_zenith_deg(self.lat[sl][finite], self.lon[sl][finite], when)
                < SZA_NIGHT
            ):
                return "loaded_day_pixel"
            self.checked.add((r, c))
        return None

    def nearest(self, l10, sites, prediction, opts, excluded=None):
        """Widen once, then None requests full GMTCO.

        A nearest pixel at or beyond ``MAX_SITE_PIXEL_M`` inside the window is a
        certified miss when the margins cover the prediction error: any pixel
        within the radius would lie inside the window (D64).
        """
        from granule import MAX_SITE_PIXEL_M

        points = xyz(sites[:, 1].astype(float), sites[:, 2].astype(float))
        nrow, ncol = self.lat.shape
        excluded = np.zeros(len(sites), bool) if excluded is None else excluded
        if not np.all(np.isfinite(prediction["row"])) or not np.all(
            np.isfinite(prediction["col"])
        ):
            return None
        for scale in (1, 2):
            windows = []
            for r, c, omit in zip(prediction["row"], prediction["col"], excluded):
                if omit:
                    windows.append(None)
                    continue
                rr, cc = (
                    int(np.clip(round(r), 0, nrow - 1)),
                    int(np.clip(round(c), 0, ncol - 1)),
                )
                r0, r1 = (
                    max(0, rr - opts.row_margin * scale),
                    min(nrow, rr + opts.row_margin * scale + 1),
                )
                c0, c1 = (
                    max(0, cc - opts.col_margin * scale),
                    min(ncol, cc + opts.col_margin * scale + 1),
                )
                self.rectangle(r0, r1, c0, c1)
                windows.append((r0, r1, c0, c1))
            use = geo_valid(self.lat, self.lon) & np.isfinite(l10)
            # A finite-M10 GEO fill violates the co-mask assumption: fall back.
            for r, c in self.loaded:
                sl = np.s_[r : r + self.dr, c : c + self.dc]
                if np.any(
                    np.isfinite(l10[sl]) & ~geo_valid(self.lat[sl], self.lon[sl])
                ):
                    self.reason = "geo_fill"
                    return None
            rr, cc = np.nonzero(use)
            if not len(rr):
                self.reason = "no_valid_pixels"
                continue
            distances, indexes = cKDTree(xyz(self.lat[use], self.lon[use])).query(
                points, k=2
            )
            idx = indexes[:, 0]
            rows, cols = rr[idx], cc[idx]
            dist = 2 * nf.EARTH_RADIUS_M * np.arcsin(np.clip(distances[:, 0] / 2, 0, 1))
            success = True
            for i, window in enumerate(windows):
                if window is None:
                    dist[i] = np.inf
                    continue
                r0, r1, c0, c1 = window
                r, c = rows[i], cols[i]
                if dist[i] >= MAX_SITE_PIXEL_M:
                    if not margins_cover_error(opts):
                        self.reason = "nearest_outside_radius"
                        success = False
                    else:
                        self.misses += 1
                elif abs(distances[i, 1] - distances[i, 0]) < 1e-12:
                    self.reason = "tie"
                    return None
                elif (
                    (r0 > 0 and r < r0 + 16)
                    or (r1 < nrow and r >= r1 - 16)
                    or (c0 > 0 and c <= c0)
                    or (c1 < ncol and c >= c1 - 1)
                ):
                    self.reason = "border"
                    success = False
            if success:
                self.widened = scale == 2
                return rows, cols, dist
        return None
