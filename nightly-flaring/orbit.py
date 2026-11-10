"""Local TLE/SGP4 scan-plane prediction for opt-in VIIRS cold reads (D64)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from sgp4.api import Satrec, jday
from sgp4.propagation import gstime

import nodd

# WGS84 geodetic sites; TEME -> pseudo Earth-fixed with GMST (no live EOP).


def interval(job: dict) -> tuple[datetime, datetime]:
    """SDR names carry both endpoints, including deciseconds and UTC rollover."""
    match = nodd.SDR_RE.search(job.get("keys", {}).get("geo", ""))
    if match:
        _, day, first, last = match.groups()
        start = datetime.strptime(day + first[:6], "%Y%m%d%H%M%S").replace(
            microsecond=int(first[6]) * 100000
        )
        end = datetime.strptime(day + last[:6], "%Y%m%d%H%M%S").replace(
            microsecond=int(last[6]) * 100000
        )
        if end < start:
            end += timedelta(days=1)
        return start, end
    match = re.fullmatch(r"d(\d{8})_t(\d{7})", job["gid"])
    if not match:
        raise ValueError("granule name has no timing")
    day, first = match.groups()
    start = datetime.strptime(day + first[:6], "%Y%m%d%H%M%S").replace(
        microsecond=int(first[6]) * 100000
    )
    # Header-only offline jobs lack endpoints. VIIRS has 48 scans at 1.779 s.
    return start, start + timedelta(seconds=84.5)


@lru_cache(maxsize=12)
def _catalog(directory, satellite):
    records = []
    container = nodd.AZURE_CONTAINERS[satellite]
    for path in sorted((Path(directory) / container).glob("*.h5")):
        with h5py.File(path, "r") as h:
            lines = bytes(h["All_Data/TLE-AUX_All/Dataset_Array_0"][()]).decode()
        lines = lines.strip().splitlines()
        if len(lines) != 2:
            continue
        sat = Satrec.twoline2rv(*lines)
        records.append((sat.jdsatepoch + sat.jdsatepochF, sat))
    return records


def tle(job: dict, directory: str | Path) -> Satrec:
    start, _ = interval(job)
    jd = sum(jday(*start.timetuple()[:6])) + start.microsecond / 86400e6
    records = _catalog(str(directory), job["satellite"])
    if not records:
        raise ValueError("no local TLE")
    epoch, sat = min(records, key=lambda item: abs(item[0] - jd))
    if abs(epoch - jd) > 3:
        raise ValueError("TLE epoch farther than 3 days")
    return sat


def ecef(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    radius = 6378.137 / np.sqrt(1 - 0.00669437999014 * np.sin(lat) ** 2)
    return np.column_stack(
        [
            radius * np.cos(lat) * np.cos(lon),
            radius * np.cos(lat) * np.sin(lon),
            radius * (1 - 0.00669437999014) * np.sin(lat),
        ]
    )


def state(sat: Satrec, start: datetime, seconds) -> tuple[np.ndarray, np.ndarray]:
    """Earth-fixed positions and inertial velocities rotated into Earth axes.

    The VIIRS scan plane uses inertial velocity, not the ground-track tangent.
    Subtracting Earth rotation produced up to 140-row errors in the samples.
    """
    start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
    jd0, fr0 = jday(*start.utctimetuple()[:6])
    fr = np.atleast_1d(fr0 + start.microsecond / 86400e6 + seconds / 86400)
    error, r, v = sat.sgp4_array(np.full(fr.shape, jd0), fr)
    if np.any(error):
        raise ValueError("SGP4 propagation error")
    theta = np.array([gstime(jd0 + f) for f in fr])
    co, si = np.cos(theta), np.sin(theta)

    def rotate(a):
        return np.column_stack(
            [co * a[:, 0] + si * a[:, 1], -si * a[:, 0] + co * a[:, 1], a[:, 2]]
        )

    r, v = rotate(r), rotate(v)
    return r, v


def scan_coordinates(job: dict, lat, lon, directory) -> tuple[np.ndarray, np.ndarray]:
    """Return scan-plane crossing seconds and signed off-nadir angle in radians."""
    start, end = interval(job)
    duration = (end - start).total_seconds()
    sat = tle(job, directory)
    points = ecef(np.asarray(lat), np.asarray(lon))
    seconds = np.full(len(points), duration / 2)
    for _ in range(4):
        r, v = state(sat, start, seconds)
        radial = r / np.linalg.norm(r, axis=1)[:, None]
        tangent = v - (v * radial).sum(axis=1)[:, None] * radial
        along = tangent / np.linalg.norm(tangent, axis=1)[:, None]
        delta = points - r
        correction = (delta * along).sum(axis=1) / np.linalg.norm(tangent, axis=1)
        seconds += np.clip(correction, -180, 180)
    r, v = state(sat, start, seconds)
    radial = r / np.linalg.norm(r, axis=1)[:, None]
    along = v - (v * radial).sum(axis=1)[:, None] * radial
    along /= np.linalg.norm(along, axis=1)[:, None]
    cross = np.cross(along, radial)
    delta = points - r
    angle = np.arctan2((delta * cross).sum(axis=1), -(delta * radial).sum(axis=1))
    return seconds, angle


# Fit once from 16,800 valid pixels in seven supplied 2024/2026 GMTCOs.
# Column aggregation: 1:1 / 2:1 / 3:1 / 2:1 / 1:1 native samples.
RAW_ANGLE_SLOPE = -3219.70057624
RAW_ANGLE_INTERCEPT = 3152.42729942
COL_KNOTS = (0, 640, 1008, 2192, 2560, 3200)
RAW_KNOTS = (0, 640, 1376, 4928, 5664, 6304)
ROW_BIAS = 5.0


def predict(job: dict, directory: str | Path, shape=(768, 3200)) -> dict:
    """Predict pixels plus a deliberately wide +/-60-degree, +/-10-second swath.

    Missing, stale or corrupt TLEs raise ValueError: callers retain the job and
    use whole GMTCO. All sites are predicted, including those outside the swath;
    T2 certifies each nearest pixel or falls back; it leaves out only sites
    predicted beyond the image by more than the window margins (D64).
    """
    sites = np.asarray(job["sites"], dtype=float).reshape(-1, 3)
    start, end = interval(job)
    seconds, angle = scan_coordinates(job, sites[:, 1], sites[:, 2], directory)
    duration = (end - start).total_seconds()
    raw = RAW_ANGLE_SLOPE * angle + RAW_ANGLE_INTERCEPT
    # Extrapolation beyond edges must not create a spurious in-swath prediction.
    col = np.interp(raw, RAW_KNOTS, COL_KNOTS)
    col = np.where(raw < 0, raw, np.where(raw > 6304, 3200 + raw - 6304, col))
    satellite = tle(job, directory)
    r, _ = state(satellite, start, seconds)
    points = ecef(sites[:, 1], sites[:, 2])
    visible = ((r - points) * points).sum(axis=1) > 0
    return dict(
        row=seconds / duration * shape[0] - ROW_BIAS,
        col=col * shape[1] / 3200,
        seconds=seconds,
        angle_deg=np.degrees(angle),
        eligible=visible
        & (seconds >= -10)
        & (seconds <= duration + 10)
        & (np.abs(angle) <= np.radians(60)),
    )
