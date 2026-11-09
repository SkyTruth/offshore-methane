# %%
"""Clear dark runs and nearby EMIT review scenes (D62)."""

from __future__ import annotations

import bisect
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import emit
import flareouts as f
import numpy as np
import pandas as pd
import requests

WINDOW = 14
MIN_QUALIFIED = 10
MAX_RENDERS = 60
DEFINITIONS = {
    "dark_night": (
        "At least one raw cloud-mask 0/1 pass and no source-copy detection "
        "within 1.5 km on any pass that night, including cloudy detections. "
        "A night is not listed when another source unit within 1.5 km had its "
        "own detections that night: the two cannot be told apart reliably."
    ),
    "burning_rate": (
        "More than half of at least 10 clear nights within 14 nights before or "
        "after had a detection within 1 km or nearest this source unit. Exclude "
        "the selected night and nights with only a neighbour's detections within "
        "1.5 km; show that excluded count. Other dark nights count as unlit. "
        "A lit clear night must precede the run inside "
        "that window."
    ),
    "recent_heat": (
        "Median measured heat over lit clear nights in that window, using clear "
        "source groups within 1 km; n counts nights with heat measured. Missing "
        "heat stays unmeasured."
    ),
    "run": (
        "One row per run of clear dark nights. Cloud-only and unobserved nights "
        "do not break a run; any detection ends it. Each qualifying night keeps "
        "its own rate, heat and passes."
    ),
    "provisional": (
        "Fewer than 14 subsequent nights have been processed; the burning rate "
        "can change next month."
    ),
    "order": (
        "Most surprising combines the 95% Wilson lower bound of lit/clear nights "
        "with 1 minus (1-p)^k, using the heat class's same-night re-detection "
        "rate p and k clear passes. Highest first, then higher recent heat. A "
        "run uses its best night. Passes on one night are not independent; this "
        "is a review order, not a probability."
    ),
    "passes": (
        "One exposure per satellite pass. View zone: near nadir 1008–2191; swath "
        "edge below 640 or at least 2560; otherwise mid-swath."
    ),
    "bracket": (
        "Last pass that saw the source before the run, first and last clear dark "
        "passes, and first pass that saw it after. The next detection is unknown "
        "while still dark."
    ),
    "emit": (
        "Scenes from two local nights before the run through two nights after. "
        "D59 labels follow the VIIRS bracket: before, during, boundary, after. A "
        "daytime scene beside a single dark night is usually boundary."
    ),
    "comparison": (
        "Up to three scenes per source, lit on the nearest VIIRS passes on both "
        "sides, outside every listed run's two-night scene window. Closest to "
        "the dark passes first."
    ),
    "imagery": (
        "At most 60 new scene renders per build by default. Reserve half the "
        "budget, rounded up, for never-attempted scenes. Retry failures after 1, "
        "2, 4, 8, 16, then 32 days. Cached chips use no budget; scenes beyond "
        "the measured Earth Engine horizon stay pending."
    ),
    "neighbours": (
        "Link runs at other listed sources within 1.5 km when they share a clear "
        "dark night; their detection apertures overlap."
    ),
}
LIMITS = (
    "A dark night is not evidence of an outage; most single dark nights at "
    "steady flares are unexplained. EMIT scenes are daytime, hours from the "
    "VIIRS passes. The CH4 index is not a concentration."
)


class Screening(dict):
    """JSON payload with full pass histories kept in memory for enrichment only."""

    def __init__(self, payload, source_passes):
        super().__init__(payload)
        self._source_passes = source_passes


def _stamp(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    stamp = (
        stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    )
    return stamp.isoformat().replace("+00:00", "Z")


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _clean(value):
    """Compact finite floats, including tiny miss expressions, without NaN."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return float(f"{value:.9g}") if math.isfinite(value) else None
    return value


def wilson_lower_95(lit: int, qualified: int) -> float:
    """Lower end of the two-sided 95% Wilson interval for a lit share."""
    if qualified <= 0:
        return 0.0
    z = 1.959963984540054
    rate = lit / qualified
    return (
        rate
        + z * z / (2 * qualified)
        - z
        * math.sqrt(rate * (1 - rate) / qualified + z * z / (4 * qualified * qualified))
    ) / (1 + z * z / qualified)


def default_key(night: dict) -> tuple:
    return (
        -(night["score"] if night["score"] is not None else -1),
        -(night["recent_heat_mw"] or 0),
        night["night"],
    )


def best_night(run: dict) -> dict:
    return min(run["nights"], key=default_key)


def _view_zone(column) -> str:
    if column is None or pd.isna(column):
        return "unknown"
    if 1008 <= column < 2192:
        return "near nadir"
    return "swath edge" if column < 640 or column >= 2560 else "mid-swath"


def pass_records(obs: pd.DataFrame, copies: pd.DataFrame) -> list[dict]:
    """Source detections replace assignment flags, including cloudy positives.

    Keep copy-only passes as positive brackets when a site exposure is missing.
    Adjacent granules of the same physical pass can differ by under ten seconds.
    """
    records, matched = [], set()
    cp = copies.reset_index(drop=True)
    times = pd.to_datetime(cp.time_utc, utc=True)
    for row in obs.itertuples():
        same = cp.satellite.eq(row.satellite) & (
            cp.granule.eq(row.granule)
            | (times - pd.Timestamp(_stamp(row.time_utc)))
            .abs()
            .lt(pd.Timedelta(seconds=10))
        )
        matched.update(cp.index[same])
        records.append(
            {
                "time_utc": _stamp(row.time_utc),
                "night": row.night,
                "satellite": row.satellite,
                "clear": bool(row.raw_clear),
                "view_zone": _view_zone(getattr(row, "col", None)),
                "detected": bool(same.any()),
                "source_detected": bool((same & cp.get("rate_source", True)).any()),
            }
        )
    for i, row in enumerate(cp.itertuples()):
        if i not in matched:
            records.append(
                {
                    "time_utc": _stamp(row.time_utc),
                    "night": row.night,
                    "satellite": row.satellite,
                    "clear": False,
                    "view_zone": "unknown",
                    "detected": True,
                    "source_detected": bool(getattr(row, "rate_source", True)),
                }
            )
    return sorted(records, key=lambda r: r["time_utc"])


def _expected(heat: float | None, k: int, curve: list[dict]) -> dict:
    row = next(
        (
            r
            for r in curve
            if heat is not None
            and heat > r["lower_mw"]
            and (r["upper_mw"] is None or heat <= r["upper_mw"])
        ),
        {},
    )
    p = row.get("probability")
    return {
        "p": p,
        "n": row.get("trials", 0),
        "detections": row.get("detections", 0),
        "k": k,
        "miss_all": (1 - p) ** k if p is not None else None,
        "lower_mw": row.get("lower_mw"),
        "upper_mw": row.get("upper_mw"),
    }


def recount(payload: dict) -> None:
    runs = payload["runs"]

    def chips(run):
        scenes = payload["sources"][run["source_id"]]["scenes"]
        return any(
            scenes[r[0]].get("imagery", {}).get("ch4")
            and emit.usable_chips(scenes[r[0]]["imagery"])
            for r in run["emit_scenes"]
        )

    payload["counts"] = {
        "dark_nights": sum(len(r["nights"]) for r in runs),
        "runs": len(runs),
        "sources": len(payload["sources"]),
        "with_emit_nearby": sum(len(r["nights"]) for r in runs if r["emit_scenes"]),
        "runs_with_emit": sum(bool(r["emit_scenes"]) for r in runs),
        "with_chips": sum(len(r["nights"]) for r in runs if chips(r)),
        "provisional": sum(n["provisional"] for r in runs for n in r["nights"]),
        "skipped_neighbour_detected": payload.get("neighbour_detected_nights", 0),
    }


def _series(h: pd.DataFrame, runs: list[dict]) -> list[dict]:
    """Disjoint column blocks, only within 14 nights of listed runs."""
    selected = pd.Series(False, index=h.index)
    for run in runs:
        start = str((pd.Timestamp(run["start"]) - pd.Timedelta(days=WINDOW)).date())
        end = str((pd.Timestamp(run["end"]) + pd.Timedelta(days=WINDOW)).date())
        selected.loc[start:end] = True
    blocks, block = [], None
    for night, row in h.iterrows():
        if not selected.loc[night]:
            block = None
            continue
        if block is None:
            block = {"start": night, "heat": [], "state": ""}
            blocks.append(block)
        heat = (
            float(row.heat_mw)
            if pd.notna(row.heat_mw) and math.isfinite(row.heat_mw)
            else None
        )
        block["heat"].append(heat)
        neighbour_only = bool(getattr(row, "neighbour_only", False))
        block["state"] += (
            "N"
            if neighbour_only
            else "L"
            if row.lit and heat is not None
            else ("H" if row.lit else "D" if row.qualified else "U")
        )
    return blocks


def _nearest_unit(cp: pd.DataFrame, units: list[dict]) -> np.ndarray:
    """Unit id nearest to each copy centroid."""
    distances = f.distance_m(
        cp.centroid_lat.to_numpy()[:, None],
        cp.centroid_lon.to_numpy()[:, None],
        np.array([u["lat"] for u in units]),
        np.array([u["lon"] for u in units]),
    )
    return np.array([u["id"] for u in units])[distances.argmin(axis=1)]


def neighbour_detection_nights(c: pd.DataFrame, units: list[dict]) -> dict:
    """Nights when another unit within 1.5 km had detections nearest to it."""
    if c.empty or not units:
        return {u["id"]: set() for u in units}
    owners = _nearest_unit(c, units)
    lookup = {u["id"]: u for u in units}
    # Ownership needs the detection within 1.5 km of its nearest unit; a lone
    # copy kilometres from every unit belongs to none of them.
    near = np.array(
        [
            f.distance_m(lookup[o]["lat"], lookup[o]["lon"], la, lo) <= 1500
            for o, la, lo in zip(owners, c.centroid_lat, c.centroid_lon)
        ],
        dtype=bool,
    )
    nights_arr = c.night.to_numpy()
    owned = {uid: set(nights_arr[(owners == uid) & near]) for uid in lookup}
    busy = {}
    for u in units:
        nights = set()
        for other in units:
            if other["id"] == u["id"]:
                continue
            if f.distance_m(u["lat"], u["lon"], other["lat"], other["lon"]) <= 1500:
                nights |= owned.get(other["id"], set())
        busy[u["id"]] = nights
    return busy


def rate_history(h, cp, unit, units):
    """Attribute rate positives locally, retaining the wider darkness veto."""
    cp = cp.copy()
    distances = f.distance_m(unit["lat"], unit["lon"], cp.centroid_lat, cp.centroid_lon)
    owners = _nearest_unit(cp, units)
    cp["rate_source"] = (distances <= 1000) | (owners == unit["id"])
    own = cp[cp.rate_source].groupby("night").size()
    h = h.copy()
    h["rate_lit"] = own.reindex(h.index).fillna(0).gt(0) & h.lit
    h["neighbour_only"] = h.lit & ~h.rate_lit
    return h, cp


def screen(nights, over, copies, *, cards: dict | None = None) -> Screening:
    """Select qualifying nights, grouped by run, without network calls."""
    days = (
        pd.date_range(nights.night.min(), nights.night.max())
        .strftime("%Y-%m-%d")
        .tolist()
    )
    p = f.prepare_passes(nights, over)
    c = f.prepare_copies(copies, days[0], days[-1])
    baseline_end = str((pd.Timestamp(days[0]) + pd.Timedelta(days=29)).date())
    units = f.source_units(c, baseline_end)
    curve = f.detection_curve(p)
    cards = cards if cards is not None else f.screen(nights, over, copies)
    payload = {
        "start": days[0],
        "as_of": days[-1],
        "definitions": DEFINITIONS.copy(),
        "limits": LIMITS,
        "caption": emit.CAPTION,
        "detection_curve": curve,
        "series_states": {
            "L": "lit with measured heat",
            "H": "lit without measured heat",
            "N": "lit only by a neighbouring source's detection",
            "D": "clear dark",
            "U": "unknown",
        },
        "pass_columns": [
            "satellite_index",
            "seconds_from_night_utc",
            "clear",
            "view_zone_index",
            "detected",
        ],
        "pass_satellites": sorted(set(p.satellite) | set(c.satellite)),
        "pass_view_zones": ["near nadir", "mid-swath", "swath edge", "unknown"],
        "scene_reference_columns": ["scene_id", "phase", "hours_from_dark_pass"],
        "runs": [],
        "sources": {},
    }
    source_passes, dark_dates = {}, {}
    busy = neighbour_detection_nights(c, units)
    vetoed = 0
    for unit in units:
        h, cp, obs = f.unit_history(unit, p, c, days)
        h, cp = rate_history(h, cp, unit, units)
        selected = []
        for spell in f.dark_spells(h, days[0]):
            dark = h.loc[spell["start"] : spell["end"]]
            dark = dark[dark.qualified & ~dark.lit]
            if dark.empty:
                continue
            run = {
                "id": f"{unit['id']}_{dark.index[0]}",
                "source_id": unit["id"],
                "start": dark.index[0],
                "end": dark.index[-1],
                "clear_dark_nights": len(dark),
                "nights": [],
                "emit_scenes": [],
                "neighbours": [],
            }
            for night in dark.index:
                if night in busy[unit["id"]]:
                    vetoed += 1
                    continue
                start = str((pd.Timestamp(night) - pd.Timedelta(days=WINDOW)).date())
                end = str((pd.Timestamp(night) + pd.Timedelta(days=WINDOW)).date())
                window = h.loc[start:end]
                q = window[window.qualified & (window.index != night)]
                excluded = int(q.neighbour_only.sum())
                q = q[~q.neighbour_only]
                lit = int(q.rate_lit.sum())
                before = q[(q.index < run["start"]) & q.rate_lit]
                if len(q) < MIN_QUALIFIED or lit <= len(q) / 2 or before.empty:
                    continue
                heat = q.loc[q.lit, "heat_mw"].dropna()
                heat = heat[heat.map(math.isfinite)]
                recent = float(heat.median()) if len(heat) else None
                k = int(h.loc[night, "clear_passes"])
                expected = _expected(recent, k, curve)
                curve_index = next(
                    (
                        i
                        for i, r in enumerate(curve)
                        if expected["p"] is not None
                        and r["lower_mw"] == expected["lower_mw"]
                    ),
                    None,
                )
                run["nights"].append(
                    {
                        "night": night,
                        "burning_rate": {
                            "lit": lit,
                            "qualified": len(q),
                            **({"neighbour_only": excluded} if excluded else {}),
                        },
                        "recent_heat_mw": recent,
                        "recent_heat_n": len(heat),
                        "expected_misses": {
                            "curve": curve_index,
                            "k": k,
                            "miss_all": expected["miss_all"],
                        },
                        "score": wilson_lower_95(lit, len(q))
                        * (1 - expected["miss_all"])
                        if expected["miss_all"] is not None
                        else None,
                        "provisional": end > days[-1],
                    }
                )
            if run["nights"]:
                run["card_id"] = next(
                    (
                        card["id"]
                        for card in cards["candidates"]
                        if card["unit"]["id"] == unit["id"]
                        and card["start"] <= run["end"]
                        and run["start"] <= card["end"]
                    ),
                    None,
                )
                selected.append(run)
                dark_dates[run["id"]] = set(dark.index)
        if not selected:
            continue
        passes = pass_records(obs, cp)
        source_passes[unit["id"]] = passes
        payload["sources"][unit["id"]] = {
            "unit": unit,
            "nightly_series": _series(h, selected),
            "scenes": {},
            "comparison_scenes": [],
            "emit_status": "not_listed",
        }
        for run in selected:
            dark_passes = [
                r["time_utc"]
                for r in passes
                if run["start"] <= r["night"] <= run["end"]
                and r["clear"]
                and not r["detected"]
            ]
            run["bracket"] = {
                "last_lit": max(
                    (
                        r["time_utc"]
                        for r in passes
                        if r["night"] < run["start"] and r["source_detected"]
                    ),
                    default=None,
                ),
                "first_dark": min(dark_passes, default=None),
                "last_dark": max(dark_passes, default=None),
                "recovery": min(
                    (
                        r["time_utc"]
                        for r in passes
                        if r["night"] > run["end"] and r["source_detected"]
                    ),
                    default=None,
                ),
            }
            for row in run["nights"]:
                origin = pd.Timestamp(row["night"], tz="UTC")
                row["passes"] = [
                    [
                        payload["pass_satellites"].index(r["satellite"]),
                        int((pd.Timestamp(r["time_utc"]) - origin).total_seconds()),
                        r["clear"],
                        payload["pass_view_zones"].index(r["view_zone"]),
                        r["detected"],
                    ]
                    for r in passes
                    if r["night"] == row["night"]
                ]
        payload["runs"].extend(selected)
    # Runs can share observations through the 1.5 km positive-evidence apertures.
    for i, run in enumerate(payload["runs"]):
        u = payload["sources"][run["source_id"]]["unit"]
        for other in payload["runs"][i + 1 :]:
            if run["source_id"] == other["source_id"] or not dark_dates[
                run["id"]
            ].intersection(dark_dates[other["id"]]):
                continue
            v = payload["sources"][other["source_id"]]["unit"]
            if f.distance_m(u["lat"], u["lon"], v["lat"], v["lon"]) <= 1500:
                run["neighbours"].append(other["id"])
                other["neighbours"].append(run["id"])
    payload["neighbour_detected_nights"] = vetoed
    payload = _clean(payload)
    payload["runs"].sort(key=lambda r: (*default_key(best_night(r)), r["id"]))
    recount(payload)
    return Screening(payload, source_passes)


def run_window(run: dict, lon: float) -> tuple[datetime, datetime]:
    start = str((pd.Timestamp(run["start"]) - pd.Timedelta(days=2)).date())
    end = str((pd.Timestamp(run["end"]) + pd.Timedelta(days=2)).date())
    return tuple(map(_instant, emit.utc_window(start, end, lon).split(",")))


def comparison_scenes(scenes, passes, dark_times, windows) -> list[dict]:
    """Both nearest passes lit; never reuse a scene in any listed run window."""
    times = [_instant(r["time_utc"]) for r in passes]
    controls = []
    for scene in scenes:
        t = _instant(scene["time_utc"])
        if any(first <= t < last for first, last in windows):
            continue
        before, after = bisect.bisect_left(times, t) - 1, bisect.bisect_right(times, t)
        if before < 0 or after == len(times):
            continue
        if passes[before].get("source_detected", passes[before]["detected"]) and passes[
            after
        ].get("source_detected", passes[after]["detected"]):
            controls.append(scene)
    controls.sort(
        key=lambda s: (
            min(abs((_instant(s["time_utc"]) - t).total_seconds()) for t in dark_times),
            s["scene_id"],
        )
    )
    return controls[:3]


def calendar_months(start: str, end: str):
    """Stable full calendar-month query keys, including the open month."""
    month = date.fromisoformat(start).replace(day=1)
    while month <= date.fromisoformat(end):
        following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield month.isoformat(), (following - timedelta(days=1)).isoformat()
        month = following


def enrich(
    payload: Screening,
    site_dir: Path | None = None,
    session=requests,
    imagery: bool = True,
    max_renders: int = MAX_RENDERS,
    cache_dir: Path = emit.CACHE,
    now: datetime | None = None,
) -> dict:
    """List by calendar month and advance a persistent, bounded imagery queue."""
    now = now or datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    groups = {}
    for run in payload["runs"]:
        groups.setdefault(run["source_id"], []).append(run)
    for uid, runs in groups.items():
        source = payload["sources"][uid]
        unit, passes = source["unit"], payload._source_passes[uid]
        entries, failures = [], set()
        months = list(calendar_months(payload["start"], payload["as_of"]))
        for product in emit.PRODUCTS:
            try:
                entries.extend(
                    (product, e)
                    for e in emit.calendar_listing(
                        product,
                        unit["lon"],
                        unit["lat"],
                        months,
                        cache_dir,
                        now.date(),
                        session,
                    )
                )
            except emit.CachedListing as cached:
                entries.extend((product, e) for e in cached.entries)
                failures.add(product)
            except Exception:  # noqa: BLE001 - external services must degrade offline
                failures.add(product)
        source["emit_status"] = (
            "unavailable"
            if len(failures) == len(emit.PRODUCTS)
            else ("partial" if failures else "available")
        )
        source["failed_products"] = sorted(failures)
        scenes = emit.scene_records(
            entries, {"unit": unit, "start": payload["start"], "end": payload["as_of"]}
        )
        record_first, record_last = map(
            _instant,
            emit.utc_window(payload["start"], payload["as_of"], unit["lon"]).split(","),
        )
        scenes = [
            s for s in scenes if record_first <= _instant(s["time_utc"]) < record_last
        ]
        windows, all_dark = [], []
        for run in runs:
            first, last = run_window(run, unit["lon"])
            windows.append((first, last))
            dark = [
                _instant(p["time_utc"])
                for p in passes
                if run["start"] <= p["night"] <= run["end"]
                and p["clear"]
                and not p["detected"]
            ]
            all_dark.extend(dark)
            run["emit_scenes"] = []
            for scene in scenes:
                t = _instant(scene["time_utc"])
                if first <= t < last:
                    hours = min(((t - d).total_seconds() / 3600 for d in dark), key=abs)
                    phase = emit.scene_phase(
                        t,
                        {
                            "unit": unit,
                            "start": run["start"],
                            "end": run["end"],
                            "viirs_times": run["bracket"],
                        },
                    )
                    run["emit_scenes"].append(
                        [scene["scene_id"], phase, round(hours, 2)]
                    )
        controls = comparison_scenes(scenes, passes, all_dark, windows)
        source["comparison_scenes"] = [s["scene_id"] for s in controls]
        ids = set(source["comparison_scenes"]) | {
            r[0] for run in runs for r in run["emit_scenes"]
        }
        source["scenes"] = {
            s["scene_id"]: {k: v for k, v in s.items() if k not in ("caption", "phase")}
            for s in scenes
            if s["scene_id"] in ids
        }
        for scene in source["scenes"].values():
            scene["products"] = sorted(set(scene["products"]))
    _render(payload, site_dir, session, imagery, max(0, max_renders), cache_dir, now)
    for uid, source in payload["sources"].items():
        folded, aliases = emit.fold_acquisitions(list(source["scenes"].values()))
        source["scenes"] = {s["scene_id"]: s for s in folded}
        for run in groups[uid]:
            references = {aliases[sid] for sid, _, _ in run["emit_scenes"]}
            run["emit_scenes"] = []
            for sid in sorted(
                references, key=lambda s: source["scenes"][s]["time_utc"]
            ):
                t = _instant(source["scenes"][sid]["time_utc"])
                phase = emit.scene_phase(
                    t, {"unit": source["unit"], "viirs_times": run["bracket"]}
                )
                dark = [
                    _instant(p["time_utc"])
                    for p in payload._source_passes[uid]
                    if run["start"] <= p["night"] <= run["end"]
                    and p["clear"]
                    and not p["detected"]
                ]
                hours = min(((t - d).total_seconds() / 3600 for d in dark), key=abs)
                run["emit_scenes"].append([sid, phase, round(hours, 2)])
        source["comparison_scenes"] = list(
            dict.fromkeys(
                aliases[sid]
                for sid in source["comparison_scenes"]
                if source["scenes"][aliases[sid]]
                .get("imagery", {})
                .get("source_coverage")
                == "covered"
            )
        )
        used = set(source["comparison_scenes"]) | {
            ref[0] for run in groups[uid] for ref in run["emit_scenes"]
        }
        source["scenes"] = {
            sid: s for sid, s in source["scenes"].items() if sid in used
        }
        # Every published scene states its coverage; unrendered ones are unknown.
        for scene in source["scenes"].values():
            scene.setdefault("imagery", {}).setdefault("source_coverage", "unknown")
    recount(payload)
    return _clean(payload)


def _render(payload, site_dir, session, imagery, budget, cache_dir, now):
    state_path = cache_dir / "darknight_attempts.json"
    try:
        state = json.loads(state_path.read_text())
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    work, seen = [], set()
    for run in payload["runs"]:
        work.extend((run["source_id"], r[0]) for r in run["emit_scenes"])
    work.extend(
        (uid, sid)
        for uid, s in payload["sources"].items()
        for sid in s["comparison_scenes"]
    )
    pending, renders, fallbacks = [], 0, {}
    client, latest = None, state.get("collection_latest_utc")
    for uid, sid in work:
        key = str(
            emit.chip_folder(
                {"scene_id": sid}, payload["sources"][uid]["unit"], cache_dir
            ).name
        )
        if key in seen:
            continue
        seen.add(key)
        source = payload["sources"][uid]
        scene, unit = source["scenes"][sid], source["unit"]
        chips = emit.cached_chips(scene, unit, cache_dir) if imagery else None
        if chips:
            _attach(scene, unit, chips, site_dir)
            if chips.get("needs_render"):
                fallbacks[key] = chips
                pending.append((key, unit, scene))
        elif not imagery:
            scene["imagery"] = {
                "status": "unavailable",
                "message": "Imagery disabled for this build.",
            }
        else:
            pending.append((key, unit, scene))
    # Cache hits never need Earth Engine. Reuse the saved horizon at zero budget;
    # measure it when absent so recent scenes still say pending.
    if pending and (budget or not latest):
        try:
            client = emit._ee_client()
            value = (
                client.ImageCollection(emit.COLLECTION)
                .aggregate_max("system:time_start")
                .getInfo()
            )
            if isinstance(value, (int, float)) and math.isfinite(value):
                latest = datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()
                state["collection_latest_utc"] = latest
        except Exception:  # noqa: BLE001 - external services must degrade offline
            # A failed horizon lookup need not discard an otherwise usable client.
            latest = state.get("collection_latest_utc")
    never, retries = [], []
    for key, unit, scene in pending:
        if latest and _instant(scene["time_utc"]) > _instant(latest):
            scene["imagery"] = {
                "status": "pending",
                "message": emit.pending_message(latest),
                "collection_latest_utc": latest,
            }
            continue
        scene["imagery"] = {"status": "queued", "message": "Queued for a later build."}
        attempt = state.get(key)
        if not attempt:
            never.append((key, unit, scene))
        elif _instant(attempt["retry_after"]) <= now:
            retries.append((key, unit, scene))
        else:
            scene["imagery"]["message"] = (
                "Render failed earlier; retry after "
                + attempt["retry_after"][:10]
                + "."
            )
    # Reserve ceil(budget/2) for new scenes; let the other side use spare slots.
    reserve = (budget + 1) // 2
    queue = never[:reserve] + retries[: max(0, budget - min(len(never), reserve))]
    queue += never[reserve : reserve + max(0, budget - len(queue))]
    for key, unit, scene in queue:
        if client is None:
            scene["imagery"] = {
                "status": "unavailable",
                "message": (
                    "Earth Engine connection unavailable; scene availability is "
                    "unknown. Retried on the next build."
                ),
                "collection_latest_utc": latest,
            }
            continue
        renders += 1
        try:
            chips = emit.fetch_chips(scene, unit, cache_dir, ee=client, session=session)
        except Exception:  # noqa: BLE001 - external services must degrade offline
            chips = {
                "status": "unavailable",
                "message": "Imagery render failed; queued for retry.",
            }
        attempts = state.get(key, {}).get("attempts", 0) + 1
        retry = now + timedelta(days=min(32, 2 ** min(attempts - 1, 5)))
        state[key] = {
            "attempts": attempts,
            "last_attempt": now.isoformat(),
            "retry_after": retry.isoformat(),
            "status": chips["status"],
        }
        if chips.get("collection_latest_utc"):
            state["collection_latest_utc"] = chips["collection_latest_utc"]
        _attach(scene, unit, chips, site_dir)
    for key, unit, scene in pending:
        if key in fallbacks and scene["imagery"]["status"] not in (
            "available",
            "no_data",
        ):
            _attach(scene, unit, fallbacks[key], site_dir)
    if imagery and (renders or state.get("collection_latest_utc")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, separators=(",", ":")))
        temp.replace(state_path)
    payload["rendering"] = {"max_renders": budget, "new_renders": renders}


def _attach(scene, unit, chips, site_dir):
    emit.attach_chips(scene, chips, site_dir, f"dn_{unit['id']}")


def main(
    nights=None,
    over=None,
    copies=None,
    site_dir=None,
    session=requests,
    imagery=True,
    max_renders=MAX_RENDERS,
) -> dict:
    """Build from combined segments or supplied frames, without publication."""
    if nights is None:
        import build_viewer as b

        segments = b.find_segments()
        nights, over = b.load_segments(segments)
        copies = b.load_copies(segments)
    return enrich(
        screen(nights, over, copies),
        site_dir=site_dir,
        session=session,
        imagery=imagery,
        max_renders=max_renders,
    )


# %%
if __name__ == "__main__":
    main()
