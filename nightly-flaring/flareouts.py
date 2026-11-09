# %%
"""Source-centred thermal-departure evidence cards (D58).

No state or score estimates physical extinction, venting or an unplanned stop.
All processing functions are deterministic and make no network requests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import betaincinv
from scipy.stats import norm

from pipeline import dedupe_overpasses
from regions import region_of

BRIGHT_MW = 3.0
PERSISTENCE_FLOOR = 0.59
FLEET_BUDGET = 0.1  # model-based budget per spell's 30-night fleet scan
SCAN_NIGHTS = 30
LIMITS = [
    "These data cannot show that a flare is out, that gas is vented, or that a stop was unplanned.",
    "A flame smaller than any detected looks like no flame. Cloud and source-location errors remain possible.",
    "The heat bound is conditional on detectable-flame re-detection rates and independent nights; it is not a physical upper limit.",
]
TIER_RULES = {
    "High": "Source baseline: preceding 29 nights, ≥12 clear nights, ≥80% with detections, median ≥3 MW; dated pre-spell heat ≥3 MW from ≥3 clear detections; ≥6 clear dark nights across ≥2 satellites; no sampled detection within 3 km; measured non-losing comparison coverage within 10 km; conservative score ≤0.1 over eligible unit-nights in the spell's own 30-night fleet scan containing its start.",
    "Mid": "The same recently bright source and satellite/absence checks, ≥6 clear dark nights, and shared heat loss or a conservative score above budget. Missing or unresolved comparison coverage caps a passing score at Mid. Independent-night scores never determine a tier.",
    "Watch": "Ongoing darkness with ≥28 clear nights, ≥12 prior detected nights, dated pre-spell heat from ≥3 clear detections, and ≥2 satellites, for sources failing the recent bright rule. Previously reduced heat is dated before the spell.",
}


def distance_m(lat, lon, other_lat, other_lon):
    """Great-circle distances, scalar or vector, including across the dateline."""
    a, b = np.radians(lat), np.radians(other_lat)
    h = (
        np.sin((b - a) / 2) ** 2
        + np.cos(a)
        * np.cos(b)
        * np.sin(np.radians(np.asarray(other_lon) - lon) / 2) ** 2
    )
    return 6371000 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _number(value):
    return round(float(value), 6) if pd.notna(value) and math.isfinite(value) else None


def prepare_passes(nights: pd.DataFrame, over: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated night passes; raw clear flags are outcome independent."""
    p = dedupe_overpasses(over)
    p = p[p.sza >= 97].copy()
    coords = nights.drop_duplicates("site_id").set_index("site_id")
    p["lon"] = p.site_id.map(coords.lon)
    p["night"] = (
        pd.to_datetime(p.time_utc) + pd.to_timedelta(p.lon / 15 - 12, "h")
    ).dt.strftime("%Y-%m-%d")
    p = p[p.night.between(nights.night.min(), nights.night.max())].copy()
    p["raw_clear"] = p.cloud_mask.isin([0, 1])
    return p


def prepare_copies(copies: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    """Retain every source group once, including unassigned/cloudy detections."""
    c = copies.drop_duplicates(["satellite", "granule", "group_id"]).copy()
    c["night"] = (
        pd.to_datetime(c.time_utc) + pd.to_timedelta(c.centroid_lon / 15 - 12, "h")
    ).dt.strftime("%Y-%m-%d")
    c["raw_clear"] = c.cloud_mask_peak.isin([0, 1])
    return c[c.night.between(first, last)].copy()


def source_units(c: pd.DataFrame, baseline_end: str) -> list[dict]:
    """Merge nearby assignment centres, with a diameter and resolved-pair guard.

    Freeze centres in the initial 29-night baseline; newly appearing sources use
    their first 29 nights. Never chain platform neighbours into a field entity.
    """
    centres = []
    for sid, g in c.dropna(subset=["assigned_site"]).groupby("assigned_site"):
        b = g[g.night <= baseline_end]
        if len(b) < 3:
            end = str((pd.Timestamp(g.night.min()) + pd.Timedelta(days=28)).date())
            b = g[g.night <= end]
        if len(b) < 3:
            continue
        centres.append(
            dict(
                site_ids=[int(sid)],
                lat=float(b.centroid_lat.median()),
                lon=float(b.centroid_lon.median()),
            )
        )
    units = []
    for centre in centres:
        for u in units:
            if not all(
                distance_m(a, b, centre["lat"], centre["lon"]) < 750
                for a, b in u["centres"]
            ):
                continue
            # Two resolved bright groups in a pass mean two independent sources.
            pair = c[
                c.assigned_site.isin(u["site_ids"] + centre["site_ids"])
                & (c.night <= baseline_end)
                & (c.group_rh_mw >= BRIGHT_MW)
            ]
            same = pair.groupby(["satellite", "granule"]).filter(lambda g: len(g) > 1)
            resolved = False
            for _, g in same.groupby(["satellite", "granule"]):
                if g.assigned_site.nunique() < 2:
                    continue
                for row in g.itertuples():
                    if (
                        distance_m(
                            row.centroid_lat,
                            row.centroid_lon,
                            g.centroid_lat,
                            g.centroid_lon,
                        )
                        >= 1000
                    ).any():
                        resolved = True
                        break
            if resolved:
                continue
            u["site_ids"] += centre["site_ids"]
            u["centres"].append((centre["lat"], centre["lon"]))
            break
        else:
            units.append({**centre, "centres": [(centre["lat"], centre["lon"])]})
    for u in units:
        centres = u.pop("centres")
        u["lat"] = float(np.median([a for a, _ in centres]))
        u["lon"] = float(np.median([b for _, b in centres]))
        # Recompute the joint centroid from baseline groups, avoiding platform IDs.
        b = c[c.assigned_site.isin(u["site_ids"]) & (c.night <= baseline_end)]
        if len(b) >= 3:
            u["lat"], u["lon"] = (
                float(b.centroid_lat.median()),
                float(b.centroid_lon.median()),
            )
        u["baseline_group_count"] = len(b)
        u["id"] = "-".join(map(str, u["site_ids"]))
        u["region"] = region_of(u["lat"], u["lon"])
    return units


def unit_history(u: dict, p: pd.DataFrame, c: pd.DataFrame, days: list[str]):
    """Source positives reset gaps even when assigned to another platform."""
    d = distance_m(u["lat"], u["lon"], c.centroid_lat, c.centroid_lon)
    cp = c[d <= 1500].copy()
    # Use the narrower heat aperture to avoid importing a resolved neighbour's
    # bright heat. The wider positive-evidence veto still resets dark spells.
    cp["heat_source"] = d[d <= 1500] <= 1000
    obs = p[p.site_id.isin(u["site_ids"])].copy()
    # A union of member site clear flags is one opportunity per satellite pass.
    obs["site_id"] = u["site_ids"][0]
    obs["detected"] = obs.raw_clear  # select a clear member exposure first
    obs = dedupe_overpasses(obs)
    q = obs.groupby("night").raw_clear.sum()
    lit = cp.groupby("night").size()
    heat = cp[cp.raw_clear & cp.heat_source].groupby("night").group_rh_mw.median()
    groups = cp[cp.raw_clear & cp.heat_source].groupby("night").size()
    h = pd.DataFrame({"night": days}).set_index("night")
    h["clear_passes"] = q.reindex(days).fillna(0).astype(int)
    h["lit"] = lit.reindex(days).fillna(0).gt(0)
    h["heat_mw"] = heat.reindex(days)
    h["clear_groups"] = groups.reindex(days).fillna(0).astype(int)
    h["qualified"] = h.clear_passes >= 1
    return h, cp, obs


def heat_states(h: pd.DataFrame) -> list[dict]:
    """Viterbi dating with normal log heat and left-censored misses.

    Three observation states: normal heat, reduced heat, unresolved darkness.
    Unknown nights have neutral emissions. Darkness is an observation regime,
    never an inferred physical extinction probability.
    """
    pos = h.heat_mw.dropna()
    if pos.empty:
        return []
    normal = float(np.log(pos.iloc[:14].median()))
    reduced = min(normal - math.log(5), float(np.log(pos.quantile(0.25))))
    means = np.array([normal, reduced, math.log(0.03)])
    sigma = max(0.65, float(np.log(pos).std()) / 2 if len(pos) > 1 else 0.65)
    transition = np.log(np.full((3, 3), 0.015) + np.eye(3) * 0.955)
    score = np.array([0.0, -4.0, -6.0])
    back = []
    for r in h.itertuples():
        if pd.notna(r.heat_mw) and r.heat_mw > 0:
            emission = norm.logpdf(math.log(r.heat_mw), means, sigma)
        elif r.lit or not r.qualified:
            emission = np.zeros(3)
        else:
            emission = norm.logcdf(math.log(0.2), means, sigma)
        alternatives = score[:, None] + transition
        back.append(alternatives.argmax(axis=0))
        score = alternatives.max(axis=0) + emission
    state = int(score.argmax())
    path = []
    for pointers in reversed(back):
        path.append(state)
        state = int(pointers[state])
    path.reverse()
    # Darkness is an observed absence, never allowed to absorb measured recovery.
    dark = ~h.lit & h.qualified
    for i, r in enumerate(h.itertuples()):
        if r.lit:
            path[i] = min(path[i], 1)
    i = 0
    while i < len(h):
        if h.iloc[i].lit:
            i += 1
            continue
        j = i
        while j < len(h) and not h.iloc[j].lit:
            j += 1
        # Only sustained gaps get a darkness row; short gaps inherit heat state.
        state = (
            2
            if int(dark.iloc[i:j].sum()) >= 3
            else (path[i - 1] if i else path[j] if j < len(h) else 0)
        )
        path[i:j] = [state] * (j - i)
        i = j
    names = ["normal heat", "reduced heat", "unresolved darkness"]
    records = []
    for night, state in zip(h.index, path):
        if not records or records[-1]["state"] != names[state]:
            records.append(dict(state=names[state], start=night, end=night))
        records[-1]["end"] = night
    for r in records:
        x = h.loc[r["start"] : r["end"]].heat_mw.dropna()
        r.update(median_mw=_number(x.median()), measured_nights=len(x))
    return records


def detection_curve(p: pd.DataFrame) -> list[dict]:
    """Same-night re-detection at another clear pass, pooled over all nights."""
    q = p[p.raw_clear].copy()
    g = q.groupby(["site_id", "night"]).detected
    q["n"] = g.transform("size")
    q["d"] = g.transform("sum")
    q = q[q.detected & (q.n > 1) & (q.rh_mw > 0)]
    edges = [0, 0.2, 0.3, 0.5, 1, 2, 5, 10, float("inf")]
    curve = []
    for low, high in zip(edges[:-1], edges[1:]):
        z = q[(q.rh_mw > low) & (q.rh_mw <= high)]
        n, d = int((z.n - 1).sum()), int((z.d - 1).sum())
        curve.append(
            dict(
                lower_mw=low,
                upper_mw=None if math.isinf(high) else high,
                trials=n,
                detections=d,
                probability=d / n if n else None,
            )
        )
    return curve


def baseline_stats(h: pd.DataFrame, end: str) -> dict:
    start = str((pd.Timestamp(end) - pd.Timedelta(days=28)).date())
    b = h.loc[start:end]
    q = b[b.qualified]
    heat = b.heat_mw.dropna()
    return dict(
        qualified_nights=len(q),
        lit_qualified_nights=int(q.lit.sum()),
        lit_fraction=float(q.lit.mean()) if len(q) else 0,
        median_mw=_number(heat.median()),
        clear_groups=int(b.clear_groups.sum()),
        start=start,
        end=end,
        transitions=transition_counts(b),
    )


def eligible(b: dict) -> bool:
    return (
        b["qualified_nights"] >= 12
        and b["lit_fraction"] >= 0.8
        and (b["median_mw"] or 0) >= BRIGHT_MW
    )


def transition_counts(h: pd.DataFrame) -> dict:
    """Transitions on this source's clear opportunities, with cloudy resets."""
    counts = dict(MM=0, MH=0, HM=0, HH=0)
    previous = None
    for r in h.itertuples():
        if r.qualified:
            current = "H" if r.lit else "M"
            if previous is not None:
                counts[previous + current] += 1
            previous = current
        elif r.lit:
            previous = "H"
    return counts


def calibrate(histories: dict, baseline_end: str, p: pd.DataFrame) -> dict:
    """Refit reference transitions in nonoverlapping 29-night stable blocks.

    Blocks must independently meet the bright baseline rule. New months update
    transition uncertainty, while persistence never falls below the R3 floor.
    """
    mm = mh = hm = hh = blocks = 0
    anchors = 0
    for h in histories.values():
        anchors += eligible(baseline_stats(h, baseline_end))
        for i in range(0, len(h) - 11, 29):
            block = h.iloc[i : i + 29]
            if not eligible(baseline_stats(block, block.index[-1])):
                continue
            blocks += 1
            previous = None
            for row in block.itertuples():
                if row.lit and not row.qualified:
                    previous = True  # positive reset, not an extra trial
                elif row.qualified:
                    if previous is not None:
                        mm += int(not previous and not row.lit)
                        mh += int(not previous and row.lit)
                        hm += int(previous and not row.lit)
                        hh += int(previous and row.lit)
                    previous = bool(row.lit)
    upper = float(betaincinv(mm + 1, mh + 1, 0.975))
    return dict(
        baseline_end=baseline_end,
        baseline_nights=29,
        bright_units=anchors,
        reference_blocks=blocks,
        transitions=dict(MM=mm, MH=mh, HM=hm, HH=hh),
        miss_persistence_point=mm / (mm + mh) if mm + mh else None,
        miss_persistence_upper=upper,
        miss_persistence_used=max(PERSISTENCE_FLOOR, upper),
        persistence_floor=PERSISTENCE_FLOOR,
        fleet_budget=FLEET_BUDGET,
        budget_scope="Eligible unit-nights in each spell's own 30-night window containing its start",
        detection_curve=detection_curve(p),
        score_limit="Model sensitivity scores, not validated false-alarm or extinction probabilities.",
    )


def dark_spells(h: pd.DataFrame, after: str) -> list[dict]:
    spells, start = [], None
    last = None
    for night, r in h.loc[after:].iterrows():
        if r.lit:
            if start:
                spells.append(
                    dict(start=start, end=last, status="ended", recovery=night)
                )
                start = None
        else:
            start = start or night
        last = night
    if start:
        spells.append(dict(start=start, end=last, status="ongoing", recovery=None))
    return spells


def spell_scan_family(spell: dict, eligible_counts: dict[str, int]) -> dict:
    """Anchor a 30-night family to this spell, preserving its tested start.

    End at the last dark night (as-of while ongoing), until the spell exceeds
    30 nights; then keep its first 30 nights for both ongoing and ended spells.
    Floor F at one tested unit per available evaluation night in this window.
    """
    end = min(
        pd.Timestamp(spell["end"]),
        pd.Timestamp(spell["start"]) + pd.Timedelta(days=SCAN_NIGHTS - 1),
    )
    start = str((end - pd.Timedelta(days=SCAN_NIGHTS - 1)).date())
    end = str(end.date())
    counts = [
        count for night, count in eligible_counts.items() if start <= night <= end
    ]
    if not counts:
        raise ValueError("A scored spell must have evaluation nights in its own window")
    total = sum(counts)
    return dict(
        scan_window_start=start,
        scan_window_end=end,
        scan_window_nights=SCAN_NIGHTS,
        evaluation_nights=len(counts),
        eligible_unit_nights=total,
        family_floor=len(counts),
        family_unit_nights=max(len(counts), total),
    )


def heat_bound(k: int, curve: list[dict]) -> dict:
    classes = []
    for r in curve:
        if r["probability"] is not None:
            classes.append({**r, "tail": (1 - r["probability"]) ** k})
    # Require the chosen class and every brighter class to pass the same gate.
    excluded = [
        r for i, r in enumerate(classes) if all(z["tail"] <= 0.05 for z in classes[i:])
    ]
    lower = excluded[0]["lower_mw"] if excluded else None
    return dict(
        excluded_from_mw=lower,
        all_detected_classes_excluded=lower == 0,
        classes=classes,
        assumption="One clear opportunity per night, independent nights; conditioned on flames detectable at least once. Below the smallest detected class is untested.",
    )


def classify_tier(
    bright: bool,
    shared: bool,
    conservative: float,
    k: int,
    isolation_resolved: bool = False,
) -> str:
    if bright and k >= 6:
        if isolation_resolved and not shared and conservative <= FLEET_BUDGET:
            return "High"
        return "Mid"
    return "Watch"


def shared_context(u: dict, spell: dict, units: list, histories: dict) -> dict:
    """Independent source heat changes at 10, 50 and 100 km, excluding self."""
    neighbours = []
    start, end = spell["start"], spell["end"]
    pre = str((pd.Timestamp(start) - pd.Timedelta(days=14)).date())
    for v in units:
        if v["id"] == u["id"]:
            continue
        km = float(distance_m(u["lat"], u["lon"], v["lat"], v["lon"])) / 1000
        if km > 100 or km <= 3:  # overlapping apertures are not independent donors
            continue
        h = histories[v["id"]]
        b = h.loc[pre:start].iloc[:-1].heat_mw.dropna()
        during = h.loc[start:end]
        if len(b) < 3 or b.median() < BRIGHT_MW:
            continue
        informative = during.heat_mw.notna() | (during.qualified & ~during.lit)
        low = informative & (
            (during.qualified & ~during.lit) | (during.heat_mw < b.median() * 0.2)
        )
        # Shared loss uses measured nights only. To claim absence of shared
        # loss, even all unknown nights losing heat must be unable to meet the
        # ≥3 and ≥half-spell rule; half coverage alone cannot establish this.
        measured = int(informative.sum())
        shared = int(low.sum()) >= 3 and low.sum() >= measured / 2
        unknown = len(during) - measured
        possible_low = int(low.sum()) + unknown
        unresolved_loss = possible_low >= 3 and possible_low >= len(during) / 2
        resolved = measured >= 3 and not unresolved_loss
        heat = during.heat_mw.dropna()
        low_dates = during.index[low]
        neighbours.append(
            dict(
                unit=v,
                distance_km=round(km, 3),
                baseline_mw=_number(b.median()),
                baseline_detected_nights=len(b),
                spell_median_mw=_number(heat.median()),
                shared_heat_loss=bool(shared),
                measured_nights=int(informative.sum()),
                coverage_resolved=bool(resolved),
                loss_start=low_dates.min() if len(low_dates) else None,
                loss_end=low_dates.max() if len(low_dates) else None,
                **(dict(series=series_records(h)) if km <= 10 else {}),
            )
        )
    # Overlapping donor apertures cannot supply independent evidence. Keep the
    # complex best supported in the frozen baseline, and retain aliases for audit.
    independent = []
    for n in sorted(neighbours, key=lambda x: -x["unit"]["baseline_group_count"]):
        overlapping = next(
            (
                old
                for old in independent
                if distance_m(
                    n["unit"]["lat"],
                    n["unit"]["lon"],
                    old["unit"]["lat"],
                    old["unit"]["lon"],
                )
                <= 3000
            ),
            None,
        )
        if overlapping is not None:
            overlapping.setdefault("overlapping_units", []).append(n["unit"])
        else:
            independent.append(n)
    neighbours = independent
    neighbours.sort(key=lambda x: (not x["shared_heat_loss"], x["distance_km"]))
    chart_neighbours = 0
    for neighbour in neighbours:
        if "series" in neighbour:
            chart_neighbours += 1
            if chart_neighbours > 4:
                neighbour.pop("series")
    scales = {
        str(r): dict(
            donors=sum(n["distance_km"] <= r for n in neighbours),
            shared=sum(
                n["distance_km"] <= r and n["shared_heat_loss"] for n in neighbours
            ),
        )
        for r in (10, 50, 100)
    }
    local = [n for n in neighbours if n["distance_km"] <= 10]
    resolved = bool(local) and all(n["coverage_resolved"] for n in local)
    label = (
        "shared heat loss"
        if scales["10"]["shared"]
        else "no comparison flare within 10 km"
        if not local
        else "isolated"
        if resolved
        else "comparison heat unresolved"
    )
    return dict(
        label=label,
        isolation_resolved=resolved,
        scales_km=scales,
        neighbours=neighbours,
        limit="Geographic co-change does not identify a shared gas system or cause. Isolation requires ≥3 measured nights at each comparison flare within 10 km and enough non-losing coverage to rule out shared loss even if every unknown night lost heat; cloud-only detections are unknown.",
    )


def series_records(h: pd.DataFrame) -> list[dict]:
    return [
        dict(
            night=r.Index,
            heat_mw=_number(r.heat_mw),
            lit=bool(r.lit),
            qualified=bool(r.qualified),
            clear_passes=int(r.clear_passes),
        )
        for r in h.itertuples()
    ]


def screen(nights: pd.DataFrame, over: pd.DataFrame, copies: pd.DataFrame) -> dict:
    """Build JSON-serialisable candidate evidence from the combined run frames."""
    days = (
        pd.date_range(nights.night.min(), nights.night.max())
        .strftime("%Y-%m-%d")
        .tolist()
    )
    # The first night may be partial. Preserve the research's 29 full-night base.
    first = str((pd.Timestamp(days[0]) + pd.Timedelta(days=1)).date())
    baseline_end = str((pd.Timestamp(first) + pd.Timedelta(days=28)).date())
    p = prepare_passes(nights, over)
    c = prepare_copies(copies, days[0], days[-1])
    units = source_units(c, baseline_end)
    histories, sources, exposures = {}, {}, {}
    for u in units:
        h, cp, obs = unit_history(u, p, c, days)
        histories[u["id"]], sources[u["id"]], exposures[u["id"]] = h, cp, obs
    calibration = calibrate(histories, baseline_end, p)
    # Keep eligibility for historical windows as well as the current one.
    # Every spell selects its own fixed-duration family containing its start.
    scan_counts = {}
    scan_days = days[30:]
    for day in scan_days:
        end = str((pd.Timestamp(day) - pd.Timedelta(days=1)).date())
        scan_counts[day] = sum(
            eligible(baseline_stats(h, end)) for h in histories.values()
        )
    calibration.update(
        eligible_units_by_night=scan_counts,
        scan_window_nights=SCAN_NIGHTS,
        eligibility="Each source's own 29 nights immediately before each scan; frozen before scoring.",
        family_definition="Each spell has its own 30-night fleet window containing its start: end at its last dark night (as-of while ongoing); for spells longer than 30 nights, keep the first 30 nights. Sum eligible source units in that window, with a floor of one tested unit per available evaluation night (at most 30); budget 0.1 per such scan. Ended spells keep their window on later builds.",
    )
    candidates = []
    veto_count = 0
    for u in units:
        h, cp = histories[u["id"]], sources[u["id"]]
        states = heat_states(h)
        if not h.lit.any():
            continue
        after = max(
            str((pd.Timestamp(h.index[h.lit][0]) + pd.Timedelta(days=1)).date()),
            str((pd.Timestamp(baseline_end) + pd.Timedelta(days=1)).date()),
        )
        for spell in dark_spells(h, after):
            g = h.loc[spell["start"] : spell["end"]]
            k = int(g.qualified.sum())
            if k < 6:
                continue
            baseline_end_spell = str(
                (pd.Timestamp(spell["start"]) - pd.Timedelta(days=1)).date()
            )
            b = baseline_stats(h, baseline_end_spell)
            dated = next(
                (
                    r
                    for r in reversed(heat_states(h.loc[:baseline_end_spell]))
                    if r["end"] < spell["start"] and r["state"] != "unresolved darkness"
                ),
                None,
            )
            # Use the dated heat regime immediately before the spell, restricted
            # to the prior week so an old bright regime cannot override a turndown.
            pre_start = str(
                (pd.Timestamp(spell["start"]) - pd.Timedelta(days=7)).date()
            )
            if dated:
                pre_start = max(pre_start, dated["start"])
            pre = cp[
                cp.night.between(pre_start, baseline_end_spell)
                & cp.raw_clear
                & cp.heat_source
            ]
            preheat = _number(pre.group_rh_mw.median())
            if preheat is None or len(pre) < 3:
                continue
            bright = eligible(b) and preheat >= BRIGHT_MW
            obs = exposures[u["id"]]
            obs = obs[obs.night.between(spell["start"], spell["end"])].copy()
            satellites = sorted(obs.loc[obs.raw_clear, "satellite"].unique().tolist())
            if len(satellites) < 2:
                continue
            transitions = b["transitions"]
            first_miss = (transitions["HM"] + 1) / (
                transitions["HH"] + transitions["HM"] + 2
            )
            tail = first_miss * calibration["miss_persistence_used"] ** (k - 1)
            family = spell_scan_family(spell, scan_counts)
            conservative = tail * family["family_unit_nights"]
            watch = (
                spell["status"] == "ongoing"
                and k >= 28
                and int(h.loc[:baseline_end_spell].lit.sum()) >= 12
                and not bright
            )
            if not watch and not bright:
                continue
            nearby = c[c.night.between(spell["start"], spell["end"])]
            d = distance_m(u["lat"], u["lon"], nearby.centroid_lat, nearby.centroid_lon)
            # Wider-aperture evidence prevents treating localisation ambiguity
            # as a bright complete absence. The narrow aperture reset is above.
            if (d <= 3000).any():
                veto_count += 1
                continue
            context = shared_context(u, spell, units, histories)
            tier = classify_tier(
                bright,
                context["label"] == "shared heat loss",
                conservative,
                k,
                context["isolation_resolved"],
            )
            why = [
                f"{k} clear nights with no flare seen across {len(satellites)} satellites. No detection within 3 km in the sampled clusters (searched within ±2 pixels of GFW sites).",
                f"Heat before darkness: {preheat:.2f} MW (n={len(pre)} clear detections), from the dated {dated['state'] if dated else 'heat'} regime.",
                f"{context['label'].capitalize()}; this describes measured heat, not the cause.",
            ]
            candidates.append(
                dict(
                    id=f"{u['id']}_{spell['start']}",
                    unit=u,
                    tier=tier,
                    **spell,
                    baseline=b,
                    pre_spell_state=dated,
                    pre_gap_heat_mw=preheat,
                    pre_gap_groups=len(pre),
                    dark_qualified_nights=k,
                    clear_passes=int(g.clear_passes.sum()),
                    satellites=satellites,
                    score=dict(
                        markov_tail=tail,
                        conservative_fleet=conservative,
                        budget_multiple=conservative / FLEET_BUDGET,
                        first_miss=first_miss,
                        **family,
                    ),
                    heat_upper_bound=heat_bound(k, calibration["detection_curve"]),
                    heat_state_history=states,
                    context=context,
                    nightly_series=series_records(h),
                    pass_list=[
                        dict(
                            time_utc=r.time_utc,
                            night=r.night,
                            satellite=r.satellite,
                            clear=bool(r.raw_clear),
                            status="clear miss" if r.raw_clear else "cloudy",
                            cloud_mask=int(r.cloud_mask)
                            if pd.notna(r.cloud_mask)
                            else None,
                        )
                        for r in obs.sort_values("time_utc").itertuples()
                    ],
                    viirs_times=dict(
                        last_lit=cp.loc[cp.night < spell["start"], "time_utc"].max(),
                        first_dark=obs.loc[obs.raw_clear, "time_utc"].min(),
                        last_dark=obs.loc[obs.raw_clear, "time_utc"].max(),
                        recovery=cp.loc[cp.night == spell["recovery"], "time_utc"].min()
                        if spell["recovery"]
                        else None,
                    ),
                    source_absence=dict(
                        radius_m=1500,
                        sensitivity_radius_m=3000,
                        nearest_recorded_source_m=_number(d.min()),
                    ),
                    why_flagged=why,
                    limits=LIMITS,
                )
            )
    candidates.sort(
        key=lambda x: (
            ["High", "Mid", "Watch"].index(x["tier"]),
            x["score"]["conservative_fleet"],
            x["id"],
        )
    )
    # One event per evidence aperture. Prefer actual baseline support over
    # satellite assignment fragments, then restore tier/score ranking.
    unique = []
    for card in sorted(candidates, key=lambda x: -x["unit"]["baseline_group_count"]):
        owner = next(
            (
                old
                for old in unique
                if distance_m(
                    card["unit"]["lat"],
                    card["unit"]["lon"],
                    old["unit"]["lat"],
                    old["unit"]["lon"],
                )
                <= 1500
                and card["start"] <= old["end"]
                and old["start"] <= card["end"]
            ),
            None,
        )
        if owner:
            owner.setdefault("suppressed_units", []).append(card["unit"]["site_ids"])
        else:
            unique.append(card)
    unique.sort(
        key=lambda x: (
            ["High", "Mid", "Watch"].index(x["tier"]),
            x["score"]["conservative_fleet"],
            x["id"],
        )
    )
    return dict(
        as_of=days[-1],
        calibration=calibration,
        tier_rules=TIER_RULES,
        candidates=unique,
        source_veto_count=veto_count,
        limits=LIMITS,
    )


def main(nights=None, over=None, copies=None) -> dict:
    """Run locally or supply combined frames for interactive/reusable builds."""
    if nights is None:
        import build_viewer as b

        segments = b.find_segments()
        nights, over = b.load_segments(segments)
        copies = b.load_copies(segments)
    return screen(nights, over, copies)


# %%
if __name__ == "__main__":
    main()
