# %%
"""Write the Potential flare-outs figures for docs/explainer.md (Figures 20-25).

The charts read the screen's payload, ``outputs/site/flareouts.json``, which
``build_viewer.main()`` writes (as of local night 4 November 2026); Figure 22's
first-miss band re-reads the v5 segments. They share
``make_figures``' primitives and night palette. ``chips_emit.png`` renders two
EMIT scenes with the Flare-outs tab's recipe (``emit.fetch_chips``: Earth
Engine, cached in ``outputs/emit_cache``); pass ``chips=False`` to skip it
offline. Run from ``nightly-flaring/``:
``.venv/bin/python docs/explainer/make_flareout_figures.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import make_figures as F  # noqa: E402
from make_figures import (  # noqa: E402
    AMBER,
    BG,
    CLEAR,
    CLOUD,
    COAST,
    CYAN,
    FL,
    GOLD,
    INK,
    INK2,
    INK3,
    PANEL,
    RAISE,
    RULE,
    Lin,
    Log,
    arrow,
    circle,
    halo,
    line,
    lines,
    path,
    rect,
    svg,
    text,
)

PAYLOAD = ROOT / "outputs" / "site" / "flareouts.json"
EXAMPLE = "4582-4583_2026-10-21"  # Mid: bright source, shared heat loss nearby
SHORT = "927_2026-10-30"  # Mid: six clear dark nights, no comparison flare
TIER_FILL = {"High": FL[2], "Mid": FL[4], "Watch": INK3}  # locator dot colours
# Two scenes before Earth Engine's EMIT cutoff (31 Aug 2026) at screen sources.
EMIT_EXAMPLES = [
    ("927_2026-10-30", "20260621T173441_2617211_045", "2026-06-21T17:34:41.000Z", 5),
    ("3644_2026-09-16", "20260728T082818_2620906_003", "2026-07-28T08:28:18.000Z", 0),
]


def load() -> dict:
    if not PAYLOAD.exists():
        raise FileNotFoundError(
            f"{PAYLOAD} is missing; run build_viewer.main() first"
        )
    return json.loads(PAYLOAD.read_text())


def card(d: dict, cid: str) -> dict:
    return next(c for c in d["candidates"] if c["id"] == cid)


def name(c: dict) -> str:
    return "/".join(str(s) for s in c["unit"]["site_ids"])


def md(night: str) -> str:
    return f"{pd.Timestamp(night):%-d %b}"


def status(r: dict) -> str:
    if r["lit"] and r["heat_mw"] is not None:
        return "heat"
    if r["lit"]:
        return "cloudlit"
    return "miss" if r["qualified"] else "cloud"


# ------------------------------------------------------------ figure 20
def fig_record(d):
    """One card night by night: the source and its comparison flare."""
    c = card(d, EXAMPLE)
    nb = next(n for n in c["context"]["neighbours"] if n["shared_heat_loss"])
    nights = [r["night"] for r in c["nightly_series"]]
    idx = {n: i for i, n in enumerate(nights)}
    x0, x1 = 96, 734
    xs = Lin(-0.6, len(nights) - 0.4, x0, x1)
    cw = xs(1) - xs(0)
    sx0, sx1 = xs(idx[c["start"]]) - cw / 2, xs(idx[c["end"]]) + cw / 2
    b = c["baseline"]
    body = ""

    def panel(top, series, title, sub):
        yt, yb = top + 34, top + 160
        lane = yb + 22
        ys = Log(0.1, 100, yb, yt)
        out = text(20, top + 12, title, size=13.5, fill=INK, weight=650)
        out += text(20, top + 28, sub, size=12, fill=INK2)
        out += F.frame(x0, yt, x1, yb)
        for v in (0.1, 1, 10, 100):
            out += line(x0, ys(v), x1, ys(v), stroke=RULE)
            out += text(x0 - 8, ys(v) + 4, F.sup(v), size=11, fill=INK3, anchor="end")
        mid = (yt + yb) / 2
        out += text(30, mid, "MW", size=11, fill=INK3, anchor="middle", transform=f"rotate(-90 30 {mid})")
        out += text(x0 - 8, lane + 4, "not seen", size=11, fill=INK3, anchor="end")
        run = []
        for r in series:
            i, s = idx[r["night"]], status(r)
            tip = f"{r['night']}: "
            if s == "heat":
                run.append((xs(i), ys(max(r["heat_mw"], 0.1))))
                continue
            if len(run) > 1:
                out += path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in run), stroke=AMBER, w=1, opacity=0.5)
            run = []
            if s == "miss":
                out += circle(xs(i), lane, 2.6, fill=CLEAR, tip=tip + "clear, no flare seen")
            elif s == "cloudlit":
                out += circle(xs(i), lane, 2.6, fill=BG, stroke=AMBER, sw=1.2, tip=tip + "seen through cloud")
            else:
                out += rect(xs(i) - cw / 2 + 0.8, lane - 4, cw - 1.6, 8, fill=CLOUD, rx=1.5)
        if len(run) > 1:
            out += path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in run), stroke=AMBER, w=1, opacity=0.5)
        for r in series:
            if status(r) == "heat":
                out += circle(xs(idx[r["night"]]), ys(max(r["heat_mw"], 0.1)), 2.6, fill=AMBER, tip=f"{r['night']}: {r['heat_mw']:.2f} MW")
        return out, ys, yt, lane

    top1 = 46
    p1, ys1, yt1, lane1 = panel(
        top1,
        c["nightly_series"],
        f"Source {name(c)}, Persian Gulf",
        "The flare the card is about",
    )
    top2 = lane1 + 34
    p2, ys2, yt2, lane2 = panel(
        top2,
        nb["series"],
        f"Comparison flare {name(nb)}, {nb['distance_km']:.1f} km away",
        "A bright neighbour within 10 km, measured on clear nights",
    )
    # dark spell band behind both panels
    body += rect(sx0, yt1 - 6, sx1 - sx0, lane2 + 10 - (yt1 - 6), fill=FL[3], opacity=0.1)
    body += p1 + p2
    k = c["dark_qualified_nights"]
    body += halo((sx0 + sx1) / 2, yt1 + 14, f"{k} clear nights,", size=11.5, fill=FL[4], anchor="middle")
    body += halo((sx0 + sx1) / 2, yt1 + 28, "no flare seen", size=11.5, fill=FL[4], anchor="middle")
    # 29-night baseline bracket above panel 1
    bx0, bx1 = xs(idx[b["start"]]) - cw / 2, xs(idx[b["end"]]) + cw / 2
    by = top1 + 4
    body += path(f"M{bx0:.1f},{by + 6:.1f} L{bx0:.1f},{by:.1f} L{bx1:.1f},{by:.1f} L{bx1:.1f},{by + 6:.1f}", stroke=CYAN, w=1.2)
    body += text(bx1, by - 6, f"29 nights before: seen on {b['lit_qualified_nights']} of {b['qualified_nights']} clear nights, median {b['median_mw']:.1f} MW", size=11.5, fill=CYAN, anchor="end")
    # recovery note
    if c["recovery"]:
        r = next(r for r in c["nightly_series"] if r["night"] == c["recovery"])
        rx, ry = xs(idx[c["recovery"]]), ys1(max(r["heat_mw"], 0.1))
        body += arrow(rx - 38, ry - 26, rx - 5, ry - 4, color=INK2, w=1.2, head=6)
        body += halo(rx - 40, ry - 30, f"seen again {md(c['recovery'])}: {r['heat_mw']:.2f} MW", size=11.5, fill=INK2, anchor="end")
    # comparison flare: before and during medians
    cx0 = xs(idx[c["start"]] - 14) - cw / 2
    cy = top2 + 4
    body += path(f"M{cx0:.1f},{cy + 6:.1f} L{cx0:.1f},{cy:.1f} L{sx0:.1f},{cy:.1f} L{sx0:.1f},{cy + 6:.1f}", stroke=INK2, w=1.2)
    body += text(sx0, cy - 6, f"14 nights before: median {nb['baseline_mw']:.1f} MW", size=11.5, fill=INK2, anchor="end")
    body += halo((sx0 + sx1) / 2, ys2(nb["spell_median_mw"]) + 22, f"median {nb['spell_median_mw']:.2f} MW", size=11.5, fill=FL[4], anchor="middle")
    # dates on the shared axis
    for i, n in enumerate(nights):
        t = pd.Timestamp(n)
        if t.day in (1, 15):
            body += line(xs(i), lane2 + 12, xs(i), lane2 + 16, stroke=INK3)
            body += text(xs(i), lane2 + 29, f"{t:%-d %b}", size=11, fill=INK3, anchor="middle")
    ly = lane2 + 56
    keys = [
        (circle(0, 0, 3.4, fill=AMBER), "flare seen, clear sky (night's MW)"),
        (circle(0, 0, 3.2, fill=BG, stroke=AMBER, sw=1.4), "seen through cloud"),
        (circle(0, 0, 3.2, fill=CLEAR), "clear, no flare seen"),
        (rect(-4, -4, 8, 8, fill=CLOUD, rx=1.5), "cloud only: unknown, skipped"),
        (rect(-5, -5, 10, 10, fill=FL[3], opacity=0.25), "dark spell"),
    ]
    for j, (mark, lab) in enumerate(keys):
        x, yy = 20 + (j % 3) * 245, ly + (j // 3) * 20
        body += f'<g transform="translate({x + 6},{yy - 4})">{mark}</g>' + text(x + 20, yy, lab, size=11.5, fill=INK2)
    svg("fo_record", ly + 36, f"Source {name(c)} and comparison flare {name(nb)}, night by night", body)


# ------------------------------------------------------------ figure 21
def fig_tiers(d):
    """Decision flow from a dark spell to High, Mid, Watch or no card."""
    counts = {t: [c for c in d["candidates"] if c["tier"] == t] for t in ("High", "Mid", "Watch")}
    body = ""
    lx, rx, bw = 20, 400, 340

    def box(x, y, w, h, title, rows, fill=PANEL, stroke=RULE, tcol=INK):
        out = rect(x, y, w, h, fill=fill, stroke=stroke, rx=9)
        out += text(x + 14, y + 22, title, size=13, fill=tcol, weight=650)
        out += lines(x + 14, y + 41, rows, size=11.5, fill=INK2, lh=1.38)
        return out

    def drop(x, y, w, h, label):
        out = rect(x, y, w, h, fill=BG, stroke=RULE, rx=8, stroke_dasharray="4 3")
        return out + text(x + w / 2, y + h / 2 + 4, label, size=11.5, fill=INK3, anchor="middle")

    def yes_no(x, y, s, anchor="start"):
        return text(x, y, s, size=11, fill=INK3, anchor=anchor, weight=600)

    y = 18
    body += box(lx, y, 720, 62, "A dark spell", [
        "Nights after the source was last seen, until it is seen again; a detection through cloud also ends it.",
        "Only clear nights count as dark. Cloud-only nights are skipped, not counted.",
    ])
    y += 62
    body += arrow(lx + bw / 2, y + 2, lx + bw / 2, y + 22)
    y += 24
    body += box(lx, y, bw, 82, "Enough clear evidence?", [
        "6 or more clear nights with no flare seen,",
        "from 2 or more satellites, and heat measured",
        "just before the spell (3+ clear detections)",
    ])
    body += arrow(lx + bw + 2, y + 41, rx - 4, y + 41) + yes_no(lx + bw + 10, y + 34, "no")
    body += drop(rx, y + 21, 200, 40, "no card")
    y += 82
    body += arrow(lx + bw / 2, y + 2, lx + bw / 2, y + 22) + yes_no(lx + bw / 2 + 8, y + 16, "yes")
    y += 24
    veto = d["source_veto_count"]
    body += box(lx, y, bw, 64, "Clear of nearby heat?", [
        "No hot cluster within 3 km of the source",
        "during the spell",
    ])
    body += arrow(lx + bw + 2, y + 32, rx - 4, y + 32) + yes_no(lx + bw + 10, y + 25, "no")
    body += drop(rx, y + 12, 200, 40, f"no card ({veto} on 4 Nov)")
    y += 64
    body += arrow(lx + bw / 2, y + 2, lx + bw / 2, y + 22) + yes_no(lx + bw / 2 + 8, y + 16, "yes")
    y += 24
    body += box(lx, y, 720, 64, "Recently bright and steady?", [
        "In its 29 nights before: 12 or more clear nights, flare seen on 80% or more of them, median 3 MW or more;",
        "and 3 MW or more in the days just before the spell.",
    ])
    y += 64
    body += arrow(lx + bw / 2, y + 2, lx + bw / 2, y + 22) + yes_no(lx + bw / 2 + 8, y + 16, "yes")
    body += arrow(rx + bw / 2, y + 2, rx + bw / 2, y + 22) + yes_no(rx + bw / 2 + 8, y + 16, "no")
    y += 24
    body += box(lx, y, bw, 82, "Isolated and long enough?", [
        "Comparison flares within 10 km were measured",
        "and kept their heat, and the spell passes the",
        "High threshold (Figure 22)",
    ])
    body += box(rx, y, bw, 82, "Long, ongoing darkness?", [
        "Still dark, 28 or more clear dark nights,",
        "and seen on 12 or more earlier nights",
    ])
    y += 82
    hy = y + 24
    body += arrow(lx + 80, y + 2, lx + 80, hy - 2) + yes_no(lx + 88, y + 16, "yes")
    body += arrow(lx + 255, y + 2, lx + 255, hy - 2) + yes_no(lx + 263, y + 16, "no")
    body += arrow(rx + 80, y + 2, rx + 80, hy - 2) + yes_no(rx + 88, y + 16, "yes")
    body += arrow(rx + 255, y + 2, rx + 255, hy - 2) + yes_no(rx + 263, y + 16, "no")

    def tier(x, w, t, rows):
        n = len(counts[t])
        out = rect(x, hy, w, 92, fill=PANEL, stroke=TIER_FILL[t], sw=1.6, rx=9)
        out += circle(x + 20, hy + 20, 6, fill=TIER_FILL[t])
        out += text(x + 34, hy + 25, f"{t}", size=14, fill=INK, weight=700)
        out += text(x + w - 12, hy + 25, f"{n} on 4 Nov", size=12, fill=TIER_FILL[t] if t != "Watch" else INK2, anchor="end", weight=600)
        out += lines(x + 14, hy + 47, rows, size=11.5, fill=INK2, lh=1.38)
        return out

    mids = counts["Mid"]
    mid_rows = [f"{name(c)}: {'heat fell at a flare nearby' if c['context']['label'] == 'shared heat loss' else 'no comparison flare in 10 km'}" for c in mids][:3]
    body += tier(lx, 160, "High", ["Strongest evidence;", "none yet"])
    body += tier(lx + 176, 164 + 0, "Mid", ["Bright source; spell", "shorter, or not isolated"])
    body += tier(rx, 160, "Watch", ["Smaller source,", "long darkness"])
    body += drop(rx + 176, hy + 26, 164, 40, "no card")
    ny = hy + 92 + 26
    body += text(lx, ny, "On 4 Nov: " + "; ".join(mid_rows) + " (Mid). Watch: " + ", ".join(name(c) for c in counts["Watch"]) + ".", size=11.5, fill=INK3)
    svg("fo_tiers", ny + 18, "How a dark spell becomes a High, Mid or Watch card", body)


# ------------------------------------------------------------ figure 22
def first_miss_range(as_of: str) -> tuple[float, float, int]:
    """First-miss rates of the sources passing the bright, steady rule at as-of."""
    import build_viewer as BV
    import flareouts as fo

    segments = BV.find_segments()
    nights, over = BV.load_segments(segments)
    copies = BV.load_copies(segments)
    days = pd.date_range(nights.night.min(), nights.night.max()).strftime("%Y-%m-%d").tolist()
    first = str((pd.Timestamp(days[0]) + pd.Timedelta(days=1)).date())
    baseline_end = str((pd.Timestamp(first) + pd.Timedelta(days=28)).date())
    p = fo.prepare_passes(nights, over)
    c = fo.prepare_copies(copies, days[0], days[-1])
    rates = []
    for u in fo.source_units(c, baseline_end):
        b = fo.baseline_stats(fo.unit_history(u, p, c, days)[0], as_of)
        if fo.eligible(b):
            t = b["transitions"]
            rates.append((t["HM"] + 1) / (t["HH"] + t["HM"] + 2))
    return min(rates), max(rates), len(rates)


def fig_confidence(d, rates):
    """Confidence that a dark spell is not chance misses, by clear dark nights."""
    c = card(d, EXAMPLE)
    sc, cal = c["score"], d["calibration"]
    p1, q, fam = sc["first_miss"], cal["miss_persistence_used"], sc["family_unit_nights"]
    lo, hi, n_src = rates
    x0, x1, y0, y1 = 64, 556, 54, 330
    kmax = 30
    xs = Lin(1, kmax, x0, x1)
    ys = Lin(0, 1, y1, y0)

    def source(k, p):  # one flare: 1 - chance of k clear misses in a row if burning
        return 1 - p * q ** (k - 1)

    def fleet(k, p, f=fam):  # every bright source-night screened in the window
        return 1 - min(1.0, f * p * q ** (k - 1))

    def independent(k, p, f=fam):
        return 1 - min(1.0, f * p**k)

    ks = [1 + i / 4 for i in range(4 * (kmax - 1) + 1)]

    def curve(fn, p, color, w=2.2, dash=None):
        pts = " L".join(f"{xs(k):.1f},{ys(fn(k, p)):.1f}" for k in ks)
        return path("M" + pts, stroke=color, w=w, stroke_dasharray=dash)

    def band(fn, color):
        top = [(xs(k), ys(fn(k, lo))) for k in ks]
        bot = [(xs(k), ys(fn(k, hi))) for k in reversed(ks)]
        d_ = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in top + bot) + " Z"
        return path(d_, stroke="none", w=0, fill=color, opacity=0.18)

    body = text(20, 24, "Confidence that a dark spell is more than missed detections", size=13, fill=INK, weight=650)
    body += text(20, 42, "Tested against a model in which the flare kept burning at its usual heat; misses allowed to cluster", size=11.5, fill=INK2)
    body += F.frame(x0, y0, x1, y1)
    for v in (0, 0.25, 0.5, 0.75, 1):
        body += line(x0, ys(v), x1, ys(v), stroke=RULE)
        body += text(x0 - 8, ys(v) + 4, f"{v:.0%}", size=11, fill=INK3, anchor="end")
    for k in (1, 5, 10, 15, 20, 25, 30):
        body += line(xs(k), y1, xs(k), y1 + 4, stroke=COAST)
        body += text(xs(k), y1 + 17, k, size=11, fill=INK3, anchor="middle")
    body += text((x0 + x1) / 2, y1 + 38, "clear nights in a row with no flare seen", size=12, fill=INK3, anchor="middle")
    # tier rules
    for k, lab in [(6, "Mid: 6 or more"), (28, "Watch: 28 or more")]:
        body += line(xs(k), y0, xs(k), y1, stroke=INK3, w=1, dash="3 4")
        body += halo(xs(k) + (5 if k < 20 else -5), y1 - 10, lab, size=11.5, fill=INK2, anchor="start" if k < 20 else "end")
    body += line(x0, ys(0.9), x1, ys(0.9), stroke=CYAN, w=1.4, dash="6 4")
    body += halo(x1 - 4, ys(0.9) - 8, "High: 90% or more across the fleet", size=11.5, fill=CYAN, anchor="end")
    body += band(source, CYAN) + band(fleet, AMBER)
    body += curve(independent, p1, INK3, w=1.6, dash="5 4")
    body += curve(source, p1, CYAN) + curve(fleet, p1, AMBER)
    body += halo(xs(1.6), ys(0.86), "this flare alone", size=12, fill=CYAN, weight=600)
    body += halo(xs(4.2), ys(0.5), "fleet, if misses", size=11.5, fill=INK2)
    body += halo(xs(4.2), ys(0.5) + 15, "were independent", size=11.5, fill=INK2)
    kf = 10.4
    body += halo(xs(kf) + 8, ys(fleet(kf, p1)) + 4, "across all bright flares screened", size=12, fill=AMBER, weight=600)
    k = c["dark_qualified_nights"]
    body += circle(xs(k), ys(fleet(k, p1)), 6.5, fill="none", stroke=INK, sw=1.6)
    body += halo(xs(k) + 10, ys(fleet(k, p1)) + 18, f"{name(c)}: {k} nights, {fleet(k, p1):.1%}", size=11.5, fill=INK)
    # notes column
    nx = 584
    body += rect(nx - 12, y0 - 6, 760 - nx - 8, y1 - y0 + 12, fill=PANEL, stroke=RULE, rx=9)
    body += text(nx, y0 + 14, "How to read it", size=12.5, fill=INK, weight=650)
    body += lines(nx, y0 + 36, [
        "This flare alone: 1 minus the",
        "model's chance of that many",
        "clear misses in a row if the",
        "flare had kept burning.",
    ], size=11.5)
    body += lines(nx, y0 + 104, [
        "Across the fleet: the same",
        f"chance times the {fam:,}",
        "source-nights screened in a",
        "30-night window, where a long",
        "run of misses can happen",
        "somewhere by chance.",
    ], size=11.5)
    body += lines(nx, y0 + 210, [
        f"Bands: the {n_src} bright, steady",
        f"sources on {md(d['as_of'])} ({lo:.1%}–{hi:.0%}",
        "first-miss rate). Lines:",
        f"{name(c)} ({p1:.1%}).",
    ], size=11.5)
    body += text(20, y1 + 64, f"Misses cluster: after a clear miss, the next clear night was also a miss {cal['transitions']['MM']} of {cal['transitions']['MM'] + cal['transitions']['MH']} times ({cal['miss_persistence_point']:.0%}); the model uses {q:.0%}.", size=11.5, fill=INK3)
    body += text(20, y1 + 82, "Not measured: whether gas still flows, or a flame below about 0.2–0.3 MW (Figure 23).", size=11.5, fill=INK3)
    svg("fo_confidence", y1 + 100, "Confidence that a dark spell is more than missed detections, by clear dark nights", body)


# ------------------------------------------------------------ figure 23
def fig_bound(d):
    """Detection rate by heat class and the conditional heat bound."""
    curve = d["calibration"]["detection_curve"]
    labels = ["<0.2", "0.2–0.3", "0.3–0.5", "0.5–1", "1–2", "2–5", "5–10", ">10"]
    cx0, cx1 = 200, 740
    colw = (cx1 - cx0) / len(curve)
    body = text(20, 24, "Seen again on another clear pass the same night", size=13, fill=INK, weight=650)
    body += text(20, 42, "Flares detected on one clear pass, by that pass's heat", size=11.5, fill=INK2)
    yt, yb = 62, 182
    ys = Lin(0, 1, yb, yt)
    for v in (0, 0.5, 1):
        body += line(cx0, ys(v), cx1, ys(v), stroke=RULE)
        body += text(cx0 - 8, ys(v) + 4, f"{v:.0%}", size=11, fill=INK3, anchor="end")
    for i, r in enumerate(curve):
        x = cx0 + i * colw
        body += rect(x + 10, ys(r["probability"]), colw - 20, yb - ys(r["probability"]), fill=AMBER, rx=2)
        body += text(x + colw / 2, ys(r["probability"]) - 6, f"{r['probability']:.0%}", size=11.5, fill=INK, anchor="middle")
        body += text(x + colw / 2, yb + 16, labels[i], size=11, fill=INK3, anchor="middle")
        body += text(x + colw / 2, yb + 30, f"n={r['trials']:,}", size=10, fill=INK3, anchor="middle")
    body += text(cx0 - 8, yb + 16, "MW", size=11, fill=INK3, anchor="end")
    body += text(20, 248, "Chance a flare of each size is missed on every clear dark night", size=13, fill=INK, weight=650)
    body += text(20, 266, "Cells at 5% or less, with every larger size also at 5% or less, are ruled out", size=11.5, fill=INK2)
    rows = [(card(d, SHORT), 290), (card(d, EXAMPLE), 344)]
    for c, y in rows:
        k = c["dark_qualified_nights"]
        tails = [(1 - r["probability"]) ** k for r in curve]
        out = [all(t <= 0.05 for t in tails[i:]) for i in range(len(tails))]
        body += text(20, y + 20, f"{k} clear dark nights", size=12, fill=INK, weight=600)
        body += text(20, y + 36, f"({name(c)})", size=11, fill=INK3)
        for i, t in enumerate(tails):
            x = cx0 + i * colw
            fill, ink = (RAISE, INK3) if out[i] else (FL[1], GOLD)
            body += rect(x + 3, y, colw - 6, 44, fill=fill, rx=5)
            s = f"{t:.0%}" if t >= 0.01 else f"{t:.1%}" if t >= 0.001 else "<0.1%"
            body += text(x + colw / 2, y + 27, s, size=12, fill=ink, anchor="middle", weight=600)
        first = out.index(True)
        bx = cx0 + first * colw
        body += line(bx, y - 4, bx, y + 48, stroke=CYAN, w=2)
        body += halo(bx + 6, y - 6 if y == rows[0][1] else y + 60, f"bound: under about {curve[first]['lower_mw']} MW", size=11.5, fill=CYAN)
    ky = 430
    body += rect(20, ky - 9, 10, 10, fill=FL[1], rx=2) + text(36, ky, "could hide on every clear night", size=11.5, fill=INK2)
    body += rect(270, ky - 9, 10, 10, fill=RAISE, rx=2) + text(286, ky, "ruled out", size=11.5, fill=INK2)
    body += text(20, ky + 22, "Assumes one clear look per night and independent nights; a flame smaller than any detected is untested.", size=11.5, fill=INK3)
    svg("fo_bound", ky + 40, "Conditional heat bound from same-night re-detection rates", body)


# ------------------------------------------------------------ figure 24
def fig_emit_timing():
    """How EMIT scenes are labelled against the VIIRS passes that bracket a spell."""
    x0, x1 = 70, 740
    days = 9
    hx = Lin(0, days * 24, x0, x1)
    vy, ty, ey = 90, 138, 186
    body = text(20, 24, "Each EMIT scene is labelled by the VIIRS passes around it", size=13, fill=INK, weight=650)
    body += text(20, 42, "Illustration: a source seen on two nights, dark on four, then seen again", size=11.5, fill=INK2)
    for dd in range(days + 1):  # nights 18:00-06:00 local
        a, b = max(0, dd * 24 - 6), min(days * 24, dd * 24 + 6)
        if b > a:
            body += rect(hx(a), vy - 16, hx(b) - hx(a), ey - vy + 32, fill=RAISE, opacity=0.55)
    body += line(x0, ty, x1, ty, stroke=COAST)
    body += text(x0 - 8, vy + 4, "VIIRS", size=11.5, fill=INK2, anchor="end")
    body += text(x0 - 8, ey + 4, "EMIT", size=11.5, fill=INK2, anchor="end")
    lit = {0, 1, 6, 7}
    passes = {}
    for n in range(8):  # pass at 01:30 local after evening n
        t = (n + 1) * 24 + 1.5
        passes[n] = t
        if n in lit:
            body += circle(hx(t), vy, 5, fill=AMBER)
        else:
            body += circle(hx(t), vy, 5, fill=CLEAR)
    last_lit, first_dark, last_dark, recovery = passes[1], passes[2], passes[5], passes[6]
    edges = [(0, last_lit, "before"), (last_lit, first_dark, "boundary"), (first_dark, last_dark, "during"), (last_dark, recovery, "boundary"), (recovery, days * 24, "after")]
    colors = {"before": INK3, "boundary": INK2, "during": FL[4], "after": INK3}
    for a, b, lab in edges:
        body += path(f"M{hx(a) + 2:.1f},{ty - 5:.1f} L{hx(a) + 2:.1f},{ty:.1f} L{hx(b) - 2:.1f},{ty:.1f} L{hx(b) - 2:.1f},{ty - 5:.1f}", stroke=colors[lab], w=2)
        body += halo((hx(a) + hx(b)) / 2, ty + 18, lab, size=12, fill=colors[lab], anchor="middle", weight=600)
    for t, lab in [(last_lit, "last pass with flare"), (first_dark, "first clear miss"), (last_dark, "last clear miss"), (recovery, "seen again")]:
        body += line(hx(t), vy + 8, hx(t), ty - 6, stroke=INK3, dash="2 3")
    body += halo(hx(last_lit) - 4, vy - 24, "last seen", size=11, fill=INK2, anchor="middle")
    body += halo(hx(recovery), vy - 24, "seen again", size=11, fill=INK2, anchor="middle")
    for t, lab in [(1 * 24 + 14, "before"), (2 * 24 + 10.5, "boundary"), (4 * 24 + 15, "during"), (6 * 24 + 11, "boundary"), (8 * 24 + 9.5, "after")]:
        x = hx(t)
        body += path(f"M{x:.1f},{ey - 7:.1f} L{x + 7:.1f},{ey:.1f} L{x:.1f},{ey + 7:.1f} L{x - 7:.1f},{ey:.1f} Z", stroke=CYAN, w=1.4, fill=BG)
    for dd in range(days):
        body += text(hx(dd * 24 + 12), ey + 34, f"day {dd + 1}", size=11, fill=INK3, anchor="middle")
    ky = ey + 64
    keys = [
        (circle(0, 0, 4.5, fill=AMBER), "VIIRS pass, flare seen"),
        (circle(0, 0, 4.5, fill=CLEAR), "VIIRS pass, clear, no flare"),
        (path("M0,-6 L6,0 L0,6 L-6,0 Z", stroke=CYAN, w=1.4, fill=BG), "EMIT daytime scene"),
        (rect(-6, -6, 12, 12, fill=RAISE), "night"),
    ]
    for j, (mark, lab) in enumerate(keys):
        x = 20 + j * 190
        body += f'<g transform="translate({x + 6},{ky - 4})">{mark}</g>' + text(x + 20, ky, lab, size=11.5, fill=INK2)
    body += text(20, ky + 26, "The listing covers 14 nights before the spell to 14 nights after it (or the latest night).", size=11.5, fill=INK3)
    svg("fo_emit", ky + 44, "How EMIT scenes are labelled before, during, boundary or after a dark spell", body)


# ------------------------------------------------------------ figure 25
def fig_emit_chips(d):
    """Two real EMIT scenes in the Flare-outs tab's review recipe."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    import emit

    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "text.color": INK,
        "axes.edgecolor": RULE,
        "savefig.facecolor": BG,
        "figure.facecolor": BG,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.1), dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.05, hspace=0.2)
    for row, (cid, scene_id, t, cloud) in zip(axes, EMIT_EXAMPLES):
        c = card(d, cid)
        r = emit.fetch_chips(dict(scene_id=scene_id, time_utc=t), c["unit"])
        if r["status"] != "available":
            raise RuntimeError(f"EMIT chips unavailable for {scene_id}: {r.get('message')}")
        folder = Path(r["cache_folder"])
        when = pd.Timestamp(t)
        sub = f"Site {name(c)} · {c['unit']['region']} · {when:%-d %b %Y %H:%M} UTC · scene cloud {cloud}%"
        for ax, png, title in zip(row, ("rgb.png", "ch4.png"), ("True colour", "CH4 absorption index")):
            im = Image.open(io.BytesIO((folder / png).read_bytes())).convert("RGB")
            ax.imshow(np.asarray(im)[24:])  # drop the stamp's title bar; keep marker and scale
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(RULE)
            ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK, pad=4)
        row[0].annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points", fontsize=8, color=INK2, va="bottom")
    cmap = LinearSegmentedColormap.from_list("ch4", ["#00131f", "#7e3d01", "#fd8e28", "#fff4d0"])
    sig = np.linspace(1, 6, 256)
    disp = ((sig - 1) / 5) ** 0.65
    cax = fig.add_axes([0.53, 0.07, 0.44, 0.022])
    cax.imshow(cmap(disp)[None, :, :3], aspect="auto", extent=[1, 6, 0, 1])
    cax.set_yticks([])
    cax.set_xticks([1, 2, 3, 4, 5, 6])
    cax.set_xticklabels(["≤1σ", "2σ", "3σ", "4σ", "5σ", "≥6σ"])
    cax.tick_params(colors=INK2, labelsize=7.5, length=2)
    for s in cax.spines.values():
        s.set_edgecolor(RULE)
    cax.set_xlabel("index above the scene's median within 15 km, in robust standard deviations", color=INK2, fontsize=7.5)
    fig.text(0.02, 0.085, "Ring: the source. Bar: 5 km. North is up.", color=INK2, fontsize=7.8)
    fig.text(0.02, 0.06, "Both scenes predate the VIIRS record (9 Aug 2026).", color=INK2, fontsize=7.8)
    fig.savefig(HERE / "chips_emit.png")
    plt.close(fig)
    print("wrote chips_emit.png")


def main(chips: bool = True):
    d = load()
    fig_record(d)
    fig_tiers(d)
    fig_confidence(d, first_miss_range(d["as_of"]))
    fig_bound(d)
    fig_emit_timing()
    if chips:
        fig_emit_chips(d)


# %%
if __name__ == "__main__":
    main()
