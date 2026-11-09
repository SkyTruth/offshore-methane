# %%
"""Write the SVG figures for docs/explainer.md.

Charts are computed from every v5 run segment (``outputs/v5_*``, joined by
``build_viewer.load_segments``): 88 nights, 9 August to 4 November 2026. The
v3-to-v5 waterfall uses the 30 nights run with both versions (6 October to
4 November). Diagrams are drawn to scale where a scale is stated. Every figure uses the viewer's SkyTruth palette on its own dark panel,
so it reads on light and dark pages alike. Run from ``nightly-flaring/``:
``.venv/bin/python docs/explainer/make_figures.py``.
"""

from __future__ import annotations

import itertools
import math
import sys
import tempfile
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import build_viewer as BV  # noqa: E402
import gates_30n as G  # noqa: E402
import nightfire as nf  # noqa: E402
import pipeline as p  # noqa: E402
import validate_myc25 as V  # noqa: E402

SEGMENTS = BV.find_segments()
TAG88 = f"{SEGMENTS[0][1][:10]}_{SEGMENTS[-1][1][11:]}"
_T0, _T1 = (pd.Timestamp(d) for d in TAG88.split("_"))
N_NIGHTS = (_T1 - _T0).days + 1
PERIOD = f"{_T0:%-d %b}–{_T1:%-d %b %Y}"

# Night palette (viewer_template.html tokens).
BG = "#00131F"
PANEL = "#001929"
RAISE = "#002640"
RULE = "#043352"
COAST = "#1D5072"
INK = "#F9F6F3"
INK2 = "#9CB7C9"
INK3 = "#60869F"
FL = ["#4C2501", "#7E3D01", "#CA6202", "#FD8E28", "#FEBD81", "#F7E5D4"]
AMBER = FL[3]
GOLD = FL[4]
CLEAR = "#60869F"
CLOUD = "#1D5072"
CYAN = "#9FC6BA"
SANS = "'PP Neue Montreal', Roboto, Arial, sans-serif"
MONO = "Roboto, Arial, sans-serif"
W = 760


# ----------------------------------------------------------------- primitives
def _a(**kw) -> str:
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        out.append(f'{k.rstrip("_").replace("_", "-")}="{v}"')
    return " ".join(out)


def text(x, y, s, size=13, fill=INK, anchor="start", weight=400, mono=False, **kw):
    fam = MONO if mono else SANS
    return (
        f'<text {_a(x=f"{x:.1f}", y=f"{y:.1f}", fill=fill, font_size=size, font_family=fam, text_anchor=anchor, font_weight=weight, **kw)}>'
        f"{escape(str(s))}</text>"
    )


def halo(x, y, s, **kw):
    """Text with a background-coloured outline, for labels over data."""
    return text(x, y, s, stroke=BG, stroke_width=4, paint_order="stroke", **kw)


def lines(x, y, rows, size=13, fill=INK2, lh=1.35, **kw):
    return "".join(
        text(x, y + i * size * lh, r, size=size, fill=fill, **kw)
        for i, r in enumerate(rows)
    )


def line(x1, y1, x2, y2, stroke=RULE, w=1, dash=None, **kw):
    return f'<line {_a(x1=f"{x1:.1f}", y1=f"{y1:.1f}", x2=f"{x2:.1f}", y2=f"{y2:.1f}", stroke=stroke, stroke_width=w, stroke_dasharray=dash, **kw)}/>'


def rect(x, y, w, h, fill="none", stroke=None, sw=1, rx=0, **kw):
    return f'<rect {_a(x=f"{x:.1f}", y=f"{y:.1f}", width=f"{w:.1f}", height=f"{h:.1f}", fill=fill, stroke=stroke, stroke_width=sw if stroke else None, rx=rx, **kw)}/>'


def circle(cx, cy, r, fill=AMBER, stroke=None, sw=1, tip=None, **kw):
    el = f'<circle {_a(cx=f"{cx:.1f}", cy=f"{cy:.1f}", r=r, fill=fill, stroke=stroke, stroke_width=sw if stroke else None, **kw)}'
    return el + (f"><title>{escape(tip)}</title></circle>" if tip else "/>")


def path(d, stroke=AMBER, w=2, fill="none", **kw):
    return f'<path {_a(d=d, stroke=stroke, stroke_width=w, fill=fill, stroke_linejoin="round", stroke_linecap="round", **kw)}/>'


def poly(pts, **kw):
    return path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts), **kw)


def arrow(x1, y1, x2, y2, color=INK3, w=1.5, head=7):
    ang = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - head * math.cos(ang - 0.45)
    hy1 = y2 - head * math.sin(ang - 0.45)
    hx2 = x2 - head * math.cos(ang + 0.45)
    hy2 = y2 - head * math.sin(ang + 0.45)
    return line(x1, y1, x2, y2, stroke=color, w=w) + path(
        f"M{hx1:.1f},{hy1:.1f} L{x2:.1f},{y2:.1f} L{hx2:.1f},{hy2:.1f}",
        stroke=color,
        w=w,
    )


def svg(name: str, h: int, label: str, body: str) -> None:
    doc = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" '
        f'role="img" aria-label="{escape(label)}"><title>{escape(label)}</title>'
        f'<rect width="{W}" height="{h}" rx="12" fill="{BG}"/>{body}</svg>\n'
    )
    (HERE / f"{name}.svg").write_text(doc)
    print(f"wrote {name}.svg")


class Lin:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


class Log(Lin):
    def __call__(self, v):
        v = max(v, 1e-300)
        return self.r0 + (math.log10(v) - math.log10(self.d0)) / (
            math.log10(self.d1) - math.log10(self.d0)
        ) * (self.r1 - self.r0)


def frame(x0, y0, x1, y1):
    """Recessive plot frame: baseline and left rule."""
    return line(x0, y1, x1, y1, stroke=COAST) + line(x0, y0, x0, y1, stroke=RULE)


def xticks(sc, vals, y, fmt=str, grid_top=None, size=11):
    out = ""
    for v in vals:
        x = sc(v)
        if grid_top is not None:
            out += line(x, grid_top, x, y, stroke=RULE, w=1)
        out += line(x, y, x, y + 4, stroke=COAST)
        out += text(x, y + 17, fmt(v), size=size, fill=INK3, anchor="middle", mono=True)
    return out


def yticks(sc, vals, x, fmt=str, grid_right=None, size=11):
    out = ""
    for v in vals:
        y = sc(v)
        if grid_right is not None:
            out += line(x, y, grid_right, y, stroke=RULE, w=1)
        out += text(x - 8, y + 4, fmt(v), size=size, fill=INK3, anchor="end", mono=True)
    return out


def nice_step(vmax: float, max_ticks: int = 6) -> int:
    """Smallest 1-2-5 step giving at most ``max_ticks`` ticks up to ``vmax``."""
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if vmax / step <= max_ticks:
            return step
    return 10000


def sup(v):
    """10^k label for log axes."""
    k = int(round(math.log10(v)))
    return {0: "1", 1: "10", 2: "100", 3: "1,000", -1: "0.1", -2: "0.01"}.get(k, f"1e{k}")


# --------------------------------------------------------------------- data
def load():
    """Every v5 segment joined: sites, records (as gates_30n.load), nights,
    MYC25 matches and the bow-tie copy table."""
    sites = pd.read_csv(p.SITES_CSV)
    nights, over = BV.load_segments(SEGMENTS)
    dd = p.dedupe_overpasses(over)
    dd = dd[dd.sza >= p.SZA_CLEAR].copy()
    lon = dd.site_id.map(sites.set_index("site_id").lon)
    local = pd.to_datetime(dd.time_utc) + pd.to_timedelta(lon / 15, "h")
    dd["night"] = (local - pd.Timedelta(hours=12)).dt.date.astype(str)
    dd["clear"] = p.strict_clear(dd)
    start, end = TAG88.split("_")
    dd = dd[(dd.night >= start) & (dd.night <= end)].copy()
    run = dict(over=over, dd=dd, det=dd[dd.detected].copy())
    _lh(run)
    run["det"] = run["dd"][run["dd"].detected].copy()
    val = V.match_sites(nights, V.load_myc(2025), 2000.0, over)
    copies = (
        pd.concat(
            [pd.read_csv(f / f"copies_{t}.csv") for f, t in SEGMENTS], ignore_index=True
        )
        .drop_duplicates(["satellite", "granule", "group_id", "scan", "peak_row", "peak_col"])
        .reset_index(drop=True)
    )
    return sites, run, nights, val, copies


# ------------------------------------------------------------------ figures
def fig_pipeline():
    steps = [
        ("Night pass", ["VIIRS scans the sea around", "01:45 local time in 750 m", "pixels, three satellites."]),
        ("Find hot pixels", ["A pixel glows at 1.6 µm far", "above the dark sea around it,", "and a second band agrees."]),
        ("Read the colours", ["Brightness in up to six", "infrared bands, 0.9–4 µm,", "traces the flame's spectrum."]),
        ("Fit a hot body", ["The spectrum's shape gives", "temperature (about 1,750 K);", "its height gives flame area."]),
        ("Radiant heat", ["Area × σT⁴ = megawatts", "of heat leaving the flame", "during that pass."]),
        ("Average, convert", ["Mean over clear passes, with", "unlit passes as zero, × 0.0116", "gives billion m³ of gas a year."]),
    ]
    body = ""
    bw, bh, gx, gy = 220, 118, 30, 34
    x0, y0 = 20, 22
    for i, (title, rows) in enumerate(steps):
        r, c = divmod(i, 3)
        x, y = x0 + c * (bw + gx), y0 + r * (bh + gy)
        body += rect(x, y, bw, bh, fill=PANEL, stroke=RULE, rx=10)
        body += circle(x + 22, y + 26, 13, fill=FL[1 + i % 5] if i < 5 else GOLD)
        body += text(x + 22, y + 31, i + 1, size=13, fill=BG, anchor="middle", weight=700, mono=True)
        body += text(x + 44, y + 31, title, size=15, fill=INK, weight=650)
        body += lines(x + 16, y + 60, rows, size=12.5, fill=INK2)
        if c < 2:
            body += arrow(x + bw + 4, y + bh / 2, x + bw + gx - 4, y + bh / 2)
    # wrap arrow from step 3 to step 4
    xa, ya = x0 + 2 * (bw + gx) + bw / 2, y0 + bh
    xb, yb = x0 + bw / 2, y0 + bh + gy
    body += path(
        f"M{xa:.1f},{ya + 2:.1f} L{xa:.1f},{ya + gy / 2:.1f} L{xb:.1f},{ya + gy / 2:.1f} L{xb:.1f},{yb - 6:.1f}",
        stroke=INK3,
        w=1.5,
    ) + arrow(xb, yb - 12, xb, yb - 2)
    svg("pipeline", 2 * bh + gy + 44, "Six steps from a satellite pass to an annual gas volume", body)


def fig_swath(run):
    dd = run["dd"]
    body = ""
    # Left: across-track pixel growth, drawn to scale (1 px = 20 m vertical scale bar).
    body += text(24, 34, "Pixel size along one scan line (schematic)", size=13, fill=INK2, weight=600)
    # Pixel along-scan sizes in km: nadir 0.75, mid 1.1, edge 1.6 (VIIRS M-band design).
    xs = []
    widths = []
    for k in range(-12, 13):
        a = abs(k) / 12
        wkm = 0.75 + 0.85 * a**2.2
        widths.append(wkm)
    total = sum(widths)
    scale = 700 / total
    x = 30
    for k, wkm in zip(range(-12, 13), widths):
        w = wkm * scale
        hgt = wkm * 30
        body += rect(x, 92 - hgt / 2, w - 1.2, hgt, fill=RAISE, stroke=COAST, sw=0.8)
        xs.append(x + w / 2)
        x += w
    body += text(xs[12], 136, "nadir: 0.75 × 0.75 km", size=12, fill=INK, anchor="middle", mono=True)
    body += text(xs[0] - 10, 136, "edge: 1.6 × 1.6 km", size=12, fill=INK, anchor="start", mono=True)
    body += arrow(xs[12], 124, xs[12], 106, color=INK3)
    body += arrow(xs[0], 124, xs[0], 118, color=INK3)
    body += text(30 + 350, 160, "about 3,000 km of sea in one sweep; pixels grow toward both edges", size=12, fill=INK3, anchor="middle")
    # Right/below: local pass times per satellite.
    y0 = 200
    body += text(24, y0, f"When each satellite passes a site (local solar time, {N_NIGHTS} nights)", size=13, fill=INK2, weight=600)
    sc = Lin(24.0, 28.0, 120, 720)
    body += line(120, y0 + 108, 720, y0 + 108, stroke=COAST)
    for h in (24.0, 25.0, 26.0, 27.0, 28.0):
        body += line(sc(h), y0 + 22, sc(h), y0 + 108, stroke=RULE)
        hh = int(h) % 24
        body += text(sc(h), y0 + 126, f"{hh:02d}:00", size=11, fill=INK3, anchor="middle", mono=True)
    for i, sat in enumerate(["SNPP", "NOAA-20", "NOAA-21"]):
        h = dd.loc[dd.satellite == sat, "lh"]
        q10, q50, q90 = np.percentile(h, [10, 50, 90])
        y = y0 + 42 + i * 26
        body += text(110, y + 4, sat, size=12, fill=INK2, anchor="end", mono=True)
        body += line(sc(q10), y, sc(q90), y, stroke=FL[2], w=6, stroke_linecap="round")
        body += circle(sc(q50), y, 6, fill=GOLD, tip=f"{sat}: median {int(q50) % 24:02d}:{int((q50 % 1) * 60):02d}")
    g = dd.groupby(["site_id", "night"])
    clear = g.clear.sum().mean()
    share = (g.clear.sum() > 0).mean()
    body += text(
        120,
        y0 + 152,
        f"Bar: middle 80% of passes; dot: median. A site averages {clear:.1f} clear looks a night; "
        f"{share:.0%} of site-nights have at least one.",
        size=12,
        fill=INK3,
    )
    svg("swath", y0 + 170, "VIIRS pixel sizes across the swath and satellite pass times", body)


def _lh(run):
    dd = run["dd"]
    sites = pd.read_csv(p.SITES_CSV).set_index("site_id")
    lon = dd.site_id.map(sites.lon)
    lt = pd.to_datetime(dd.time_utc) + pd.to_timedelta(lon / 15, "h")
    h = lt.dt.hour + lt.dt.minute / 60
    dd["lh"] = np.where(h < 12, h + 24, h)


def fig_subpixel():
    body = ""
    # Pixel 750 m drawn as 300 px: 1 px = 2.5 m.
    px0, py0, S = 40, 40, 300
    body += rect(px0, py0, S, S, fill=RAISE, stroke=COAST, sw=1.2)
    body += text(px0, py0 - 12, "One nadir pixel, 750 m × 750 m", size=13, fill=INK2, weight=600)
    # Platform ~ 60 m x 40 m at centre -> 24 x 16 px; flame 2.1 m² -> 1.45 m side -> 0.58 px.
    cx, cy = px0 + 0.62 * S, py0 + 0.45 * S
    body += rect(cx - 12, cy - 8, 24, 16, fill=COAST)
    body += circle(cx + 13, cy - 9, 1.2, fill=GOLD)
    body += circle(cx + 13, cy - 9, 9, fill="none", stroke=GOLD, sw=1)
    body += text(px0 + S / 2, py0 + S + 22, "drawn to scale: platform 60 × 40 m, flame 2 m²", size=12, fill=INK3, anchor="middle")
    # Zoom box
    zx, zy, Z = 430, 60, 260
    body += path(f"M{cx + 22:.1f},{cy - 9:.1f} L{zx:.1f},{zy + Z / 2:.1f}", stroke=INK3, w=1, stroke_dasharray="3 4")
    body += rect(zx, zy, Z, Z, fill=PANEL, stroke=RULE, rx=8)
    body += text(zx, zy - 12, "Zoomed 40×: 26 m across", size=13, fill=INK2, weight=600)
    # 26 m across -> 10 px per m. Platform deck edge and a flame of 2.1 m² (1.45 m).
    body += rect(zx + 20, zy + 150, 200, 90, fill=COAST)
    body += line(zx + 150, zy + 150, zx + 175, zy + 92, stroke=INK3, w=3)
    fl = 14.5
    body += f'<ellipse cx="{zx + 178:.1f}" cy="{zy + 80:.1f}" rx="{fl / 2:.1f}" ry="{fl:.1f}" fill="{GOLD}"/>'
    body += f'<ellipse cx="{zx + 178:.1f}" cy="{zy + 84:.1f}" rx="{fl / 4:.1f}" ry="{fl / 2:.1f}" fill="{FL[5]}"/>'
    body += lines(zx + 18, zy + 30, ["Flame on the flare boom:", "about 2 m² of 1,750 K gas"], size=12.5, fill=INK)
    body += text(zx + 18, zy + Z - 12, "deck", size=11, fill=INK3)
    body += lines(
        40,
        py0 + S + 58,
        [
            "The flame covers about 1/270,000 of the pixel. VIIRS cannot see its shape; it records how much",
            "extra light the whole pixel gives off, and the method works back from that to temperature and size.",
        ],
        size=13,
        fill=INK2,
    )
    svg("subpixel", py0 + S + 100, "A flare fills a tiny fraction of one VIIRS pixel", body)


def fig_spectrum():
    x0, x1, y0, y1 = 84, 730, 30, 300
    xs = Log(0.6, 14, x0, x1)
    ys = Log(1e-6, 1e2, y1, y0)
    esf = 2.1 / 562_500
    body = frame(x0, y0, x1, y1)
    for v in (1e-6, 1e-4, 1e-2, 1, 1e2):
        body += line(x0, ys(v), x1, ys(v), stroke=RULE)
    body += yticks(ys, (1e-6, 1e-4, 1e-2, 1, 1e2), x0, fmt=lambda v: {1e-6: "10⁻⁶", 1e-4: "10⁻⁴", 1e-2: "10⁻²", 1: "1", 1e2: "100"}[v])
    body += xticks(xs, (0.7, 1, 2, 4, 7, 10), y1, fmt=lambda v: f"{v:g}")
    body += text((x0 + x1) / 2, y1 + 38, "wavelength (micrometres)", size=12, fill=INK3, anchor="middle")
    body += text(18, (y0 + y1) / 2, "pixel radiance", size=12, fill=INK3, anchor="middle", transform=f"rotate(-90 18 {(y0 + y1) / 2})")
    for name, wl in nf.BAND_UM.items():
        x = xs(wl)
        body += rect(x - 4, y0, 8, y1 - y0, fill=CYAN, opacity=0.08)
        anchor = {"M12": "end", "M13": "start"}.get(name, "middle")
        dx = {"M12": -2, "M13": 2}.get(name, 0)
        body += text(x + dx, y0 - 6, name, size=11, fill=CYAN, anchor=anchor, mono=True)
    wl = np.geomspace(0.6, 14, 300)
    fl = [esf * nf.planck(w, 1750) for w in wl]
    sea = [0.98 * nf.planck(w, 290) for w in wl]
    fpts = [(xs(w), ys(v)) for w, v in zip(wl, fl) if v > 1e-6]
    spts = [(xs(w), ys(v)) for w, v in zip(wl, sea) if v > 1e-6]
    body += poly(spts, stroke=CYAN, w=2.2)
    body += poly(fpts, stroke=AMBER, w=2.4)
    body += text(xs(1.05), ys(0.6), "flare, 1,750 K, 2 m² in the pixel", size=12.5, fill=INK, weight=600)
    body += text(xs(5.6), ys(30), "sea surface, 290 K", size=12.5, fill=INK, weight=600)
    body += lines(xs(0.65), ys(4e-5), ["Short-wave infrared: the flare", "outshines the night sea", "about 500,000 to 1"], size=12, fill=INK2)
    body += lines(xs(5.0), ys(1.2e-3), ["Thermal infrared: the sea", "outshines the flare", "about 2,400 to 1"], size=12, fill=INK2)
    svg("spectrum", y1 + 52, "Light from a flare and from the night sea at VIIRS wavelengths", body)


def fig_detection():
    rng = np.random.default_rng(7)
    n = 48
    v = rng.normal(0.012, 0.004, n).clip(0.001)
    v[23], v[22], v[24] = 0.36, 0.05, 0.03
    mean, sd = 0.012, 0.004
    thr = mean + 4 * sd
    x0, x1, y0, y1 = 70, 730, 34, 250
    xs = Lin(-0.5, n - 0.5, x0, x1)
    ys = Log(0.001, 1, y1, y0)
    body = frame(x0, y0, x1, y1)
    body += yticks(ys, (0.001, 0.01, 0.1, 1), x0, fmt=lambda t: f"{t:g}", grid_right=x1)
    bw = (x1 - x0) / n - 3
    for i, val in enumerate(v):
        hot = val > thr
        x = xs(i) - bw / 2
        body += rect(x, ys(val), bw, y1 - ys(val), fill=AMBER if hot else CLEAR, rx=2)
    body += line(x0, ys(thr), x1, ys(thr), stroke=GOLD, w=1.5, dash="5 4")
    body += text(x0 + 6, ys(thr) - 8, "threshold: background mean + 4 standard deviations", size=12, fill=GOLD)
    body += text(xs(23) + 12, ys(0.36) + 4, "flare pixel", size=12.5, fill=INK, weight=600)
    body += text(xs(25) + 8, ys(0.08), "glow in the neighbours", size=11.5, fill=INK2)
    body += text((x0 + x1) / 2, y1 + 22, "pixels along one scan line (illustration)", size=12, fill=INK3, anchor="middle")
    body += text(16, (y0 + y1) / 2, "1.6 µm radiance", size=12, fill=INK3, anchor="middle", transform=f"rotate(-90 16 {(y0 + y1) / 2})")
    body += lines(x0, y1 + 50, [
        "A pixel becomes a candidate when it clears the line; it is confirmed when another band also shows heat,",
        "which rules out particle hits on the detector. Only the brightest pixel of a hot patch counts as a flare peak.",
    ], size=12.5, fill=INK2)
    svg("detection", y1 + 86, "Finding a flare pixel against the night background", body)


SPEC_3802 = dict(
    bands=["M10", "M7", "M8", "M11", "M12", "M13"],
    obs=[9.605476723518223, 0.12164172530174255, 4.8704103087075055, 7.821446444024332, 2.8952914235266864, 1.5903347364643163],
    noise=[0.008896060746724067, 0.007632865683929609, 0.014036094260076327, 0.005066581405699472, 0.002404313521851856, 0.003405299007027461],
)


def _spectrum_axes(ymax=30):
    x0, x1, y0, y1 = 84, 560, 30, 290
    xs = Lin(0.7, 4.3, x0, x1)
    ys = Log(0.05, ymax, y1, y0)
    body = frame(x0, y0, x1, y1)
    ticks = [t for t in (0.1, 1, 10, 100) if t <= ymax]
    body += yticks(ys, ticks, x0, fmt=lambda v: f"{v:g}", grid_right=x1)
    body += xticks(xs, (1, 2, 3, 4), y1, fmt=lambda v: f"{v:g}")
    body += text((x0 + x1) / 2, y1 + 38, "wavelength (micrometres)", size=12, fill=INK3, anchor="middle")
    body += text(18, (y0 + y1) / 2, "radiance (W m⁻² sr⁻¹ µm⁻¹)", size=12, fill=INK3, anchor="middle", transform=f"rotate(-90 18 {(y0 + y1) / 2})")
    return body, xs, ys, (x0, x1, y0, y1)


def _curve(xs, ys, esf, temp, color, w=2.4, dash=None):
    wl = np.linspace(0.75, 4.25, 200)
    pts = [(xs(v), ys(esf * nf.planck(v, temp))) for v in wl if esf * nf.planck(v, temp) > 0.05]
    return poly(pts, stroke=color, w=w, stroke_dasharray=dash)


def _points(xs, ys, dropped=()):
    out = ""
    for b, o in zip(SPEC_3802["bands"], SPEC_3802["obs"]):
        x, y = xs(nf.BAND_UM[b]), ys(o)
        if b in dropped:
            out += circle(x, y, 6, fill=BG, stroke=INK2, sw=2, tip=f"{b}: {o:.3g} (dropped)")
        else:
            out += circle(x, y, 6, fill=INK, stroke=BG, sw=2, tip=f"{b}: {o:.3g}")
        out += text(x + 9, y - 9, b, size=11, fill=INK2, mono=True)
    return out


def fig_fit():
    t, esf, chi, used, flag = nf.fit_planck(SPEC_3802["bands"], SPEC_3802["obs"], SPEC_3802["noise"])
    body, xs, ys, (x0, x1, y0, y1) = _spectrum_axes()
    body += _curve(xs, ys, esf, t, AMBER)
    body += _points(xs, ys, dropped=("M7",))
    tx = x1 + 22
    body += text(tx, 54, "Measured (dots)", size=13, fill=INK, weight=600)
    body += lines(tx, 74, ["Six bands, NOAA-20,", "10 Oct 2026, site 3802"], size=12, fill=INK2)
    body += text(tx, 128, "Best-fit hot body", size=13, fill=AMBER, weight=600)
    body += lines(tx, 148, [f"temperature {t:,.0f} K", f"fills {esf * 100:.3f}% of pixel"], size=12, fill=INK2, mono=True)
    body += text(tx, 204, "Hollow dot", size=13, fill=INK, weight=600)
    body += lines(tx, 224, ["M7 sits far off the", "curve and is set aside;", "the other five agree."], size=12, fill=INK2)
    svg("fit", y1 + 52, "Fitting a flame temperature and size to six infrared bands", body)


def fig_heat(run):
    det = run["det"]
    t = np.percentile(det.temp_k.dropna(), [5, 50, 95])
    a = np.percentile(det.area_m2.dropna(), [5, 50, 95])
    r = np.percentile(det.rh_mw, [5, 50, 95])
    body = ""
    body += text(W / 2, 46, "radiant heat  =  flame area  ×  σ  ×  temperature⁴", size=20, fill=INK, anchor="middle", weight=650)
    body += text(W / 2, 74, "σ = 5.67 × 10⁻⁸ W m⁻² K⁻⁴ (Stefan–Boltzmann constant)", size=12, fill=INK3, anchor="middle", mono=True)
    tiles = [
        ("Temperature", "K", t, lambda v: f"{v:,.0f}"),
        ("Flame area", "m²", a, lambda v: f"{v:,.1f}" if v < 100 else f"{v:,.0f}"),
        ("Radiant heat per pass", "MW", r, lambda v: f"{v:,.2f}" if v < 10 else f"{v:,.0f}"),
    ]
    tw, gap = 222, 17
    x = (W - 3 * tw - 2 * gap) / 2
    for name, unit, q, fmt in tiles:
        body += rect(x, 100, tw, 112, fill=PANEL, stroke=RULE, rx=10)
        body += text(x + 16, 124, name.upper(), size=11, fill=INK3, letter_spacing="0.08em")
        body += text(x + 16, 162, f"{fmt(q[1])} {unit}", size=26, fill=GOLD if unit == "MW" else INK, weight=600, mono=True)
        body += text(x + 16, 192, f"5th–95th pct {fmt(q[0])}–{fmt(q[2])}", size=11.5, fill=INK2, mono=True)
        x += tw + gap
    body += text(W / 2, 244, f"Typical values over {len(det):,} detections, {N_NIGHTS} nights (v5). One megawatt is the output of about 400 electric space heaters.", size=12, fill=INK3, anchor="middle")
    svg("heat", 264, "Radiant heat from temperature and flame area, with typical values", body)


def fig_nights(run, nights):
    dd = run["dd"]
    sid = 928
    s = dd[dd.site_id == sid].copy()
    nl = [d.strftime("%Y-%m-%d") for d in pd.date_range(_T0, _T1, freq="D")]
    idx = {n: i for i, n in enumerate(nl)}
    s["i"] = s.night.map(idx)
    x0, x1, y0, y1 = 70, 730, 36, 240
    xs = Lin(-0.6, len(nl) - 0.4, x0, x1)
    ymax = max(10, float(s.rh_mw.max()) * 1.08)
    ys = Lin(0, ymax, y1, y0)
    body = frame(x0, y0, x1, y1)
    body += yticks(ys, range(0, int(ymax) + 1, nice_step(ymax)), x0, fmt=lambda v: f"{v}", grid_right=x1)
    body += text(16, (y0 + y1) / 2, "MW per pass", size=12, fill=INK3, anchor="middle", transform=f"rotate(-90 16 {(y0 + y1) / 2})")
    # date labels on the 1st and 15th of each month
    for i, n in enumerate(nl):
        d = pd.Timestamp(n)
        if d.day in (1, 15):
            body += line(xs(i), y1, xs(i), y1 + 4, stroke=INK3)
            body += text(xs(i), y1 + 17, d.strftime("%-d %b"), size=11, fill=INK3, anchor="middle", mono=True)
    # cloud strip: one cell per night, darker the more of its passes were cloudy
    cy = y1 + 34
    body += text(x0 - 8, cy + 4, "cloud", size=11, fill=INK3, anchor="end")
    site_n = nights[nights.site_id == sid].set_index("night")
    cw = xs(1) - xs(0)
    jit = 0.28
    for n, f in s.groupby("night"):
        i = idx[n]
        cloudy = int((~f.clear & ~f.detected).sum())
        if cloudy:
            share = cloudy / len(f)
            body += rect(xs(i) - cw / 2 + 0.6, cy - 4, cw - 1.2, 8, fill=CLOUD, opacity=round(0.35 + 0.65 * share, 2),
                         rx=1.5) .replace("/>", f"><title>{n}: {cloudy} of {len(f)} passes cloudy</title></rect>")
        offs = np.linspace(-jit, jit, len(f)) if len(f) > 1 else [0]
        for j, r in zip(offs, f.itertuples()):
            x = xs(i + j)
            tip = f"{n} {r.satellite}: "
            if r.clear and r.detected:
                body += circle(x, ys(r.rh_mw), 2.6, fill=AMBER, tip=tip + f"{r.rh_mw:.2f} MW, clear")
            elif r.clear:
                body += circle(x, ys(0), 2.3, fill=CLEAR, tip=tip + "clear, no flare seen (counts as 0)")
            elif r.detected:
                body += circle(x, ys(r.rh_mw), 2.6, fill=BG, stroke=AMBER, sw=1.2, tip=tip + f"{r.rh_mw:.2f} MW under cloud (not averaged)")
    for n, r in site_n.iterrows():
        if n in idx and pd.notna(r.rh_mw) and r.n_clear > 0:
            i = idx[n]
            body += line(xs(i) - cw * 0.42, ys(r.rh_mw), xs(i) + cw * 0.42, ys(r.rh_mw), stroke=GOLD, w=2.2)
    mean = site_n.rh_mw.mean()
    body += line(x0, ys(mean), x1, ys(mean), stroke=CYAN, w=1.5, dash="6 4")
    label = f"{len(nl)}-night mean {mean:.2f} MW → {mean * 0.0116:.3f} BCM/yr"
    body += text(x1, 22, label, size=12.5, fill=CYAN, anchor="end", weight=600)
    body += line(x1 - 7.4 * len(label) - 34, 18, x1 - 7.4 * len(label) - 8, 18, stroke=CYAN, w=1.5, dash="6 4")
    ly = cy + 34
    keys = [
        (circle(0, 0, 3.4, fill=AMBER), "flare seen, clear sky"),
        (circle(0, 0, 3, fill=CLEAR), "clear, no flare (0 MW)"),
        (circle(0, 0, 3.4, fill=BG, stroke=AMBER, sw=1.4), "seen through cloud (not averaged)"),
        (rect(-4, -4, 8, 8, fill=CLOUD, rx=1.5), "cloudy passes that night"),
        (line(-7, 0, 7, 0, stroke=GOLD, w=2.4), "night's mean"),
    ]
    for k, (mark, lab) in enumerate(keys):
        x, yy = x0 + (k % 3) * 220, ly + (k // 3) * 20
        body += f'<g transform="translate({x + 6},{yy - 4})">{mark}</g>' + text(x + 20, yy, lab, size=11.5, fill=INK2)
    clear_n = int((site_n.n_clear > 0).sum())
    flare_n = int((site_n.status == "flaring").sum())
    body += text(x0, ly + 52, f"{len(s)} passes; {clear_n} of {len(nl)} nights had a clear pass, and the flare was seen on {flare_n} of them.", size=12, fill=INK3)
    svg("nights", ly + 70, f"{len(nl)} nights of passes over one Bay of Campeche site", body)


def fig_volume(nights):
    m = nights.groupby("site_id").rh_mw.mean()
    fl = m[m > 0]
    x0, x1 = 60, 700
    xs = Log(0.01, 100, x0, x1)
    body = ""
    ya, yb = 70, 140
    body += text(x0, 40, "site's mean radiant heat (MW)", size=12.5, fill=INK2, weight=600)
    body += line(x0, ya, x1, ya, stroke=COAST, w=1.5)
    body += line(x0, yb, x1, yb, stroke=COAST, w=1.5)
    for v in (0.01, 0.1, 1, 10, 100):
        x = xs(v)
        body += line(x, ya - 6, x, ya + 6, stroke=INK3)
        body += text(x, ya - 12, f"{v:g}", size=12, fill=INK, anchor="middle", mono=True)
        body += line(x, yb - 6, x, yb + 6, stroke=INK3)
        b = v * 0.0116
        body += text(x, yb + 22, f"{b:.4g}", size=12, fill=INK, anchor="middle", mono=True)
        body += line(x, ya + 8, x, yb - 8, stroke=RULE, dash="2 4")
    body += text(x0, yb + 48, "gas flared (billion m³ per year)  =  0.0116 × MW", size=12.5, fill=INK2, weight=600)
    # distribution of flaring-site means as a strip of ticks
    for v in fl.values:
        x = xs(min(max(v, 0.0101), 99))
        body += line(x, ya + 14, x, ya + 26, stroke=AMBER, w=1, opacity=0.5)
    q = fl.median()
    body += circle(xs(q), ya + 34, 4.5, fill=GOLD)
    body += text(xs(q) + 9, ya + 38, f"median flaring site {q:.2f} MW", size=11.5, fill=INK)
    total = m.sum()
    body += lines(x0, yb + 84, [
        f"Ticks: the {len(fl)} sites seen flaring on clear nights, {PERIOD}. Together they average {total:,.0f} MW,",
        f"which converts to {total * 0.0116:.1f} billion m³ a year. One megawatt held all year is 11.6 million m³ of gas.",
    ], size=12.5, fill=INK2)
    svg("volume", yb + 118, "Converting mean radiant heat into an annual gas volume", body)


def fig_bowtie():
    body = ""
    x0, x1 = 92, 720
    xc = (x0 + x1) / 2
    H = 46
    def half(x):
        a = abs(x - xc) / (x1 - xc)
        return H / 2 * (1 + 1.05 * a**2)
    cols = [FL[1], FL[2], CLEAR]
    for k, ccol in enumerate(cols):
        yc = 92 + k * H
        top = [(x, yc - half(x)) for x in np.linspace(x0, x1, 60)]
        bot = [(x, yc + half(x)) for x in np.linspace(x1, x0, 60)]
        pts = top + bot
        body += path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z", stroke=ccol, w=1.4, fill=ccol, fill_opacity=0.16)
        body += text(x0 - 10, yc + 4, f"scan {k + 1}", size=11.5, fill=INK3, anchor="end", mono=True)
    fx, fy = x0 + 46, 92 + 0.5 * H + 4
    body += circle(fx, fy, 5, fill=GOLD)
    body += circle(fx, fy, 11, fill="none", stroke=GOLD, sw=1)
    body += text(fx + 18, fy - 36, "flare in the overlap:", size=12.5, fill=INK, weight=600)
    body += text(fx + 18, fy - 20, "seen by scans 1 and 2", size=12.5, fill=INK2)
    body += text(xc, 26, "nadir: scans meet edge to edge", size=12.5, fill=INK2, anchor="middle")
    body += arrow(xc, 32, xc, 46)
    body += text(x1, 26, "edge: scans overlap", size=12.5, fill=INK2, anchor="end")
    body += lines(x0, 92 + 3 * H + 34, [
        "VIIRS sweeps 16 detector rows per scan. Toward the swath edges the footprint widens, so one flare can be recorded",
        "twice, about one pixel apart, in consecutive scans. These bow-tie copies are one flare, not two.",
    ], size=12.5, fill=INK2)
    svg("bowtie", 92 + 3 * H + 72, "Bow-tie overlap: one flare recorded in two consecutive scans", body)


def fig_neighbours():
    body = ""
    def panel(ox, title, note, assign_both):
        out = text(ox, 34, title, size=13.5, fill=INK, weight=650)
        g = 64
        for r, c in itertools.product(range(3), range(4)):
            val = {(1, 1): 1.0, (1, 2): 0.32, (0, 1): 0.12, (2, 1): 0.1, (1, 0): 0.1}.get((r, c), 0.0)
            col = FL[1 + min(4, int(val * 5))] if val > 0.05 else RAISE
            out += rect(ox + c * g, 52 + r * g, g - 2, g - 2, fill=col, opacity=0.35 + 0.65 * val if val > 0.05 else 1)
        ax, ay = ox + 1.45 * g, 52 + 1.5 * g
        bx, by = ox + 2.6 * g, 52 + 1.35 * g
        out += rect(ax - 7, ay - 5, 14, 10, fill=INK2)
        out += rect(bx - 7, by - 5, 14, 10, fill=INK2)
        out += text(ax, ay + 24, "A", size=13, fill=BG, anchor="middle", weight=700)
        out += text(bx, by + 24, "B", size=13, fill=INK, anchor="middle", weight=700)
        out += circle(ax + 6, ay - 12, 4, fill=GOLD)
        out += text(ox, 52 + 3 * g + 22, "A: flaring", size=12, fill=GOLD, mono=True)
        out += text(ox + 120, 52 + 3 * g + 22, "B: flaring" if assign_both else "B: no flare", size=12, fill=GOLD if assign_both else INK3, mono=True)
        out += lines(ox, 52 + 3 * g + 48, note, size=12, fill=INK2)
        return out
    body += panel(30, "Pixel by pixel", ["Glow spilling into B's pixel", "counts as a second flare."], True)
    body += panel(410, "As one hot patch", ["The patch is one source, given", "to the site nearest its centre."], False)
    body += text(30, 380, "Platforms A and B sit 1.2 km apart; edge pixels are 1.6 km wide. 3,858 of 7,637 sites have another site within 2 km.", size=12, fill=INK3)
    svg("neighbours", 398, "Assigning one flare to one of two nearby platforms", body)


def fig_viewangle(edge):
    body = ""
    # Left: flame seen from above and from the side.
    body += text(30, 34, "Same flame, two views", size=13.5, fill=INK, weight=650)
    R, Hh = 26, 42
    body += f'<ellipse cx="90" cy="120" rx="{R}" ry="{R}" fill="{AMBER}" opacity="0.85"/>'
    body += text(90, 170, "from above", size=12, fill=INK2, anchor="middle")
    body += text(90, 188, "area πR²", size=12, fill=INK3, anchor="middle", mono=True)
    body += f'<ellipse cx="200" cy="120" rx="{R}" ry="{Hh}" fill="{AMBER}" opacity="0.85"/>'
    body += text(200, 186, "from the side", size=12, fill=INK2, anchor="middle")
    body += text(200, 204, "area πRH", size=12, fill=INK3, anchor="middle", mono=True)
    # Right: apparent area vs satellite zenith for H/R 1, 1.6, 2.
    x0, x1, y0, y1 = 330, 730, 40, 230
    xs = Lin(0, 70, x0, x1)
    ys = Lin(0.9, 2.1, y1, y0)
    body += frame(x0, y0, x1, y1)
    body += yticks(ys, (1.0, 1.5, 2.0), x0, fmt=lambda v: f"{v:.1f}×", grid_right=x1)
    body += xticks(xs, (0, 20, 40, 60), y1, fmt=lambda v: f"{v}°")
    body += rect(xs(0), y0, xs(36) - xs(0), y1 - y0, fill=CLEAR, opacity=0.12)
    body += rect(xs(53), y0, xs(70) - xs(53), y1 - y0, fill=FL[2], opacity=0.14)
    body += text(xs(18), y1 - 10, "nadir zone", size=11.5, fill=INK2, anchor="middle")
    body += text(xs(61.5), y1 - 10, "edge zone", size=11.5, fill=INK2, anchor="middle")
    for hr, col in ((1.0, FL[2]), (1.6, FL[3]), (2.0, FL[4])):
        pts = [(xs(z), ys(G.flame_area_ratio(z, hr))) for z in np.linspace(0, 70, 80)]
        body += poly(pts, stroke=col, w=2.2)
        body += halo(xs(70) - 4, ys(G.flame_area_ratio(70, hr)) - 7, f"H/R {hr:g}", size=11, fill=INK2, anchor="end", mono=True)
    body += text((x0 + x1) / 2, y1 + 36, "satellite zenith angle", size=12, fill=INK3, anchor="middle")
    body += text(x0 - 40, y0 - 14, "apparent flame area vs. straight down", size=12, fill=INK3)
    body += lines(30, 300, [
        "Flames are taller than wide (published H/R 1–4, mean 1.6), so a satellite near the swath edge sees more flame.",
        f"Measured at {edge['n_sites']} persistent flares over {N_NIGHTS} nights: edge passes read {edge['median']:.2f}× nadir passes (median);",
        f"H/R 1.6 predicts {G.flame_zone_ratio(1.6):.2f}. EOG's calibration does not correct for this, and neither does this pipeline.",
    ], size=12.5, fill=INK2)
    svg("viewangle", 356, "Flames look larger from the side than from above", body)


def fig_sizes(run):
    det = run["det"]
    x0, x1, y0, y1 = 70, 730, 56, 230
    bins = np.geomspace(0.03, 1000, 46)
    h, _ = np.histogram(det.rh_mw.clip(0.03, 999), bins)
    xs = Log(0.03, 1000, x0, x1)
    ys = Lin(0, h.max() * 1.1, y1, y0)
    body = frame(x0, y0, x1, y1)
    step = nice_step(h.max() * 1.1)
    body += yticks(ys, range(0, int(h.max() * 1.1), step), x0, fmt=lambda v: f"{v:,}", grid_right=x1)
    body += xticks(xs, (0.1, 1, 10, 100, 1000), y1, fmt=lambda v: f"{v:g}")
    for i, c in enumerate(h):
        if c == 0:
            continue
        xa, xb = xs(bins[i]) + 1, xs(bins[i + 1]) - 1
        body += f'<rect x="{xa:.1f}" y="{ys(c):.1f}" width="{xb - xa:.1f}" height="{y1 - ys(c):.1f}" fill="{AMBER}" rx="2"><title>{bins[i]:.2g}–{bins[i + 1]:.2g} MW: {c} detections</title></rect>'
    med = det.rh_mw.median()
    body += line(xs(med), y0, xs(med), y1, stroke=CYAN, w=1.5, dash="5 4")
    # Reserve a label band above the plot, clear of bars and the SVG edge.
    body += text(xs(med) + 6, y0 - 16, f"median {med:.2f} MW", size=12, fill=CYAN)
    body += text((x0 + x1) / 2, y1 + 38, "radiant heat per detection (MW, log scale)", size=12, fill=INK3, anchor="middle")
    body += text(16, (y0 + y1) / 2, "detections", size=12, fill=INK3, anchor="middle", transform=f"rotate(-90 16 {(y0 + y1) / 2})")
    body += lines(x0, y1 + 66, [
        f"{len(det):,} detections, {PERIOD}. The left edge is set by the sensor, not by the flares:",
        "flares below roughly 0.3 MW are often missed, more so at the swath edge than at nadir.",
    ], size=12.5, fill=INK2)
    svg("sizes", y1 + 100, "How large detected flares are", body)


def fig_satellites(run):
    d = run["det"]
    x = d[d.clear].groupby(["night", "site_id", "satellite"]).rh_mw.mean().unstack()
    q = x[["NOAA-20", "NOAA-21"]].dropna()
    r = np.log2(q["NOAA-20"] / q["NOAA-21"])
    x0, x1, y0, y1 = 70, 730, 30, 230
    bins = np.linspace(-4, 4, 41)
    h, _ = np.histogram(r.clip(-3.99, 3.99), bins)
    xs = Lin(-4, 4, x0, x1)
    ys = Lin(0, h.max() * 1.15, y1, y0)
    body = frame(x0, y0, x1, y1)
    body += rect(xs(-4), y0, xs(-1) - xs(-4), y1 - y0, fill=FL[1], opacity=0.18)
    body += rect(xs(1), y0, xs(4) - xs(1), y1 - y0, fill=FL[1], opacity=0.18)
    body += yticks(ys, range(0, int(h.max() * 1.15), nice_step(h.max() * 1.15)), x0, fmt=lambda v: f"{v:,}", grid_right=x1)
    for v, lab in ((-3, "⅛×"), (-2, "¼×"), (-1, "½×"), (0, "same"), (1, "2×"), (2, "4×"), (3, "8×")):
        body += text(xs(v), y1 + 17, lab, size=11.5, fill=INK3, anchor="middle", mono=True)
    for i, c in enumerate(h):
        xa, xb = xs(bins[i]) + 1, xs(bins[i + 1]) - 1
        body += f'<rect x="{xa:.1f}" y="{ys(c):.1f}" width="{xb - xa:.1f}" height="{y1 - ys(c):.1f}" fill="{CYAN if abs(bins[i] + 0.1) < 1 else AMBER}" rx="2"><title>{c} site-nights</title></rect>'
    out = float((r.abs() > 1).mean())
    body += text(xs(-2.5), y0 + 18, "differ by more than 2×", size=12, fill=INK, anchor="middle")
    body += text(xs(2.5), y0 + 18, "differ by more than 2×", size=12, fill=INK, anchor="middle")
    body += text((x0 + x1) / 2, y1 + 38, "NOAA-20 reading ÷ NOAA-21 reading, same site, same night", size=12, fill=INK3, anchor="middle")
    body += lines(x0, y1 + 66, [
        f"{len(q):,} site-nights where both satellites saw the flare through clear sky. {out:.0%} differ by more than 2×;",
        "flares flicker, and the satellites see them minutes apart from different angles. One pass is a sample.",
    ], size=12.5, fill=INK2)
    svg("satellites", y1 + 100, "How two satellites' readings of the same flare compare", body)


def fig_validation(val):
    v = val[(val.bcm_per_year > 0) & (val.myc_bcm > 0)]
    x0, x1, y0, y1 = 80, 450, 30, 400
    # axis floor at 10^-6; the rare value below it is drawn hollow on the edge
    lo = max(1e-6, 10 ** math.floor(math.log10(min(v.bcm_per_year.min(), v.myc_bcm.min()))))
    xs = Log(lo, 3, x0, x1)
    ys = Log(lo, 3, y1, y0)
    body = frame(x0, y0, x1, y1)
    ticks = [10.0**k for k in range(int(round(math.log10(lo))), 1)]
    fmt = lambda t: f"{t:.{max(0, -int(round(math.log10(t))))}f}"  # noqa: E731
    body += yticks(ys, ticks, x0, fmt=fmt, grid_right=x1)
    body += xticks(xs, ticks, y1, fmt=fmt, grid_top=y0)
    body += line(xs(lo), ys(lo), xs(3), ys(3), stroke=INK3, w=1.2, dash="5 4")
    body += text(xs(1.2), ys(2.2) - 6, "1:1", size=11.5, fill=INK3, anchor="end", mono=True)
    n_edge = 0
    for r in v.itertuples():
        edge = r.myc_bcm < lo or r.bcm_per_year < lo
        n_edge += edge
        tip = f"site {r.site_id}: ours {r.bcm_per_year:.2g}, EOG {r.myc_bcm:.2g} BCM/yr"
        body += circle(xs(max(r.myc_bcm, lo)), ys(max(r.bcm_per_year, lo)), 3.2,
                       fill=BG if edge else AMBER, stroke=AMBER if edge else None, sw=1.3,
                       opacity=1 if edge else 0.75, tip=tip)
    body += text((x0 + x1) / 2, y1 + 38, "EOG annual catalog, 2025 (BCM/yr)", size=12, fill=INK3, anchor="middle")
    body += text(18, (y0 + y1) / 2, f"this pipeline, {N_NIGHTS} nights (BCM/yr)", size=12, fill=INK3, anchor="middle", transform=f"rotate(-90 18 {(y0 + y1) / 2})")
    lr = np.corrcoef(np.log(v.myc_bcm), np.log(v.bcm_per_year))[0, 1]
    ratio = float((v.bcm_per_year / v.myc_bcm).median())
    tx = 486
    body += text(tx, 60, f"{len(v)} sites", size=22, fill=INK, weight=600, mono=True)
    body += lines(tx, 84, ["flaring in both,", "matched within 2 km"], size=12, fill=INK2)
    body += text(tx, 146, f"r = {lr:.2f}", size=22, fill=INK, weight=600, mono=True)
    body += lines(tx, 170, ["correlation of the logs"], size=12, fill=INK2)
    body += text(tx, 216, f"{ratio:.2f}×", size=22, fill=INK, weight=600, mono=True)
    body += lines(tx, 240, ["median of ours ÷ EOG"], size=12, fill=INK2)
    body += lines(tx, 290, [
        "Different years (Aug–Nov 2026",
        f"vs all of 2025), {N_NIGHTS} nights",
        "vs 365, and a different site list.",
        "Agreement is in rank and order",
        "of magnitude, not site by site.",
    ], size=12, fill=INK3)
    if n_edge:
        body += lines(tx, 394, [
            f"Hollow: {n_edge} site{'s' if n_edge > 1 else ''} below {lo:.6f}",
            "BCM/yr, drawn on the edge.",
        ], size=12, fill=INK3)
    svg("validation", y1 + 52, "Site volumes from this pipeline against EOG's 2025 annual catalog", body)


def fig_fitguard():
    t, esf, *_ = nf.fit_planck(SPEC_3802["bands"], SPEC_3802["obs"], SPEC_3802["noise"])
    body, xs, ys, (x0, x1, y0, y1) = _spectrum_axes(150)
    body += _curve(xs, ys, 0.03793270655238951, 916.1618169511964, CLEAR, w=2.4, dash="7 5")
    body += _curve(xs, ys, esf, t, AMBER)
    body += _points(xs, ys, dropped=("M7",))
    tx = x1 + 22
    body += text(tx, 54, "Before (v3)", size=13, fill=CLEAR, weight=650)
    body += lines(tx, 74, ["passes through M7, M8:", "916 K, 3.8% of pixel", "→ 3,127 MW"], size=12, fill=INK2, mono=True)
    body += text(tx, 150, "After (v5)", size=13, fill=AMBER, weight=650)
    body += lines(tx, 170, [f"five bands, {t:,.0f} K", "→ 115 MW", "(28× smaller)"], size=12, fill=INK2, mono=True)
    body += lines(tx, 234, ["The cool, huge flame", "overshoots M10 by 2.5×", "and M11–M13 by 9–55×."], size=12, fill=INK3)
    svg("fitguard", y1 + 52, "A fit that used the wrong bands, and the corrected fit", body)


def fig_rh75():
    body = ""
    vals = [
        [0.0, 0.4, 0.6, 0.3, 0.0],
        [0.5, 8.0, 71.1, 9.0, 0.3],
        [0.0, 6.0, 12.0, 5.5, 0.2],
        [0.0, 0.2, 0.4, 1775.9, 0.0],
    ]
    def panel(ox, title, chosen, note):
        out = text(ox, 34, title, size=13.5, fill=INK, weight=650)
        g = 62
        for r, row in enumerate(vals):
            for c, v in enumerate(row):
                x, y = ox + c * g, 50 + r * g
                sel = (r, c) in chosen
                col = AMBER if sel else (RAISE if v < 1 else COAST)
                out += rect(x, y, g - 3, g - 3, fill=col, rx=4)
                if v > 0:
                    lab = f"{v:,.0f}" if v >= 100 else f"{v:g}"
                    out += text(x + (g - 3) / 2, y + 35, lab, size=11.5, fill=BG if sel else INK2, anchor="middle", mono=True)
        out += lines(ox, 50 + 4 * g + 24, note, size=12, fill=INK2)
        return out
    body += panel(30, "Old rule (v3)", {(3, 3)}, ["Anchored on the highest-heat pixel:", "one stray pixel with a bad fit", "set the site's value: 1,776 MW."])
    body += panel(410, "Peak-anchored (v5)", {(1, 2)}, ["Anchored on the 1.6 µm peak; only", "its 8 neighbours within 0.75–1.5×", "the peak are added: 71 MW."])
    body += text(30, 404, "MW per pixel, NOAA-20, 3 Nov 2026, site 3804 (values simplified for display).", size=12, fill=INK3)
    svg("rh75", 420, "Which pixels add up to a flare's radiant heat", body)


def fig_waterfall():
    steps = [
        ("v3 total", 76784, "total"),
        ("Impossible fits removed", -5917, "d"),
        ("Double counts removed", -3783, "d"),
        ("Distant pixels no longer summed", -3010, "d"),
        ("Bow-tie copies averaged", -1981, "d"),
        ("Moved or dropped at 2 km", -1397, "d"),
        ("Splits and refits", -263, "d"),
        ("New detections", 719, "d"),
        ("v5 total", 61153, "total"),
    ]
    x0, x1, y0 = 250, 730, 24
    xs = Lin(0, 80000, x0, x1)
    row = 30
    body = ""
    for v in (0, 20000, 40000, 60000, 80000):
        body += line(xs(v), y0, xs(v), y0 + row * len(steps), stroke=RULE)
        body += text(xs(v), y0 + row * len(steps) + 16, f"{v // 1000:,}k", size=11, fill=INK3, anchor="middle", mono=True)
    run = 0
    for i, (lab, v, kind) in enumerate(steps):
        y = y0 + i * row
        body += text(x0 - 10, y + 19, lab, size=12.5, fill=INK if kind == "total" else INK2, anchor="end", weight=650 if kind == "total" else 400)
        if kind == "total":
            body += rect(xs(0), y + 6, xs(v) - xs(0), row - 10, fill=GOLD if i == 0 else AMBER, rx=3)
            body += text(xs(v) - 8, y + 19, f"{v:,} MW", size=12, fill=BG, anchor="end", weight=600, mono=True)
            run = v
        else:
            a, b = sorted((run, run + v))
            body += rect(xs(a), y + 8, max(xs(b) - xs(a), 2), row - 14, fill=CLEAR if v < 0 else CYAN, rx=2)
            body += text(xs(a) - 6, y + 19, f"{v:+,}", size=12, fill=INK2, anchor="end", mono=True)
            run += v
    body += lines(30, y0 + row * len(steps) + 44, [
        "Radiant heat summed over every detection, 6 October to 4 November 2026: the 30 nights run with both v3 and v5.",
        "81% of the drop removes v3 errors.",
    ], size=12, fill=INK3)
    svg("waterfall", y0 + row * len(steps) + 80, "Where the 20% drop in radiant heat from v3 to v5 comes from", body)


def edge_nadir(run, val, copies) -> dict:
    """Gate 7's edge/nadir ratio (sites detected on >= 80% of clear passes at
    both geometries) over every segment."""
    with tempfile.TemporaryDirectory() as tmp:
        val.to_csv(Path(tmp) / f"validation_myc25_{TAG88}.csv", index=False)
        g7 = G.gate7(run, run, copies, Path(tmp), TAG88, {})
    return g7["run_site_col"]["rh_mw"]["rate_ge_0.8"]


def main():
    sites, run, nights, val, copies = load()
    fig_pipeline()
    fig_swath(run)
    fig_subpixel()
    fig_spectrum()
    fig_detection()
    fig_fit()
    fig_heat(run)
    fig_nights(run, nights)
    fig_volume(nights)
    fig_bowtie()
    fig_neighbours()
    fig_viewangle(edge_nadir(run, val, copies))
    fig_sizes(run)
    fig_satellites(run)
    fig_validation(val)
    fig_fitguard()
    fig_rh75()
    fig_waterfall()


# %%
if __name__ == "__main__":
    main()
