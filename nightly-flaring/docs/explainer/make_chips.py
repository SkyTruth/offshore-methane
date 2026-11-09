# %%
"""Render real VIIRS image chips for docs/explainer.md.

Each chip shows the 1.6 µm (M10) radiance of one satellite pass around a site,
with every pixel drawn at its ground footprint from the granule's geolocation:
pixels grow and overlap toward the swath edge exactly as in the data. Pixels
NOAA blanks for bow-tie overlap are not drawn. Cloudy pixels (cloud mask 2/3)
are hatched. Site markers are GFW oil sites; values come from the v5 30-night
run. Needs network for the first run (NOAA NODD, about 30 MB per granule);
windows are cached in ``outputs/chips_cache``.
Run from ``nightly-flaring/``: ``.venv/bin/python docs/explainer/make_chips.py``.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import gates_30n as G  # noqa: E402
import granule as gr  # noqa: E402
import nodd  # noqa: E402
import pipeline as p  # noqa: E402

CACHE = p.OUT_DIR / "chips_cache"
BG, PANEL, RULE, INK, INK2, INK3 = (
    "#00131F",
    "#001929",
    "#043352",
    "#F9F6F3",
    "#9CB7C9",
    "#60869F",
)
GOLD, CYAN = "#FEBD81", "#9FC6BA"
CMAP = LinearSegmentedColormap.from_list(
    "m10",
    [
        (0.00, "#000D16"),
        (0.30, "#002640"),
        (0.42, "#1D5072"),
        (0.50, "#4C2501"),
        (0.60, "#7E3D01"),
        (0.70, "#CA6202"),
        (0.80, "#FD8E28"),
        (0.90, "#FEBD81"),
        (1.00, "#F7E5D4"),
    ],
)
NORM = LogNorm(0.004, 20, clip=True)
plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "text.color": INK,
        "axes.edgecolor": RULE,
        "savefig.facecolor": BG,
        "figure.facecolor": BG,
    }
)

SITES = pd.read_csv(p.SITES_CSV)


def _records():
    dd = G.load(p.OUT_DIR / "v5_30n", G.TAG, SITES)["dd"]
    return dd


def fetch(sat: str, gid: str, lat0: float, lon0: float, half_km: float) -> dict:
    """Window of geolocation, M10 radiance and cloud mask around a point."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{sat}_{gid}_{lat0:.3f}_{lon0:.3f}_{half_km:g}.npz"
    if path.exists():
        return dict(np.load(path))
    session = requests.Session()
    day = datetime.strptime(gid[1:9], "%Y%m%d")
    keys = nodd.granule_index(sat, day, session)[gid]
    base = nodd.bucket_url(nodd.BUCKETS[sat][0])
    geo = "All_Data/VIIRS-MOD-GEO-TC_All"
    (lat, lon), _ = gr._read(f"{base}/{keys['geo']}", [f"{geo}/Latitude", f"{geo}/Longitude"])
    l10, _, _ = gr._band(f"{base}/{keys['M10']}", "M10")
    x, y = _km(lat, lon, lat0, lon0)
    near = (np.abs(x) < half_km + 4) & (np.abs(y) < half_km + 4) & (lat > -900)
    rr, cc = np.nonzero(near)
    r0, r1 = max(rr.min() - 2, 0), min(rr.max() + 3, lat.shape[0])
    c0, c1 = max(cc.min() - 2, 0), min(cc.max() + 3, lat.shape[1])
    cloud = np.full((r1 - r0, c1 - c0), -1, dtype=np.int16)
    if keys.get("cloud"):
        f = nodd.HTTPRangeFile(f"{base}/{keys['cloud']}", session)
        with h5py.File(f, "r") as h:
            ds = h["CloudMask"]
            if ds.shape == lat.shape:
                cloud = ds[r0:r1, c0:c1].astype(np.int16)
    out = dict(
        lat=lat[r0:r1, c0:c1],
        lon=lon[r0:r1, c0:c1],
        l10=l10[r0:r1, c0:c1],
        cloud=cloud,
        row0=np.int64(r0),
        col0=np.int64(c0),
    )
    np.savez_compressed(path, **out)
    return out


def _km(lat, lon, lat0, lon0):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111.32
    y = (lat - lat0) * 110.57
    return x, y


def footprints(w: dict, lat0: float, lon0: float):
    """Pixel quadrilaterals in km around (lat0, lon0), with values and cloud."""
    x, y = _km(w["lat"], w["lon"], lat0, lon0)
    nr, nc = x.shape
    row0 = int(w["row0"])
    polys, vals, cloudy = [], [], []
    for r in range(nr):
        scan = (row0 + r) // 16
        for c in range(nc):
            v = w["l10"][r, c]
            if not np.isfinite(v) or w["lat"][r, c] < -900:
                continue
            ca, cb = max(c - 1, 0), min(c + 1, nc - 1)
            ra = r - 1 if r > 0 and (row0 + r - 1) // 16 == scan else r
            rb = r + 1 if r < nr - 1 and (row0 + r + 1) // 16 == scan else r
            if cb == ca or rb == ra:
                continue
            ax = (x[r, cb] - x[r, ca]) / (cb - ca) / 2
            ay = (y[r, cb] - y[r, ca]) / (cb - ca) / 2
            bx = (x[rb, c] - x[ra, c]) / (rb - ra) / 2
            by = (y[rb, c] - y[ra, c]) / (rb - ra) / 2
            px, py = x[r, c], y[r, c]
            polys.append(
                [
                    (px - ax - bx, py - ay - by),
                    (px + ax - bx, py + ay - by),
                    (px + ax + bx, py + ay + by),
                    (px - ax + bx, py - ay + by),
                ]
            )
            vals.append(max(float(v), 1e-4))
            cloudy.append(int(w["cloud"][r, c]) in (2, 3))
    return polys, np.array(vals), np.array(cloudy)


def draw_chip(ax, w, lat0, lon0, half_km, site_ids=(), labels=None, mark=None):
    polys, vals, cloudy = footprints(w, lat0, lon0)
    pc = PolyCollection(polys, array=vals, cmap=CMAP, norm=NORM, edgecolors="#00131F", linewidths=0.35)
    ax.add_collection(pc)
    if cloudy.any():
        cl = PolyCollection(
            [pp for pp, k in zip(polys, cloudy) if k],
            facecolors=(0.85, 0.9, 0.95, 0.10),
            edgecolors=(0.85, 0.9, 0.95, 0.45),
            linewidths=0.0,
            hatch="////",
        )
        ax.add_collection(cl)
    near = SITES[
        (np.abs(_km(SITES.lat, SITES.lon, lat0, lon0)[0]) < half_km)
        & (np.abs(_km(SITES.lat, SITES.lon, lat0, lon0)[1]) < half_km)
    ]
    sx, sy = _km(near.lat, near.lon, lat0, lon0)
    ax.scatter(sx, sy, s=16, marker="s", facecolors="none", edgecolors="white", linewidths=0.9, zorder=5)
    for sid, text in (labels or {}).items():
        row = SITES.set_index("site_id").loc[sid]
        tx, ty = _km(row.lat, row.lon, lat0, lon0)
        ax.annotate(
            text,
            (tx, ty),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7.5,
            color=INK,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc=(0.04, 0.08, 0.11, 0.8), ec="none"),
        )
    ax.set_xlim(-half_km, half_km)
    ax.set_ylim(-half_km, half_km)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(RULE)
    # 5 km scale bar
    x0, y0 = half_km - 0.07 * 2 * half_km - 5, -half_km + 0.07 * 2 * half_km
    ax.plot([x0, x0 + 5], [y0, y0], color=INK, lw=1.6, solid_capstyle="butt", zorder=6)
    ax.text(x0 + 2.5, y0 + 0.035 * 2 * half_km, "5 km", ha="center", fontsize=7.5, color=INK, zorder=6)
    return pc


def panel_title(ax, title, sub, value=None):
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK, pad=17)
    ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=7.8, color=INK2, va="bottom")
    if value:
        ax.annotate(value, (1, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", fontsize=9.5, color=GOLD, va="bottom", ha="right", fontweight="bold")


def colorbar(fig, pc, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_ticks([0.01, 0.1, 1, 10])
    cb.set_ticklabels(["0.01", "0.1", "1", "10"])
    cb.ax.tick_params(colors=INK2, labelsize=7.5, length=2)
    cb.outline.set_edgecolor(RULE)
    cb.set_label("1.6 µm (M10) radiance, W m⁻² sr⁻¹ µm⁻¹  ·  hatched: cloud  ·  white squares: GFW oil sites", color=INK2, fontsize=7.5)


def zone(col):
    col = int(col)
    if 1008 <= col < 2192:
        return "near nadir"
    if col < 640 or col >= 2560:
        return "swath edge"
    return "mid-swath"


def when(sat, gid):
    t = datetime.strptime(gid[1:16], "%Y%m%d_t%H%M%S")
    return f"{sat} · {t:%-d %b %H:%M} UTC"


def rec(dd, sat, gid, site):
    r = dd[(dd.satellite == sat) & (dd.granule == gid) & (dd.site_id == site)]
    return r.iloc[0]


def fig_states(dd):
    site = 928
    s = SITES.set_index("site_id").loc[site]
    half = 8
    cases = [
        ("Flare, clear sky", "SNPP", "d20261031_t0806577"),
        ("Clear sky, no flare", "SNPP", "d20261009_t0819165"),
        ("Flare under cloud", "NOAA-20", "d20261007_t0736028"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.45), dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.24, wspace=0.06)
    for ax, (title, sat, gid) in zip(axes, cases):
        w = fetch(sat, gid, s.lat, s.lon, half)
        r = rec(dd, sat, gid, site)
        pc = draw_chip(ax, w, s.lat, s.lon, half)
        val = f"{r.rh_mw:.1f} MW" if r.detected else "0 MW"
        if r.detected and not r.clear:
            val = "not used"
        panel_title(ax, title, f"{when(sat, gid)} · {zone(r.col)}", val)
    colorbar(fig, pc, [0.18, 0.13, 0.64, 0.032])
    fig.savefig(HERE / "chips_states.png")
    plt.close(fig)
    print("wrote chips_states.png")


def fig_night(dd):
    site = 928
    s = SITES.set_index("site_id").loc[site]
    half = 9
    cases = [
        ("NOAA-21", "d20261102_t0652090"),
        ("SNPP", "d20261102_t0729083"),
        ("NOAA-20", "d20261102_t0749023"),
        ("NOAA-21", "d20261102_t0831431"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.0), dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.28, wspace=0.06)
    for ax, (sat, gid) in zip(axes, cases):
        w = fetch(sat, gid, s.lat, s.lon, half)
        r = rec(dd, sat, gid, site)
        pc = draw_chip(ax, w, s.lat, s.lon, half)
        t = datetime.strptime(gid[1:16], "%Y%m%d_t%H%M%S")
        panel_title(ax, f"{sat} {t:%H:%M}", f"{zone(r.col)}, column {int(r.col):,}", f"{r.rh_mw:.1f} MW")
    colorbar(fig, pc, [0.18, 0.15, 0.64, 0.035])
    fig.savefig(HERE / "chips_night.png")
    plt.close(fig)
    print("wrote chips_night.png")


def fig_bowtie(dd):
    site, sat, gid = 3802, "NOAA-20", "d20261010_t2330408"
    s = SITES.set_index("site_id").loc[site]
    half = 9
    w = fetch(sat, gid, s.lat, s.lon, half)
    copies = pd.read_csv(p.OUT_DIR / "v5_30n" / f"copies_{G.TAG}.csv")
    cp = copies[(copies.satellite == sat) & (copies.granule == gid) & (copies.assigned_site == site)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.0), dpi=200, gridspec_kw=dict(width_ratios=[1, 1]))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.27, wspace=0.12)
    # Left: the image as recorded, rows and columns, scans separated.
    row0, col0 = int(w["row0"]), int(w["col0"])
    pr, pc_ = int(cp.peak_row.iloc[0]), int(cp.peak_col.iloc[0])
    rs = slice(max(pr - 14 - row0, 0), pr + 20 - row0)
    cs = slice(max(pc_ - 10 - col0, 0), pc_ + 10 - col0)
    img = np.where(np.isfinite(w["l10"][rs, cs]), w["l10"][rs, cs], np.nan)
    rr0, cc0 = rs.start + row0, cs.start + col0
    im = a1.imshow(
        img,
        cmap=CMAP,
        norm=NORM,
        interpolation="nearest",
        extent=(cc0 - 0.5, cc0 + img.shape[1] - 0.5, rr0 + img.shape[0] - 0.5, rr0 - 0.5),
        aspect="auto",
    )
    a1.set_facecolor("#001929")
    a1.set_ylim(rr0 + img.shape[0] - 0.5, rr0 - 0.5)
    for k in range((rr0 // 16) * 16, rr0 + img.shape[0] + 16, 16):
        a1.axhline(k - 0.5, color=INK3, lw=0.8, ls=(0, (3, 3)))
    a1.tick_params(colors=INK3, labelsize=7)
    a1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    others = copies[
        (copies.satellite == sat)
        & (copies.granule == gid)
        & (copies.assigned_site != site)
        & copies.peak_row.between(rr0, rr0 + img.shape[0] - 1)
        & copies.peak_col.between(cc0, cc0 + img.shape[1] - 1)
    ]
    for r in others.itertuples():
        a1.annotate(
            f"another flare (site {int(r.assigned_site)})",
            (r.peak_col, r.peak_row),
            xytext=(14, 0),
            textcoords="offset points",
            color=INK2,
            fontsize=7,
            va="center",
            arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8),
        )
    a1.set_xlabel("column", color=INK3, fontsize=7.5)
    a1.set_ylabel("row (16 rows per scan)", color=INK3, fontsize=7.5)
    for r in cp.itertuples():
        a1.annotate(
            f"copy, scan {r.scan}: {r.rh75_mw:.0f} MW",
            (r.peak_col, r.peak_row),
            xytext=(14, 0),
            textcoords="offset points",
            color=INK,
            fontsize=7.5,
            va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
        )
    panel_title(a1, "As recorded: two hot spots", f"{when(sat, gid)} · swath edge (column {pc_:,})")
    # Right: the same pixels on the ground.
    draw_chip(a2, w, s.lat, s.lon, half)
    for r in cp.itertuples():
        x, y = _km(r.peak_lat, r.peak_lon, s.lat, s.lon)
        a2.scatter([x], [y], s=110, facecolors="none", edgecolors=CYAN, linewidths=1.0, zorder=7)
    d = float(cp.pair_distance_px.astype(str).str.split(";").str[0].astype(float).iloc[0])
    gm = float(cp.group_rh_mw.iloc[0])
    panel_title(a2, "On the ground: one flare", f"copies {d:.2f} pixel apart · averaged", f"{gm:.0f} MW")
    colorbar(fig, im, [0.18, 0.09, 0.64, 0.028])
    fig.savefig(HERE / "chips_bowtie.png")
    plt.close(fig)
    print("wrote chips_bowtie.png")


def fig_field(dd):
    site, sat, gid = 4500, "NOAA-20", "d20261104_t2223011"
    s = SITES.set_index("site_id").loc[site]
    half = 11
    w = fetch(sat, gid, s.lat, s.lon, half)
    r = dd[(dd.satellite == sat) & (dd.granule == gid) & dd.detected]
    r = r.merge(SITES[["site_id", "lat", "lon"]], on="site_id")
    x, y = _km(r.lat, r.lon, s.lat, s.lon)
    r = r[(np.abs(x) < half - 0.5) & (np.abs(y) < half - 0.5)]
    labels = {int(q.site_id): f"{q.rh_mw:.1f} MW" for q in r.itertuples()}
    fig, ax = plt.subplots(figsize=(5.8, 6.3), dpi=200)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.89, bottom=0.15)
    pc = draw_chip(ax, w, s.lat, s.lon, half, labels=labels)
    col = int(dd[(dd.satellite == sat) & (dd.granule == gid) & (dd.site_id == site)].col.iloc[0])
    n_sites = int(
        ((np.abs(_km(SITES.lat, SITES.lon, s.lat, s.lon)[0]) < half) & (np.abs(_km(SITES.lat, SITES.lon, s.lat, s.lon)[1]) < half)).sum()
    )
    panel_title(ax, "A crowded field: Abu Dhabi offshore", f"{when(sat, gid)} · {zone(col)} · {n_sites} GFW sites, {len(r)} flaring", f"{r.rh_mw.sum():.0f} MW")
    colorbar(fig, pc, [0.08, 0.075, 0.84, 0.025])
    fig.savefig(HERE / "chips_field.png")
    plt.close(fig)
    print("wrote chips_field.png")


def main():
    dd = _records()
    fig_states(dd)
    fig_night(dd)
    fig_bowtie(dd)
    fig_field(dd)


# %%
if __name__ == "__main__":
    main()
