# %%
"""Build the flaring map website from one or more pipeline run segments.

A segment is a run folder holding site-night and overpass Parquet exports
(or legacy CSVs). ``find_segments`` collects every segment
under ``outputs/v5_*``; ``main`` joins them into one night series and writes a
static site to ``outputs/site/``:

- ``index.html``: the map, ranking and timeline, with static fields, totals, a
  1:50m country basemap and the "How it works" explainer inlined;
- ``data/nights-<year>.<hash>.json``: status and heat, fetched in parallel;
- ``data/clearpasses-<year>.<hash>.json``: digits fetched after first render;
- ``passes/<site_id>.json``: every satellite pass at a site that flared,
  fetched when the site is opened, so the page stays small as nights accrue;
- ``img/``: the explainer's image chips.

``publish_pages`` pushes that folder to the repo's GitHub Pages branch.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import markdown
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import validate_myc25  # noqa: E402
import store  # noqa: E402
from pipeline import dedupe_overpasses, strict_clear  # noqa: E402
from regions import REGIONS, region_of  # noqa: E402

OUT_DIR = HERE / "outputs"
SITE_DIR = OUT_DIR / "site"
SEGMENT_GLOB = "v5_*"
TEMPLATE = HERE / "viewer_template.html"
LOGO = HERE / "assets" / "skytruth-logo-dark.svg"
EXPLAINER = HERE / "docs" / "explainer.md"
REPO_BLOB = "https://github.com/SkyTruth/offshore-methane/blob/nightly-flaring/nightly-flaring/docs/"
PAGES_URL = "https://skytruth.github.io/offshore-methane/"
BASEMAP_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json"
BCM_PER_MW_YEAR = 0.0116
STATUS_LABEL = "v5 · reviewed"
STATUS = (
    "Pipeline v5 (8 Nov 2026), after five two-model reviews (GPT-6.1 Sol, Claude "
    "Opus 5.5): physical fit limits, peak-anchored RH75%, bow-tie copies paired "
    "and averaged, one source per flare assigned by its centre, strict cloud rule. "
    "Review 5 passed the 30-night run; the backfill extends it to 9 August. "
    "Not yet: M11-only and "
    "mid-infrared-only triggers, atmospheric correction, view-angle and "
    "cross-satellite calibration, non-upstream flagging."
)
SATELLITES = ["SNPP", "NOAA-20", "NOAA-21"]
STATUS_CODE = {
    "flaring": "F",
    "flaring_cloud": "D",
    "clear_no_flare": "C",
    "cloudy": "K",
    "not_observed": "N",
}
DIGITS36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def load_basemap(out_dir: Path = OUT_DIR) -> str:
    path = out_dir / "basemap_countries-50m.json"
    if not path.exists():
        resp = requests.get(BASEMAP_URL, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path.read_text()


def _r(x: float, nd: int = 3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- segments
def find_segments(
    base: Path = OUT_DIR, pattern: str = SEGMENT_GLOB
) -> list[tuple[Path, str]]:
    """Run segments ``(folder, tag)`` under ``base/pattern``, oldest start first."""
    segments = []
    for folder in base.glob(pattern):
        if not folder.is_dir():
            continue
        tags = {
            path.stem.removeprefix("site_nights_")
            for suffix in ("csv", "parquet")
            for path in folder.glob(f"site_nights_*_*.{suffix}")
        }
        for tag in tags:
            if store.segment_path(folder, tag, "overpasses").exists():
                segments.append((folder, tag))
    if not segments:
        raise FileNotFoundError(f"no run segments in {base}/{pattern}")
    return sorted(segments, key=lambda s: (s[1][:10], s[1][11:], s[0].name))


def granule_owners(segments: list[tuple[Path, str]]) -> dict:
    """Latest complete snapshot owns copies and passes, including empty exports.

    An incomplete boundary retry must not erase a prior measured recovery.
    Older exports without a completion column predate partial-run acceptance.
    """
    owners = {}
    for i, (folder, tag) in enumerate(segments):
        manifest = folder / f"jobs_{tag}.csv"
        if manifest.exists():
            coverage = pd.read_csv(
                manifest,
                usecols=lambda col: col in {"satellite", "granule", "complete"},
            )
        else:
            table = (
                "overpasses"
                if store.segment_path(folder, tag, "overpasses").exists()
                else "copies"
            )
            coverage = store.read_segment(
                folder, tag, table, columns=["satellite", "granule"]
            )
        if "complete" in coverage:
            coverage = coverage[coverage.complete.eq(True)]
        for key in (
            coverage[["satellite", "granule"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            owners[key] = i
    return owners


def load_segments(
    segments: list[tuple[Path, str]],
    *,
    _ordered: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate segments; where nights overlap the later segment wins."""
    nights, over = [], []
    owners = granule_owners(segments)
    for i, (folder, tag) in enumerate(segments):
        n = store.read_segment(folder, tag, "site_nights")
        n["_segment"] = i
        if _ordered:
            n["_row"] = np.arange(len(n))
        nights.append(n)
        o = store.read_segment(folder, tag, "overpasses")
        o["_segment"] = i
        if _ordered:
            o["_row"] = np.arange(len(o))
        keep = pd.Series(
            [
                owners.get(key) == i
                for key in o[["satellite", "granule"]].itertuples(
                    index=False, name=None
                )
            ],
            index=o.index,
            dtype=bool,
        )
        over.append(o.loc[keep])
    nights = (
        pd.concat(nights, ignore_index=True)
        .sort_values("_segment", kind="stable")
        .drop_duplicates(["site_id", "night"], keep="last")
        .drop(columns=[] if _ordered else ["_segment"])
        .reset_index(drop=True)
    )
    # A granule near a segment boundary is read by both segments.
    over = (
        pd.concat(over, ignore_index=True)
        .sort_values("_segment", kind="stable")
        .drop_duplicates(["satellite", "granule", "site_id", "time_utc"], keep="last")
        .drop(columns=[] if _ordered else ["_segment"])
        .reset_index(drop=True)
    )
    return nights, over


def load_copies(
    segments: list[tuple[Path, str]], *, _ordered: bool = False
) -> pd.DataFrame:
    """Join source-copy exports; the latest complete granule snapshot wins.

    Missing exports are an error: absence needs the source reassignment check.
    """
    owners = granule_owners(segments)
    tables = [
        store.read_segment(folder, tag, "copies").assign(_segment=i)
        for i, (folder, tag) in enumerate(segments)
    ]
    if _ordered:
        for frame in tables:
            frame["_row"] = np.arange(len(frame))
    nonempty = [table for table in tables if not table.empty]
    copies = pd.concat(nonempty or tables[:1], ignore_index=True)
    keep = [
        owners.get((sat, granule)) == segment
        for sat, granule, segment in copies[
            ["satellite", "granule", "_segment"]
        ].itertuples(index=False, name=None)
    ]
    return (
        copies.loc[keep]
        .drop(columns=[] if _ordered else ["_segment"])
        .drop_duplicates(
            ["satellite", "granule", "group_id", "peak_row", "peak_col"], keep="last"
        )
        .reset_index(drop=True)
    )


def load_tables(
    base: Path = OUT_DIR, root: Path | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sync the night store and return the website's three legacy frames."""
    root = base / "store" if root is None else root
    return store.load_tables(find_segments(base), root)


def night_series(nights: pd.DataFrame) -> list[str]:
    """Every night from the first to the last, including nights with no data."""
    days = pd.date_range(nights["night"].min(), nights["night"].max(), freq="D")
    return [d.strftime("%Y-%m-%d") for d in days]


def encode_site_nights(g: pd.DataFrame, night_list: list[str]) -> tuple[str, list, str]:
    """Status string, sparse values and clear-pass digits for one site.

    ``st`` has one code per night (F flaring, D lit through cloud, C clear with
    no flare, K cloudy, N not observed). ``rv`` holds the clear-sky radiant
    heat for each F or D night in order (None for a D night with no clear
    pass); C nights are 0 and K/N nights have no value. ``cp`` gives the
    clear passes per night as one base-36 digit.
    """
    g = g.set_index("night").reindex(night_list)
    codes = g["status"].map(STATUS_CODE).fillna("N")
    st = "".join(codes)
    rv = [_r(v) for c, v in zip(codes, g["rh_mw"]) if c in "FD"]
    cp = "".join(
        DIGITS36[min(int(v), 35)] if np.isfinite(v) else "0"
        for v in g["n_clear"].astype(float)
    )
    return st, rv, cp


def calendar_bins(night_list: list[str], unit: str) -> list[tuple[int, int]]:
    """Inclusive, calendar-aligned bins, clipped to the supplied record."""
    dates = pd.DatetimeIndex(night_list)
    if unit == "night":
        return [(i, i) for i in range(len(dates))]
    periods = dates.to_period(
        {"week": "W-SUN", "month": "M", "quarter": "Q", "year": "Y"}[unit]
    )
    edges = np.flatnonzero(np.r_[True, periods[1:] != periods[:-1], True])
    return [(int(a), int(b - 1)) for a, b in pairwise(edges)]


def decode_nights(row: list) -> tuple[np.ndarray, np.ndarray]:
    """Decode the public sparse representation without changing its rounding."""
    st, values = row[4], iter(row[5])
    heat = np.array(
        [0 if c == "C" else next(values) if c in "FD" else np.nan for c in st],
        dtype=float,
    )
    return heat, np.frombuffer(st.encode(), dtype="S1")


def bin_totals(data: dict) -> dict:
    """Network and unfiltered regional sums of site means, never pooled heat.

    Network columns are MW, flaring sites, mean clear sites/night, mean
    observed sites/night, sites with a clear night. Regions omit nightly totals.
    """
    result = {}
    plans = {
        u: calendar_bins(data["nights"], u)
        for u in ("night", "week", "month", "quarter", "year")
    }
    for unit, bins in plans.items():
        result[unit] = {
            "network": np.zeros((len(bins), 5)),
            "regions": (
                {r[8]: np.zeros(len(bins)) for r in data["sites"]}
                if unit != "night"
                else {}
            ),
        }
    for row in data["sites"]:
        heat, st = decode_nights(row)
        clear = np.isfinite(heat)
        lit, obs = st == b"F", st != b"N"
        clear_night = (st == b"F") | (st == b"C")
        prefixes = [
            np.r_[0, np.cumsum(x)]
            for x in (np.nan_to_num(heat), clear, lit, obs, clear_night)
        ]
        for unit, bins in plans.items():
            a, b = np.array(bins).T
            sums, counts, flaring, observed, clear_nights = [
                p[b + 1] - p[a] for p in prefixes
            ]
            means = np.divide(sums, counts, out=np.zeros(len(bins)), where=counts > 0)
            target = result[unit]
            target["network"] += np.column_stack(
                (
                    means,
                    flaring > 0,
                    counts / (b - a + 1),
                    observed / (b - a + 1),
                    clear_nights > 0,
                )
            )
            if unit != "night":
                target["regions"][row[8]] += means
    # Six decimal places retain far more than the displayed precision.
    for target in result.values():
        target["network"] = target["network"].round(6).tolist()
        target["regions"] = {
            r: values.round(6).tolist() for r, values in target["regions"].items()
        }
    return result


def write_night_chunks(data: dict, site_dir: Path) -> dict:
    """Return static page data and write deterministic, content-addressed years."""
    page = copy.deepcopy(data)
    page["meta"]["chunks"] = []
    page["meta"]["clearpasses"] = []
    target = site_dir / "data"
    target.mkdir(parents=True, exist_ok=True)
    for year in sorted({d[:4] for d in data["nights"]}):
        indices = [i for i, d in enumerate(data["nights"]) if d.startswith(year)]
        a, b = indices[0], indices[-1] + 1
        rows, pass_rows = [], []
        for row in sorted(data["sites"], key=lambda row: row[0]):
            st = row[4]
            # An archive-only site has no rows in other calendar years.
            if set(st[a:b]) == {"N"}:
                continue
            before = sum(c in "FD" for c in st[:a])
            count = sum(c in "FD" for c in st[a:b])
            lit = count > 0
            rows.append(
                [
                    row[0],
                    st[a:b],
                    row[5][before : before + count] if lit else [],
                ]
            )
            if row[14]:
                pass_rows.append([row[0], row[14][a:b]])
        # A date, rather than a global offset, keeps this year's hash stable
        # when older years are added ahead of it during an archive backfill.
        chunk = {"start": data["nights"][a], "nights": b - a, "sites": rows}
        blob = json.dumps(chunk, separators=(",", ":"), allow_nan=False).encode()
        name = f"nights-{year}.{hashlib.sha256(blob).hexdigest()[:8]}.json"
        (target / name).write_bytes(blob)
        page["meta"]["chunks"].append(f"data/{name}")
        passes = {"start": data["nights"][a], "nights": b - a, "sites": pass_rows}
        blob = json.dumps(passes, separators=(",", ":"), allow_nan=False).encode()
        name = f"clearpasses-{year}.{hashlib.sha256(blob).hexdigest()[:8]}.json"
        (target / name).write_bytes(blob)
        page["meta"]["clearpasses"].append(f"data/{name}")
    for row in page["sites"]:
        row[4], row[5], row[14] = None, None, any(c in "FD" for c in row[4])
    page["totals"] = bin_totals(data)
    return page


def window_flareouts(data: dict) -> dict:
    """Window the page copy only; retain full evidence on disk."""
    result = copy.deepcopy(data)
    for card in result["candidates"]:
        full = card["nightly_series"]
        groups: dict[str, list] = {}
        for row in full:
            groups.setdefault(row["night"][:7], []).append(row)
        monthly = []
        for month, rows in sorted(groups.items()):
            values = [
                r["heat_mw"] if r["heat_mw"] is not None else 0
                for r in rows
                if r["heat_mw"] is not None or (r["qualified"] and not r["lit"])
            ]
            monthly.append(
                [
                    month,
                    _r(np.mean(values)) if values else None,
                    sum(r["qualified"] and r["lit"] for r in rows),
                    sum(r["qualified"] for r in rows),
                ]
            )
        card["record_monthly"] = monthly
        lo = (pd.Timestamp(card["start"]) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        hi = (pd.Timestamp(card["end"]) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        card["nightly_series"] = [r for r in full if lo <= r["night"] <= hi]
        for neighbour in card["context"]["neighbours"]:
            if neighbour.get("series"):
                neighbour["series"] = [
                    r for r in neighbour["series"] if lo <= r["night"] <= hi
                ]
    return result


# ---------------------------------------------------------------- data
def build_data(
    nights: pd.DataFrame, over: pd.DataFrame, out_dir: Path = OUT_DIR
) -> tuple[dict, dict[int, list]]:
    """Page data and per-site pass rows for sites that flared at least once."""
    per_granule = over.groupby(["satellite", "granule"])["bytes_granule"].max()
    sites = pd.read_csv(out_dir / "gfw_oil_sites.csv").set_index("site_id")
    night_list = night_series(nights)
    nidx = {n: i for i, n in enumerate(night_list)}

    myc = validate_myc25.match_sites(
        nights, validate_myc25.load_myc(2025), 2000.0, over
    )
    validation = validate_myc25.validation_summary(myc, 2000.0)
    myc = myc.set_index("site_id")

    # Passes counted in the nightly values: solar zenith >= 97, granule-edge
    # duplicates collapsed, clear by the same rule as the aggregation.
    obs = dedupe_overpasses(over[over["sza"] >= 97.0]).merge(
        sites[["lon"]], left_on="site_id", right_index=True
    )
    t = pd.to_datetime(obs["time_utc"])
    local = t + pd.to_timedelta(obs["lon"] / 15, "h")
    obs["night"] = (local - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    obs = obs[obs["night"].isin(nidx)].copy()
    obs["clear"] = strict_clear(obs)
    t0 = pd.Timestamp(night_list[0])
    minutes = (pd.to_datetime(obs["time_utc"]) - t0).dt.total_seconds() / 60
    obs["tmin"] = minutes.round().astype(int)
    flaring_ids = set(
        nights.loc[nights["status"].isin(["flaring", "flaring_cloud"]), "site_id"]
    )
    passes = {}
    for sid, g in obs[obs["site_id"].isin(flaring_ids)].groupby("site_id"):
        passes[int(sid)] = [
            [
                int(r.tmin),
                SATELLITES.index(r.satellite),
                _r(r.rh_mw, 3) if r.detected else 0,
                int(bool(r.detected)) | (int(bool(r.clear)) << 1),
                _r(r.temp_k, 0) if r.detected else None,
            ]
            for r in g.sort_values("tmin").itertuples()
        ]

    rows = []
    for sid, g in nights.groupby("site_id"):
        st, rv, cp = encode_site_nights(g, night_list)
        vals = g["rh_mw"].dropna()
        s = sites.loc[sid]
        m = myc.loc[sid]
        has_myc = isinstance(m["myc_flare_id"], str)
        rows.append(
            [
                int(sid),
                round(float(s.lat), 4),
                round(float(s.lon), 4),
                int(s.n_structures),
                st,
                rv,
                _r(g["temp_k"].max(), 0),
                _r(vals.mean()) if len(vals) else None,
                region_of(float(s.lat), float(s.lon)),
                m["myc_regime"] if has_myc else None,
                _r(float(m["myc_bcm"]), 4) if has_myc else None,
                m["myc_country_iso"] if has_myc else None,
                m["myc_upstream_class"] if has_myc else None,
                str(s.structure_ids).split()[:12],
                cp,
            ]
        )

    # Flag satellites whose passes stop or start well inside the period.
    notes = []
    period_end = pd.Timestamp(night_list[-1]) + pd.Timedelta(days=1, hours=12)
    period_start = pd.Timestamp(night_list[0]) + pd.Timedelta(hours=12)
    for sat in SATELLITES:
        ts = pd.to_datetime(obs.loc[obs["satellite"] == sat, "time_utc"])
        if ts.empty:
            notes.append(f"{sat} returned no passes in this period.")
            continue
        if ts.max() < period_end - pd.Timedelta(days=1):
            notes.append(f"{sat} has no passes after {ts.max():%-d %b %Y %H:%M} UTC.")
        if ts.min() > period_start + pd.Timedelta(days=1):
            notes.append(f"{sat} has no passes before {ts.min():%-d %b %Y %H:%M} UTC.")

    data = {
        "meta": {
            "start": night_list[0],
            "end": night_list[-1],
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "n_sites": int(len(sites)),
            "n_structures": int(sites["n_structures"].sum()),
            "granules": int(len(per_granule)),
            "gb": round(float(per_granule.sum()) / 1e9, 1),
            "overpasses": int(len(obs)),
            "by_satellite": {s: int((obs["satellite"] == s).sum()) for s in SATELLITES},
            "bcm_per_mw_year": BCM_PER_MW_YEAR,
            "notes": " ".join(notes),
            "status": STATUS,
            "status_label": STATUS_LABEL,
            "validation": validation,
        },
        "nights": night_list,
        "satellites": SATELLITES,
        "regions": [[n, x0, y0, x1, y1] for n, x0, y0, x1, y1 in REGIONS],
        "sites": rows,
    }
    return data, passes


# ---------------------------------------------------------------- explainer
def explainer_html(path: Path = EXPLAINER, img_dir: Path | None = None) -> str:
    """Render the explainer Markdown with its figures.

    SVG figures are inlined. PNG figures are copied to ``img_dir`` and linked
    (or inlined as base64 when ``img_dir`` is None). Italic "Figure n."
    paragraphs become captions, tables get a scroll wrapper, and links to
    other repository docs point at GitHub.
    """
    html = markdown.markdown(
        path.read_text(), extensions=["tables", "sane_lists", "toc"]
    )

    def figure(m: re.Match) -> str:
        src = path.parent / m.group(2)
        if src.suffix == ".svg":
            return f"<figure>{src.read_text().strip()}</figure>"
        if img_dir is None:
            url = "data:image/png;base64," + base64.b64encode(src.read_bytes()).decode()
        else:
            img_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, img_dir / src.name)
            url = f"{img_dir.name}/{src.name}"
        # PNG header size, so lazy images reserve their space and section
        # links land where they point.
        w, h = struct.unpack(">II", src.read_bytes()[16:24])
        return (
            f'<figure class="photo"><img alt="{m.group(1)}" width="{w}" '
            f'height="{h}" loading="lazy" decoding="async" src="{url}"></figure>'
        )

    html = re.sub(
        r'<p><img alt="([^"]*)" src="(explainer/[^"]+\.(?:svg|png))" ?/?></p>',
        figure,
        html,
    )
    html = html.replace("<p><em>Figure", '<p class="cap"><em>Figure')
    html = html.replace("<table>", '<div class="tbl-wrap"><table>').replace(
        "</table>", "</table></div>"
    )
    html = re.sub(
        r'href="(?!https?:|#)([^"]+)"',
        lambda m: f'href="{REPO_BLOB}{m.group(1)}" target="_blank" rel="noopener"',
        html,
    )
    return html


# ---------------------------------------------------------------- build
def main(
    segments: list[tuple[Path, str]] | None = None,
    site_dir: Path = SITE_DIR,
    data: dict | None = None,
    passes: dict | None = None,
    evidence_dir: Path | None = None,
    skip_emit: bool = False,
    base: Path = OUT_DIR,
    root=None,
) -> Path:
    """Build the website from ``segments`` (default: every ``outputs/v5_*`` run)."""
    copies = None
    if data is None:
        if not segments:
            nights, over, copies = load_tables(base, root)
        else:
            for folder, tag in segments:
                print(f"segment {folder.name}: {tag}")
            nights, over = load_segments(segments)
        data, passes = build_data(nights, over)
    elif evidence_dir is None:
        raise ValueError("prebuilt data requires evidence_dir")
    passes = passes or {}
    print(
        f"{len(data['nights'])} nights, {len(data['sites'])} sites, "
        f"{len(passes)} with pass files"
    )

    if site_dir.is_symlink():
        raise ValueError("build destination must not be a symlink")
    if site_dir.exists():
        if evidence_dir is None and not skip_emit:
            import emit

            # Chips keep working after a longer record moves source centres.
            for name in ("flareouts.json", "darknights.json"):
                try:
                    payload = json.loads((site_dir / name).read_text())
                except (OSError, ValueError):
                    continue
                emit.note_payload_centres(payload)
        shutil.rmtree(site_dir)
    (site_dir / "passes").mkdir(parents=True)
    for sid, rows in passes.items():
        (site_dir / "passes" / f"{sid}.json").write_text(
            json.dumps(rows, separators=(",", ":"))
        )
    import darknights
    import emit
    import flareouts

    if evidence_dir is not None:
        flareout_data = json.loads((evidence_dir / "flareouts.json").read_text())
        darknight_data = json.loads((evidence_dir / "darknights.json").read_text())
        for folder in ("passes", "img/emit"):
            source = evidence_dir / folder
            if source.exists():
                shutil.copytree(source, site_dir / folder, dirs_exist_ok=True)
    else:
        if copies is None:
            copies = load_copies(segments)
        flareout_data = flareouts.screen(nights, over, copies)
        if not skip_emit:
            emit.enrich(flareout_data, site_dir=site_dir)
        darknight_data = darknights.screen(nights, over, copies, cards=flareout_data)
        if not skip_emit:
            darknight_data = darknights.enrich(darknight_data, site_dir=site_dir)
    (site_dir / "flareouts.json").write_text(json.dumps(flareout_data, allow_nan=False))
    (site_dir / "darknights.json").write_text(
        json.dumps(darknight_data, separators=(",", ":"), allow_nan=False)
    )
    print(f"single dark nights: {darknight_data['counts']}")
    counts = {
        tier: sum(c["tier"] == tier for c in flareout_data["candidates"])
        for tier in ("High", "Mid", "Watch")
    }
    print(f"flare-outs: {counts}")
    flareout_blob = json.dumps(
        window_flareouts(flareout_data), separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    page = write_night_chunks(data, site_dir)
    blob = json.dumps(page, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text()
    html = html.replace(
        "__SKYTRUTH_LOGO__", LOGO.read_text().split("-->", 1)[-1].strip()
    )
    html = html.replace("__EXPLAINER_HTML__", explainer_html(img_dir=site_dir / "img"))
    html = html.replace("__FLARING_DATA__", blob)
    html = html.replace("__FLAREOUTS_DATA__", flareout_blob)
    html = html.replace("__BASEMAP_TOPOJSON__", load_basemap().replace("</", "<\\/"))
    index = site_dir / "index.html"
    index.write_text(html)
    total = sum(p.stat().st_size for p in site_dir.rglob("*") if p.is_file())
    print(
        f"wrote {site_dir}: index.html {index.stat().st_size / 1e6:.1f} MB, "
        f"{total / 1e6:.1f} MB in all"
    )
    return site_dir


# ---------------------------------------------------------------- publish
def _git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(HERE), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _previous_chunk_blobs(previous_ref: str) -> dict[Path, str]:
    """Keep exactly the chunks named by the preceding published page."""
    html = _git("show", f"{previous_ref}:index.html")
    match = re.search(
        r'<script type="application/json" id="flaring-data">(.*?)</script>',
        html,
        re.DOTALL,
    )
    page = json.loads(match[1].replace("<\\/", "</")) if match else {}
    meta = page.get("meta", {})
    chunks = meta.get("chunks", []) + meta.get("clearpasses", [])
    result = {}
    for name in chunks:
        if not re.fullmatch(
            r"data/(?:nights|clearpasses)-\d{4}\.[0-9a-f]{8}\.json", name
        ):
            raise ValueError(f"invalid published chunk path: {name}")
        result[Path(name)] = _git("rev-parse", f"{previous_ref}:{name}")
    return result


def _write_tree(site_dir: Path, retained: dict[Path, str] | None = None) -> str:
    """Write ``site_dir`` (plus ``.nojekyll``) into the object store as a tree."""
    files = sorted(p for p in site_dir.rglob("*") if p.is_file())
    paths = "\n".join(map(str, files))
    blobs = _git("hash-object", "-w", "--stdin-paths", stdin=paths).split()
    entries: dict[Path, list[str]] = {Path("."): []}
    combined = dict(retained or {})
    combined.update((p.relative_to(site_dir), b) for p, b in zip(files, blobs))
    for rel, blob in sorted(combined.items()):
        entries.setdefault(rel.parent, []).append(f"100644 blob {blob}\t{rel.name}")
        for parent in rel.parents:
            entries.setdefault(parent, [])
    empty = _git("hash-object", "-w", "--stdin", stdin="")
    entries[Path(".")].append(f"100644 blob {empty}\t.nojekyll")
    # deepest folders first, so each parent can list its subtrees
    for folder in sorted(entries, key=lambda p: -len(p.parts)):
        if folder == Path("."):
            continue
        tree = _git("mktree", stdin="\n".join(entries[folder]) + "\n")
        entries[folder.parent].append(f"040000 tree {tree}\t{folder.name}")
    return _git("mktree", stdin="\n".join(entries[Path(".")]) + "\n")


def publish_pages(
    site_dir: Path = SITE_DIR,
    branch: str = "gh-pages",
    remote: str = "origin",
) -> str:
    """Push ``site_dir`` to ``branch`` for GitHub Pages.

    Each parentless publish retains the preceding index's referenced chunks
    for cached pages. Older unreferenced data expires after one build.
    The working tree and current branch are untouched.
    """
    refs = _git("ls-remote", "--heads", remote, branch)
    retained = {}
    if refs:
        _git("fetch", "--no-tags", remote, f"refs/heads/{branch}")
        retained = _previous_chunk_blobs("FETCH_HEAD")
    tree = _write_tree(site_dir, retained)
    source = _git("rev-parse", "--short", "HEAD")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = f"Publish flaring viewer from {source} ({stamp})"
    commit = _git("commit-tree", tree, "-m", message)
    _git("push", "--force", remote, f"{commit}:refs/heads/{branch}")
    print(f"pushed {commit[:10]} to {remote}/{branch}; site: {PAGES_URL}")
    return commit


# %%
if __name__ == "__main__":
    main()
