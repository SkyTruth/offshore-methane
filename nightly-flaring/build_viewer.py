# %%
"""Build the flaring map website from one or more pipeline run segments.

A segment is a run folder holding ``site_nights_<start>_<end>.csv`` and
``overpasses_<start>_<end>.csv``. ``find_segments`` collects every segment
under ``outputs/v5_*``; ``main`` joins them into one night series and writes a
static site to ``outputs/site/``:

- ``index.html``: the map, ranking and timeline, with the per-night data, a
  1:50m country basemap and the "How it works" explainer inlined;
- ``passes/<site_id>.json``: every satellite pass at a site that flared,
  fetched when the site is opened, so the page stays small as nights accrue;
- ``img/``: the explainer's image chips.

``publish_pages`` pushes that folder to the repo's GitHub Pages branch.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import markdown
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import validate_myc25  # noqa: E402
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
        for path in folder.glob("site_nights_*_*.csv"):
            tag = path.stem.removeprefix("site_nights_")
            if (folder / f"overpasses_{tag}.csv").exists():
                segments.append((folder, tag))
    if not segments:
        raise FileNotFoundError(f"no run segments in {base}/{pattern}")
    return sorted(segments, key=lambda s: (s[1][:10], s[1][11:], s[0].name))


def load_segments(
    segments: list[tuple[Path, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate segments; where nights overlap the later segment wins."""
    nights, over = [], []
    for i, (folder, tag) in enumerate(segments):
        n = pd.read_csv(folder / f"site_nights_{tag}.csv")
        n["_segment"] = i
        nights.append(n)
        o = pd.read_csv(folder / f"overpasses_{tag}.csv", low_memory=False)
        o["_segment"] = i
        over.append(o)
    nights = (
        pd.concat(nights, ignore_index=True)
        .sort_values("_segment", kind="stable")
        .drop_duplicates(["site_id", "night"], keep="last")
        .drop(columns="_segment")
        .reset_index(drop=True)
    )
    # A granule near a segment boundary is read by both segments.
    over = (
        pd.concat(over, ignore_index=True)
        .sort_values("_segment", kind="stable")
        .drop_duplicates(["satellite", "granule", "site_id", "time_utc"], keep="last")
        .drop(columns="_segment")
        .reset_index(drop=True)
    )
    return nights, over


def night_series(nights: pd.DataFrame) -> list[str]:
    """Every night from the first to the last, including nights with no data."""
    days = pd.date_range(nights["night"].min(), nights["night"].max(), freq="D")
    return [d.strftime("%Y-%m-%d") for d in days]


def encode_site_nights(
    g: pd.DataFrame, night_list: list[str]
) -> tuple[str, list, str]:
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
                cp if int(sid) in flaring_ids else None,
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
            notes.append(f"{sat} has no passes after {ts.max():%-d %b %H:%M} UTC.")
        if ts.min() > period_start + pd.Timedelta(days=1):
            notes.append(f"{sat} has no passes before {ts.min():%-d %b %H:%M} UTC.")

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
    html = markdown.markdown(path.read_text(), extensions=["tables", "sane_lists"])

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
        return (
            f'<figure class="photo"><img alt="{m.group(1)}" loading="lazy" '
            f'decoding="async" src="{url}"></figure>'
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
) -> Path:
    """Build the website from ``segments`` (default: every ``outputs/v5_*`` run)."""
    segments = segments or find_segments()
    for folder, tag in segments:
        print(f"segment {folder.name}: {tag}")
    nights, over = load_segments(segments)
    data, passes = build_data(nights, over)
    print(f"{len(data['nights'])} nights, {len(data['sites'])} sites, "
          f"{len(passes)} with pass files")

    if site_dir.exists():
        shutil.rmtree(site_dir)
    (site_dir / "passes").mkdir(parents=True)
    for sid, rows in passes.items():
        (site_dir / "passes" / f"{sid}.json").write_text(
            json.dumps(rows, separators=(",", ":"))
        )
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text()
    html = html.replace("__SKYTRUTH_LOGO__", LOGO.read_text().split("-->", 1)[-1].strip())
    html = html.replace("__EXPLAINER_HTML__", explainer_html(img_dir=site_dir / "img"))
    html = html.replace("__FLARING_DATA__", blob)
    html = html.replace("__BASEMAP_TOPOJSON__", load_basemap().replace("</", "<\\/"))
    index = site_dir / "index.html"
    index.write_text(html)
    total = sum(p.stat().st_size for p in site_dir.rglob("*") if p.is_file())
    print(f"wrote {site_dir}: index.html {index.stat().st_size / 1e6:.1f} MB, "
          f"{total / 1e6:.1f} MB in all")
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


def _write_tree(site_dir: Path) -> str:
    """Write ``site_dir`` (plus ``.nojekyll``) into the object store as a tree."""
    files = sorted(p for p in site_dir.rglob("*") if p.is_file())
    paths = "\n".join(map(str, files))
    blobs = _git("hash-object", "-w", "--stdin-paths", stdin=paths).split()
    entries: dict[Path, list[str]] = {Path("."): []}
    for path, blob in zip(files, blobs):
        rel = path.relative_to(site_dir)
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

    Each publish is a single parentless commit holding the site and
    ``.nojekyll``, force-pushed, so the branch never accumulates copies of
    the data. The working tree and current branch are untouched.
    """
    tree = _write_tree(site_dir)
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
