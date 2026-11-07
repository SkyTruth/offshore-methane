# %%
"""Build the self-contained flaring map viewer from a pipeline run.

Reads ``outputs/site_nights_<tag>.csv`` and ``outputs/overpasses_<tag>.csv``,
matches sites to EOG's MYC25Q4 annual catalog (``validate_myc25.py``), and
writes one HTML file with the data and a 1:50m country basemap inlined.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import validate_myc25  # noqa: E402
from regions import REGIONS, region_of  # noqa: E402

OUT_DIR = HERE / "outputs"
TEMPLATE = HERE / "viewer_template.html"
BASEMAP_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json"
BCM_PER_MW_YEAR = 0.0116
STATUS_LABEL = "v3 · review fixes"
STATUS = (
    "Pipeline v3: fixes from the 7 Nov 2026 two-model review (GPT-6.1 Sol, Claude "
    "Opus 5.5) applied: finite-radiance site pixels, flare clusters with a glow "
    "test assigned to one site, strict cloud rule, noise-weighted fits with M7/M8 "
    "and QF1 saturation, one-to-one validation. Not yet: M11-only and MWIR-only "
    "triggers, atmospheric correction, cross-satellite calibration, non-upstream "
    "flagging."
)
SATELLITES = ["SNPP", "NOAA-20", "NOAA-21"]
STATUS_CODE = {
    "flaring": "F",
    "flaring_cloud": "D",
    "clear_no_flare": "C",
    "cloudy": "K",
    "not_observed": "N",
}


def load_basemap(out_dir: Path = OUT_DIR) -> str:
    path = out_dir / "basemap_countries-50m.json"
    if not path.exists():
        resp = requests.get(BASEMAP_URL, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path.read_text()


def _r(x: float, nd: int = 3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def build_data(tag: str, out_dir: Path = OUT_DIR) -> dict:
    nights = pd.read_csv(out_dir / f"site_nights_{tag}.csv")
    over = pd.read_csv(out_dir / f"overpasses_{tag}.csv")
    per_granule = over.groupby(["satellite", "granule"])["bytes_granule"].max()
    sites_path = out_dir / "gfw_oil_sites.csv"
    if not sites_path.exists():
        sites_path = OUT_DIR / "gfw_oil_sites.csv"
    sites = pd.read_csv(sites_path).set_index("site_id")
    night_list = sorted(nights["night"].unique())
    nidx = {n: i for i, n in enumerate(night_list)}

    myc = validate_myc25.match_sites(
        nights, validate_myc25.load_myc(2025), 2000.0, over
    )
    validation = validate_myc25.validation_summary(myc, 2000.0)
    myc = myc.set_index("site_id")

    # Per-overpass detail (solar zenith >= 97, as in the nightly aggregation).
    over = over[over["sza"] >= 97.0].merge(
        sites[["lon"]], left_on="site_id", right_index=True
    )
    t = pd.to_datetime(over["time_utc"])
    local = t + pd.to_timedelta(over["lon"] / 15, "h")
    over["night"] = (local - pd.Timedelta(hours=12)).dt.date.astype(str)
    over = over[over["night"].isin(nidx)]
    t0 = pd.Timestamp(night_list[0])
    over["tmin"] = ((t - t0).dt.total_seconds() / 60).round().astype(int)
    flaring_ids = set(
        nights.loc[nights["status"].isin(["flaring", "flaring_cloud"]), "site_id"]
    )
    detail = {}
    for sid, g in over[over["site_id"].isin(flaring_ids)].groupby("site_id"):
        detail[str(int(sid))] = [
            [
                int(r.tmin),
                SATELLITES.index(r.satellite),
                _r(r.rh_mw, 3) if r.detected else 0,
                int(bool(r.detected)),
                int(r.cloud_mask),
                _r(r.temp_k, 0) if r.detected else None,
            ]
            for r in g.sort_values("tmin").itertuples()
        ]

    rows = []
    for sid, g in nights.groupby("site_id"):
        g = g.sort_values("night")
        st = "".join(STATUS_CODE[s] for s in g["status"])
        rh = [_r(v) for v in g["rh_mw"]]
        vals = [v for v in rh if v is not None]
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
                rh,
                _r(g["temp_k"].max(), 0),
                _r(np.mean(vals)) if vals else None,
                region_of(float(s.lat), float(s.lon)),
                m["myc_regime"] if has_myc else None,
                _r(float(m["myc_bcm"]), 4) if has_myc else None,
                m["myc_country_iso"] if has_myc else None,
                m["myc_upstream_class"] if has_myc else None,
                str(s.structure_ids).split()[:12],
            ]
        )

    # Flag satellites whose passes stop or start well inside the period.
    notes = []
    period_end = pd.Timestamp(night_list[-1]) + pd.Timedelta(days=1, hours=12)
    period_start = pd.Timestamp(night_list[0]) + pd.Timedelta(hours=12)
    for sat in SATELLITES:
        ts = pd.to_datetime(over.loc[over["satellite"] == sat, "time_utc"])
        if ts.empty:
            notes.append(f"{sat} returned no passes in this period.")
            continue
        if ts.max() < period_end - pd.Timedelta(days=1):
            notes.append(f"{sat} has no passes after {ts.max():%-d %b %H:%M} UTC.")
        if ts.min() > period_start + pd.Timedelta(days=1):
            notes.append(f"{sat} has no passes before {ts.min():%-d %b %H:%M} UTC.")

    return {
        "meta": {
            "start": night_list[0],
            "end": night_list[-1],
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "n_sites": int(len(sites)),
            "n_structures": int(sites["n_structures"].sum()),
            "granules": int(len(per_granule)),
            "gb": round(float(per_granule.sum()) / 1e9, 1),
            "overpasses": int(len(over)),
            "by_satellite": {
                s: int((over["satellite"] == s).sum()) for s in SATELLITES
            },
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
        "detail": detail,
    }


def latest_run(out_dir: Path = OUT_DIR) -> str:
    """Tag of the run with the latest end date, longest span on ties."""
    tags = [
        p.stem.removeprefix("site_nights_")
        for p in out_dir.glob("site_nights_*_*.csv")
        if (out_dir / p.name.replace("site_nights_", "overpasses_")).exists()
    ]
    if not tags:
        raise FileNotFoundError(f"no site_nights_<start>_<end>.csv in {out_dir}")
    # latest end date first, then the earliest start (longest span)
    return min(tags, key=lambda tg: (-date.fromisoformat(tg[11:]).toordinal(), tg[:10]))


def main(
    start: str | None = None,
    end: str | None = None,
    out_html: Path | None = None,
) -> Path:
    """Build the viewer for ``start``..``end`` (default: the latest, longest run)."""
    tag = f"{start}_{end}" if start and end else latest_run()
    print(f"building viewer for run {tag}")
    data = build_data(tag)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text()
    html = html.replace("__FLARING_DATA__", blob)
    html = html.replace("__BASEMAP_TOPOJSON__", load_basemap().replace("</", "<\\/"))
    out_html = out_html or OUT_DIR / "flaring_viewer.html"
    out_html.write_text(html)
    print(f"wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")
    return out_html


# %%
if __name__ == "__main__":
    main()
