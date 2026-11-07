# %%
"""Build the self-contained flaring map viewer from a pipeline run.

Reads ``outputs/site_nights_<tag>.csv`` and ``outputs/overpasses_<tag>.csv``,
matches sites to EOG's MYC25Q4 annual catalog (``validate_myc25.py``), and
writes one HTML file with the data and a 1:50m country basemap inlined.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import validate_myc25  # noqa: E402

OUT_DIR = HERE / "outputs"
TEMPLATE = HERE / "viewer_template.html"
BASEMAP_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json"
BCM_PER_MW_YEAR = 0.0116
SATELLITES = ["SNPP", "NOAA-20", "NOAA-21"]
STATUS_CODE = {
    "flaring": "F",
    "clear_no_flare": "C",
    "cloudy": "K",
    "not_observed": "N",
}
# name, lon_min, lat_min, lon_max, lat_max (first match wins)
REGIONS = [
    ("Persian Gulf", 47.0, 23.0, 57.5, 30.6),
    ("Caspian Sea", 46.0, 36.5, 55.0, 47.5),
    ("Red Sea & Suez", 32.0, 12.0, 44.0, 31.0),
    ("Gulf of Mexico", -98.5, 17.0, -80.0, 31.0),
    ("North Sea", -4.5, 50.5, 10.0, 62.5),
    ("Norwegian & Barents Sea", 0.0, 62.5, 45.0, 75.0),
    ("Gulf of Guinea", -4.0, -7.0, 14.0, 7.5),
    ("Angola & Congo", 8.0, -18.0, 14.5, -7.0),
    ("Brazil", -50.0, -30.0, -34.0, 0.0),
    ("Trinidad & Venezuela", -72.0, 8.0, -58.0, 13.5),
    ("Mediterranean", -6.0, 30.0, 36.0, 46.0),
    ("Bohai & Yellow Sea", 117.0, 33.0, 126.0, 41.5),
    ("South China Sea", 105.0, 5.0, 122.0, 23.0),
    ("Gulf of Thailand", 99.0, 5.0, 105.0, 14.0),
    ("Malaysia & Indonesia", 95.0, -9.0, 125.0, 7.5),
    ("India & Bay of Bengal", 66.0, 5.0, 95.0, 25.0),
    ("Australia & Timor Sea", 110.0, -40.0, 155.0, -8.0),
    ("Alaska & Cook Inlet", -170.0, 55.0, -140.0, 72.0),
    ("Sakhalin", 140.0, 45.0, 146.0, 55.0),
]


def region_of(lat: float, lon: float) -> str:
    for name, x0, y0, x1, y1 in REGIONS:
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return name
    return "Other"


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
    sites = pd.read_csv(out_dir / "gfw_oil_sites.csv").set_index("site_id")
    night_list = sorted(nights["night"].unique())
    nidx = {n: i for i, n in enumerate(night_list)}

    myc = validate_myc25.match_sites(nights, validate_myc25.load_myc(2025), 1000.0)
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
    flaring_ids = set(nights.loc[nights["status"] == "flaring", "site_id"])
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

    # Validation summary against MYC25Q4 2025.
    observed = myc[myc["nights_clear"] > 0]
    matched = observed[observed["myc_flare_id"].notna()]
    by_regime = {}
    for regime, g in matched.groupby("myc_regime"):
        by_regime[regime] = [int((g["nights_flaring"] > 0).sum()), int(len(g))]
    both = matched[
        (matched["myc_bcm"].astype(float) > 0) & (matched["bcm_per_year"] > 0)
    ]
    log_r = None
    if len(both) > 2:
        x = np.log10(both["myc_bcm"].astype(float))
        y = np.log10(both["bcm_per_year"])
        log_r = round(float(np.corrcoef(x, y)[0, 1]), 2)
    unmatched_flaring = int(
        ((observed["nights_flaring"] > 0) & observed["myc_flare_id"].isna()).sum()
    )

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
            "validation": {
                "matched": int(len(matched)),
                "observed": int(len(observed)),
                "by_regime": by_regime,
                "log_r": log_r,
                "n_both": int(len(both)),
                "unmatched_flaring": unmatched_flaring,
            },
        },
        "nights": night_list,
        "satellites": SATELLITES,
        "regions": [[n, x0, y0, x1, y1] for n, x0, y0, x1, y1 in REGIONS],
        "sites": rows,
        "detail": detail,
    }


def main(
    start: str = "2026-10-29",
    end: str = "2026-11-04",
    out_html: Path | None = None,
) -> Path:
    tag = f"{start}_{end}"
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
