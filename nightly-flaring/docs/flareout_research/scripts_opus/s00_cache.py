"""Cache combined site_nights, deduped overpasses (night-labelled), copies and
EOG MYC25 match to pickle in scripts_opus/cache/. Run from nightly-flaring/."""
# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
import build_viewer as b  # noqa: E402
import pipeline  # noqa: E402
import validate_myc25 as v  # noqa: E402

OUT = Path("outputs/flareout_research/scripts_opus/cache")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    segs = b.find_segments()
    print(segs)
    nights, over = b.load_segments(segs)
    nights.to_pickle(OUT / "nights.pkl")
    print("nights", nights.shape)
    sites = pd.read_csv("outputs/gfw_oil_sites.csv")
    od = pipeline.dedupe_overpasses(over)
    od = od.merge(sites[["site_id", "lon"]], on="site_id")
    local = pd.to_datetime(od["time_utc"]) + pd.to_timedelta(od["lon"] / 15, "h")
    od["night"] = (local - pd.Timedelta(hours=12)).dt.date.astype(str)
    od["clear"] = pipeline.strict_clear(od)
    keep = [
        "satellite", "granule", "time_utc", "site_id", "row", "col", "dist_m",
        "sza", "cloud_mask", "cloud_clear_neighbors", "detected", "rh_mw",
        "rh_peak_mw", "temp_k", "area_m2", "peak_lat", "peak_lon", "peak_dist_m",
        "fit_flag", "night", "clear", "n_hot_m10", "n_sources",
    ]
    od = od[[c for c in keep if c in od.columns]]
    od.to_pickle(OUT / "over.pkl")
    print("over dedup", od.shape, "raw", over.shape)
    copies = []
    for folder, tag in segs:
        c = pd.read_csv(folder / f"copies_{tag}.csv", low_memory=False)
        copies.append(c)
    copies = pd.concat(copies, ignore_index=True).drop_duplicates(
        ["satellite", "granule", "component_id", "time_utc"], keep="last"
    )
    copies.to_pickle(OUT / "copies.pkl")
    print("copies", copies.shape)
    m = v.match_sites(nights, v.load_myc(2025), 2000.0, over)
    m.to_pickle(OUT / "myc_match.pkl")
    print("myc", m.shape, list(m.columns))


main()
