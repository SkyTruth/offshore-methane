"""Prepare research-only local tables; run from nightly-flaring/ with its venv.

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    outputs/flareout_research/scripts_sol/prepare.py
"""

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
import build_viewer as b
from pipeline import dedupe_overpasses, strict_clear
from regions import region_of

OUT = Path("outputs/flareout_research/scripts_sol")


def main():
    start = time.time()
    print("Loading named v5 segments", flush=True)
    segments = b.find_segments()
    print([(str(p), t) for p, t in segments], flush=True)
    nights, over = b.load_segments(segments)
    sites = pd.read_csv("outputs/gfw_oil_sites.csv")
    sites["region"] = [region_of(a, o) for a, o in zip(sites.lat, sites.lon)]
    # Chord distance on the unit sphere, converted to great-circle metres.
    lat, lon = np.deg2rad(sites[["lat", "lon"]].to_numpy()).T
    xyz = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    tree = cKDTree(xyz)
    chord, nearest = tree.query(xyz, k=2)
    sites["nearest_m"] = 2 * 6371000 * np.arcsin(chord[:, 1]/2)
    sites["nearest_id"] = sites.site_id.to_numpy()[nearest[:, 1]]
    pairs = tree.query_pairs(2*np.sin(2000/(2*6371000)))
    parent = np.arange(len(sites))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in pairs:
        parent[root(j)] = root(i)
    sites["cluster_2km"] = [int(sites.site_id.iloc[root(i)]) for i in range(len(sites))]
    sites.to_pickle(OUT / "sites.pkl")
    nights.to_pickle(OUT / "nights.pkl")
    print(f"Loaded {len(nights):,} nights, {len(over):,} raw passes in {time.time()-start:.1f}s", flush=True)
    raw_len = len(over)
    over = dedupe_overpasses(over[over.sza >= 97])
    over = over.merge(sites[["site_id", "lon", "region", "cluster_2km"]], on="site_id")
    local = pd.to_datetime(over.time_utc) + pd.to_timedelta(over.lon/15, unit="h")
    over["night"] = (local - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    over = over[over.night.between(nights.night.min(), nights.night.max())].copy()
    over["clear"] = strict_clear(over)
    over["cloud01"] = over.cloud_mask.isin([0, 1])
    over["high_clear"] = over.cloud01 & (over.cloud_clear_neighbors >= 6)
    over["central"] = over.col.between(800, 2399)
    cols = ["site_id", "satellite", "granule", "time_utc", "night", "col", "sza",
            "cloud_mask", "cloud_clear_neighbors", "clear", "cloud01", "high_clear",
            "central", "detected", "rh_mw", "rh_peak_mw", "temp_k", "peak_lat",
            "peak_lon", "peak_dist_m", "peak_col", "fit_flag", "region", "cluster_2km"]
    over[cols].to_pickle(OUT / "obs.pkl")
    summary = {
        "segments": [(str(p), t) for p, t in segments],
        "sites": len(sites), "nights": nights.night.nunique(),
        "site_nights": len(nights), "raw_passes": raw_len,
        "deduped_night_passes": len(over),
        "clear_passes": int(over.clear.sum()),
        "strict_detections": int((over.clear & over.detected).sum()),
        "cloud_detections": int((~over.clear & over.detected).sum()),
        "status_counts": nights.status.value_counts().to_dict(),
        "sites_ever_clear_detected": int(nights.loc[nights.n_detected > 0, "site_id"].nunique()),
        "sites_ever_detected": int(nights.loc[nights.n_detected + nights.n_detected_cloud > 0, "site_id"].nunique()),
        "neighbor_2km_sites": int((sites.nearest_m <= 2000).sum()),
        "clusters_2km": int(sites.cluster_2km.nunique()),
        "cluster_sizes": sites.groupby("cluster_2km").size().value_counts().sort_index().to_dict(),
        "counts_by_satellite": over.groupby("satellite").agg(passes=("site_id", "size"), clear=("clear", "sum"), detections=("detected", "sum")).to_dict("index"),
    }
    (OUT / "preparation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Done in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
