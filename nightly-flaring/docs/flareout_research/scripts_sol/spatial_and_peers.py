"""Check identity ambiguity and peer activity for the frozen-baseline gaps."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT = Path("outputs/flareout_research/scripts_sol")


def dist(lat, lon, lat2, lon2):
    a, b, c, d = map(np.deg2rad, [lat, lon, lat2, lon2])
    return 2*6371000*np.arcsin(np.minimum(1, np.sqrt(np.sin((a-c)/2)**2 + np.cos(a)*np.cos(c)*np.sin((b-d)/2)**2)))


def main():
    sites = pd.read_pickle(OUT / "sites.pkl").set_index("site_id")
    n = pd.read_pickle(OUT / "night_features.pkl")
    o = pd.read_pickle(OUT / "obs.pkl")
    anchors = pd.read_csv(OUT / "anchors.csv").set_index("site_id")
    runs = pd.read_csv(OUT / "runs.csv")
    copies = []
    for segment, tag in (("v5_backfill", "2026-08-09_2026-10-05"), ("v5_30n", "2026-10-06_2026-11-04")):
        c = pd.read_csv(f"outputs/{segment}/copies_{tag}.csv", low_memory=False)
        copies.append(c)
    c = pd.concat(copies, ignore_index=True).drop_duplicates(["satellite", "granule", "group_id"])
    c = c.merge(sites[["lon"]], left_on="assigned_site", right_index=True, how="left", suffixes=("", "_site"))
    # Source-centred local date also covers groups with no assigned GFW site.
    local = pd.to_datetime(c.time_utc) + pd.to_timedelta(c.centroid_lon/15, "h")
    c["night"] = (local - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    c.to_pickle(OUT / "copy_groups.pkl")
    rows = []
    for r in runs[runs.k>=3].itertuples(index=False):
        a = anchors.loc[r.site_id]
        near_dist = dist(a.lat, a.lon, sites.lat, sites.lon)
        nearby = sites[(near_dist<=2000) & (sites.index!=r.site_id)]
        nn = n[n.night.between(r.start, r.end)]
        neighbor = nn[nn.site_id.isin(nearby.index)]
        own = nn[nn.site_id==r.site_id]
        neighbor_lit = neighbor.groupby("night").lit.max()
        qdates = set(own.loc[own.q2, "night"])
        farpeer = anchors[(anchors.region==a.region) & (anchors.cluster_2km!=a.cluster_2km)]
        pn = nn[nn.site_id.isin(farpeer.index) & nn.q2]
        base = c[(c.assigned_site==r.site_id) & c.night.between("2026-08-10", "2026-09-07")]
        clat, clon = base.centroid_lat.median(), base.centroid_lon.median()
        baseline_radius = dist(clat, clon, base.centroid_lat, base.centroid_lon)
        gap_c = c[c.night.between(r.start, r.end)].copy()
        gap_c["dcenter"] = dist(clat, clon, gap_c.centroid_lat, gap_c.centroid_lon)
        close500 = gap_c[gap_c.dcenter<=500]
        close1000 = gap_c[gap_c.dcenter<=1000]
        close2000 = gap_c[gap_c.dcenter<=2000]
        rows.append(dict(site_id=r.site_id, start=r.start, end=r.end, k=r.k, days=r.days, tail=r.tail, bonf=r.bonf,
            neighbor_sites=len(nearby), neighbor_lit_nights=int(neighbor_lit.sum()), neighbor_lit_qnights=int(neighbor_lit.reindex(list(qdates), fill_value=0).sum()),
            closest_sites=";".join(f"{sid}:{near_dist.loc[sid]:.0f}" for sid in nearby.index),
            source_center_lat=clat, source_center_lon=clon, source_base_n=len(base),
            source_base_r50=float(np.median(baseline_radius)), source_base_r95=float(np.quantile(baseline_radius,.95)),
            near500_groups=len(close500), near500_lit_nights=close500.night.nunique(),
            near1000_groups=len(close1000), near1000_lit_nights=close1000.night.nunique(),
            near2000_groups=len(close2000), near2000_lit_nights=close2000.night.nunique(),
            near1000_assignees=";".join(f"{int(sid)}:{cnt}" for sid,cnt in close1000.assigned_site.value_counts().items()),
            peer_sites=len(farpeer), peer_qnights=len(pn), peer_success=int(pn.lit.sum()), peer_rate=float(pn.lit.mean()) if len(pn) else None,
            peer_baseline=float(farpeer.q2_s.sum()/farpeer.q2_n.sum()) if len(farpeer) else None,
        ))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "spatial_runs.csv", index=False)
    print(result.drop(columns=["closest_sites", "near1000_assignees"]).to_string(index=False))
    print("NEARBY ASSIGNEES")
    print(result[["site_id", "start", "closest_sites", "near1000_assignees"]].to_string(index=False))
    print("BASELINE VS LATER ANCHOR RATES")
    ss = n[n.site_id.isin(anchors.index) & n.q2 & (n.night>="2026-09-08")].groupby("site_id").agg(test_n=("lit","size"), test_d=("lit","sum"))
    ss["test_p"] = ss.test_d/ss.test_n
    print(anchors[["q2_n","q2_s","region"]].join(ss).to_string())
    print("4497 AND NEIGHBOURS NIGHT HISTORY")
    near = sites[dist(sites.loc[4497,"lat"], sites.loc[4497,"lon"], sites.lat, sites.lon) <= 2000].index
    pivot = n[(n.site_id.isin(near)) & (n.night>="2026-08-10")].pivot(index="night",columns="site_id", values="lit").astype(int)
    print(pivot.to_string())
    (OUT / "spatial_summary.json").write_text(json.dumps(dict(
        q3_runs=len(result), nominal_runs=int((result.bonf<=.05).sum()),
        q3_runs_with_neighbor_all_q=int((result.neighbor_lit_qnights==result.k).sum()),
        nominal_runs_with_neighbor_all_q=int(((result.bonf<=.05)&(result.neighbor_lit_qnights==result.k)).sum()),
    ), indent=2))


if __name__ == "__main__":
    main()
