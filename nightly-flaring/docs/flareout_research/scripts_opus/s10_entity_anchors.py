"""Source-centred eligibility (Sol's Q1): DBSCAN entities, chaining check, and
entity-level anchors under Sol's baseline rule; 927 cloudy-pass evidence."""
# %%
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import passes  # noqa: E402
from s09_r2_checks import BASE0, BASE1, hav  # noqa: E402,F401


def main():
    p = passes()
    p["c01"] = p.cloud_mask.isin([0, 1])
    d = p[p.detected & p.peak_lat.notna()].copy()
    lab = DBSCAN(eps=750 / 6_371_000, min_samples=5, metric="haversine",
                 algorithm="ball_tree").fit_predict(np.radians(d[["peak_lat", "peak_lon"]].to_numpy()))
    d["ent"] = lab
    d = d[d.ent >= 0]
    c = d.groupby("ent")[["peak_lat", "peak_lon"]].median()
    d = d.join(c, on="ent", rsuffix="_c")
    d["r"] = hav(d.peak_lat, d.peak_lon, d.peak_lat_c, d.peak_lon_c)
    r95 = d.groupby("ent").r.quantile(0.95)
    print("entities", len(c), "; r95 median", round(r95.median()), "m; r95 > 1500 m:", int((r95 > 1500).sum()))
    members = d.groupby("ent").site_id.unique()
    # entity-night: lit if any entity detection; Q if any member site has >=2 cloud-0/1 passes
    sn = p.groupby(["site_id", "night"]).c01.sum().rename("n01").reset_index()
    rows = []
    base = (sn.night >= BASE0) & (sn.night <= BASE1)
    for e, ms in members.items():
        if r95[e] > 1500:
            continue
        q = sn[base & sn.site_id.isin(ms)].groupby("night").n01.max() >= 2
        lit = set(d[(d.ent == e) & (d.night >= BASE0) & (d.night <= BASE1)].night)
        qn = int(q.sum())
        qs = int(sum(1 for nt, v in q.items() if v and nt in lit))
        rh = d[(d.ent == e) & d.c01 & (d.night >= BASE0) & (d.night <= BASE1)].rh_mw.median()
        rows.append(dict(ent=e, n_sites=len(ms), qn=qn, qs=qs, med_rh=rh))
    t = pd.DataFrame(rows)
    t["f"] = t.qs / t.qn.replace(0, np.nan)
    for rh in (3, 1, 0.5):
        el = t[(t.qn >= 12) & (t.f >= 0.8) & (t.med_rh >= rh)]
        print(f"entity anchors (Q>=12, lit>=0.8, median RH>={rh} MW): {len(el)}; "
              f"drawing on >1 site: {int((el.n_sites > 1).sum())}")
    # 927: cloudy-pass detection rate in baseline, cloudy passes in dark spell
    g = p[p.site_id == 927]
    cb = g[~g.c01 & (g.night >= BASE0) & (g.night <= "2026-10-28")]
    cs = g[~g.c01 & (g.night >= "2026-10-30")]
    print(f"927 cloud-flagged passes 08-10..10-28: {len(cb)}, detected {int(cb.detected.sum())} "
          f"({cb.detected.mean():.2f}); from 10-30: {len(cs)}, detected {int(cs.detected.sum())}")


main()
