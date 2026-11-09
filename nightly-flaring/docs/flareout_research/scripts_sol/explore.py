"""Local feasibility summaries; cached inputs are written by prepare.py."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import betaln

OUT = Path("outputs/flareout_research/scripts_sol")
TRAIN_START = "2026-08-10"
TRAIN_END = "2026-09-07"
TEST_START = "2026-09-08"


def main():
    n = pd.read_pickle(OUT / "nights.pkl")
    o = pd.read_pickle(OUT / "obs.pkl")
    sites = pd.read_pickle(OUT / "sites.pkl").set_index("site_id")
    o["dclear"] = o.cloud01 & o.detected
    o["dhq"] = o.high_clear & o.detected
    o["dcentral"] = o.cloud01 & o.central & o.detected
    o["qcentral"] = o.cloud01 & o.central
    night = o.groupby(["site_id", "night"]).agg(
        n01=("cloud01", "sum"), nhq=("high_clear", "sum"),
        nc=("qcentral", "sum"), lit=("detected", "max"),
        dc=("dclear", "sum"), dhq=("dhq", "sum"),
        dcentral=("dcentral", "sum"),
    ).reset_index()
    n = n.merge(night, on=["site_id", "night"], how="left")
    for c in night.columns[2:]:
        n[c] = n[c].fillna(0)
    n["lit"] = n.lit.astype(bool)
    n["q2"] = n.n01 >= 2
    n["qhq"] = n.nhq >= 2
    n["qc"] = n.nc >= 1
    train = n[n.night.between(TRAIN_START, TRAIN_END)]
    f = train.groupby("site_id").agg(
        clear_nights=("n_clear", lambda x: int((x>0).sum())),
        lit_nights=("lit", "sum"), q2_n=("q2", "sum"),
        qhq_n=("qhq", "sum"), qc_n=("qc", "sum"),
        clear_passes=("n_clear", "sum"), detections=("n_detected", "sum"),
    )
    for q in ("q2", "qhq", "qc"):
        f[q+"_s"] = train[train[q]].groupby("site_id").lit.sum().reindex(f.index, fill_value=0)
        f[q+"_p"] = f[q+"_s"] / f[q+"_n"].replace(0, np.nan)
    f["med_rh"] = o[o.night.between(TRAIN_START, TRAIN_END) & o.cloud01 & o.detected].groupby("site_id").rh_mw.median()
    f = f.join(sites)
    f.to_pickle(OUT / "features.pkl")
    n.to_pickle(OUT / "night_features.pkl")
    print("OBSERVATION COUNTS", n[["q2", "qhq", "qc"]].sum().to_dict())
    print("BASELINE ELIGIBILITY")
    rows = []
    for q in ("q2", "qhq", "qc"):
        for mn in (10, 12, 15):
            for p in (.7, .8, .9):
                for rh in (1, 3, 5):
                    elig = (f[q+"_n"] >= mn) & (f[q+"_p"] >= p) & (f.med_rh >= rh)
                    rows.append(dict(quality=q, min_n=mn, min_p=p, min_rh=rh, sites=int(elig.sum()), isolated=int((elig & (f.nearest_m > 2000)).sum())))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "eligibility.csv", index=False)
    print(tbl[(tbl.min_n==12) & (tbl.min_rh==3)].to_string(index=False))
    anchors = f[(f.q2_n>=12) & (f.q2_p>=.8) & (f.med_rh>=3)].copy()
    anchors.to_csv(OUT / "anchors.csv")
    print("ANCHORS", len(anchors))
    print(anchors[["q2_n", "q2_s", "q2_p", "med_rh", "nearest_m", "region"]].to_string())
    runs = []
    for sid, g in n[n.site_id.isin(anchors.index)].groupby("site_id"):
        g = g.sort_values("night")
        a = anchors.loc[sid]
        pre = g[g.night < TEST_START]
        last_lit = pre.loc[pre.lit, "night"].max()
        gap = []
        def finish(gap, recovery):
            if not gap:
                return
            z = pd.DataFrame(gap)
            k = int(z.q2.sum())
            k_hq = int(z.qhq.sum())
            k_c = int(z.qc.sum())
            lp = betaln(a.q2_s+1, a.q2_n-a.q2_s+1+k) - betaln(a.q2_s+1, a.q2_n-a.q2_s+1)
            runs.append(dict(site_id=int(sid), start=z.night.min(), end=z.night.max(), before=last_lit, recovery=recovery,
                days=len(z), k=k, k_hq=k_hq, k_c=k_c, passes=int(z.n01.sum()),
                tail=float(np.exp(lp)), n=int(a.q2_n), s=int(a.q2_s), median_rh=float(a.med_rh),
                region=a.region, nearest_m=float(a.nearest_m), cluster_2km=int(a.cluster_2km)))
        for row in g[g.night>=TEST_START].itertuples(index=False):
            if row.lit:
                finish(gap, row.night)
                gap = []
                last_lit = row.night
            else:
                gap.append(row._asdict())
        finish(gap, None)
    r = pd.DataFrame(runs).sort_values(["tail", "k"], ascending=[True, False])
    r["bonf"] = (r["tail"] * len(anchors) * 58).clip(upper=1)
    r.to_csv(OUT / "runs.csv", index=False)
    print("RUNS", len(r), "k>=3", int((r.k>=3).sum()), "k>=5", int((r.k>=5).sum()), "bonf<=.05", int((r.bonf<=.05).sum()))
    print(r.head(35).to_string(index=False))
    print("LARGEST CLUSTERS", sites.groupby("cluster_2km").size().nlargest(10).to_dict())
    print("HIGH-QUALITY CLOUD01 DETECTION RATES BY COL")
    oo = o[o.site_id.isin(anchors.index) & o.cloud01].copy()
    oo["zone"] = pd.cut(oo.col, [-1,399,799,1199,1999,2399,2799,3199])
    print(oo.groupby(["satellite", "zone"], observed=True).agg(n=("detected","size"), d=("detected","sum"), p=("detected","mean"), rh=("rh_mw", lambda x: x[x>0].median())).to_string())
    print("ANCHOR CLEAR-PASS PERSISTENCE", oo.groupby("site_id").detected.mean().describe().to_dict())
    (OUT / "explore_summary.json").write_text(json.dumps(dict(anchors=len(anchors), candidate_runs=len(r), at_least_3=int((r.k>=3).sum()), at_least_5=int((r.k>=5).sum()), nominal_bonf_05=int((r.bonf<=.05).sum())), indent=2))


if __name__ == "__main__":
    main()
