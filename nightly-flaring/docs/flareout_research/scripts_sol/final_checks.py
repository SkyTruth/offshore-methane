"""Conservative gates, observed-schedule injection power, and donor summaries."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import betaln

from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")


def main():
    n=pd.read_pickle(OUT/"night_features.pkl")
    o=pd.read_pickle(OUT/"obs.pkl")
    c=pd.read_pickle(OUT/"copy_groups.pkl")
    anchors=pd.read_csv(OUT/"anchors.csv").set_index("site_id")
    sites=pd.read_pickle(OUT/"sites.pkl").set_index("site_id")
    r=pd.read_csv(OUT/"runs.csv")
    out=[]
    for row in r[r.bonf<=.05].itertuples(index=False):
        a=anchors.loc[row.site_id]
        past28=(pd.Timestamp(row.start)-pd.Timedelta(days=28)).strftime("%Y-%m-%d")
        past7=(pd.Timestamp(row.start)-pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        recent=n[(n.site_id==row.site_id)&n.q2&n.night.between(past28,row.start,inclusive="left")]
        rh=o[(o.site_id==row.site_id)&o.cloud01&o.detected&o.night.between(past7,row.start,inclusive="left")]
        nn,ss=len(recent),int(recent.lit.sum())
        tail=np.exp(betaln(ss+1,nn-ss+1+row.k)-betaln(ss+1,nn-ss+1))
        donors=anchors[(anchors.region==a.region)&(anchors.cluster_2km!=a.cluster_2km)]
        dn=n[n.site_id.isin(donors.index)&n.night.between(row.start,row.end)&n.q2].merge(donors[["cluster_2km"]],left_on="site_id",right_index=True)
        dg=dn.groupby(["cluster_2km","night"]).lit.max()
        train=n[n.site_id.isin(donors.index)&n.night.between("2026-08-10","2026-09-07")&n.q2].merge(donors[["cluster_2km"]],left_on="site_id",right_index=True)
        db=train.groupby(["cluster_2km","night"]).lit.max()
        out.append(dict(site_id=row.site_id,start=row.start,end=row.end,k=row.k,
            preceding28_n=nn,preceding28_s=ss,preceding28_tail=tail,preceding28_bonf=min(1,tail*21*58),
            pre7_detected_n=len(rh),pre7_rh_median=rh.rh_mw.median(),
            pre7_bright=bool(len(rh)>=3 and rh.rh_mw.median()>=3),
            donor_clusters=int(donors.cluster_2km.nunique()),donor_group_nights=len(dg),donor_group_lit=int(dg.sum()),
            donor_group_rate=float(dg.mean()),donor_baseline_n=len(db),donor_baseline_lit=int(db.sum()),
            donor_baseline_rate=float(db.mean())))
    out=pd.DataFrame(out)
    out.to_csv(OUT/"final_candidate_gates.csv",index=False)
    print("CANDIDATE GATES",out.to_string(index=False))
    cutoff=.05/(len(anchors)*58)
    inj=[]
    need=[]
    for sid,a in anchors.iterrows():
        k=next(k for k in range(1,89) if np.exp(betaln(a.q2_s+1,a.q2_n-a.q2_s+1+k)-betaln(a.q2_s+1,a.q2_n-a.q2_s+1))<=cutoff)
        need.append(dict(site_id=int(sid),minimum_qnights=k))
        q=n[(n.site_id==sid)&(n.night>="2026-09-08")].sort_values("night").q2.to_numpy().astype(int)
        for length in (4,7,14):
            for start in range(len(q)-length+1):
                # All thermal evidence within the inserted window is erased.
                cumsum=np.cumsum(q[start:start+length])
                detection=np.flatnonzero(cumsum>=k)
                inj.append(dict(site_id=sid,length=length,start_index=start,qnights=int(cumsum[-1]),
                    found=bool(len(detection)),latency=int(detection[0]+1) if len(detection) else None))
    inj=pd.DataFrame(inj)
    power=inj.groupby("length").agg(injections=("found","size"),detected=("found","sum"),power=("found","mean"),median_latency=("latency","median"))
    power.to_csv(OUT/"injection_power.csv")
    pd.DataFrame(need).to_csv(OUT/"required_qnights.csv",index=False)
    print("INJECTION POWER",power.to_string())
    print("REQUIRED QNIGHTS",pd.DataFrame(need).to_string(index=False))
    # Baseline-centred detections during the strong 4582 gap, not just assignees.
    a=anchors.loc[4582]
    base=c[(c.assigned_site==4582)&c.night.between("2026-08-10","2026-09-07")]
    clat,clon=base.centroid_lat.median(),base.centroid_lon.median()
    near=c[c.night.between("2026-10-21","2026-11-04")].copy()
    near["source_dist_m"]=dist(clat,clon,near.centroid_lat,near.centroid_lon)
    near=near[near.source_dist_m<=3000]
    print("4582 NEAR SOURCES",near[["night","satellite","assigned_site","source_dist_m","group_rh_mw","centroid_lat","centroid_lon"]].to_string(index=False))
    near.to_csv(OUT/"4582_nearby_groups.csv",index=False)
    print("4497 PERIODIC LIT RECORD")
    sub=o[(o.site_id==4497)&o.night.between("2026-09-13","2026-10-01")&o.detected]
    print(sub[["night","satellite","col","rh_mw","peak_lat","peak_lon","fit_flag"]].to_string(index=False))
    summary=dict(anchors=len(anchors),anchor_regions=anchors.region.value_counts().to_dict(),
        initial_events=int((r.bonf<=.05).sum()),recent_bright_events=int(out.pre7_bright.sum()),
        injection_power=power.to_dict("index"),minimum_qnights=pd.DataFrame(need).minimum_qnights.value_counts().sort_index().to_dict())
    (OUT/"final_checks.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
