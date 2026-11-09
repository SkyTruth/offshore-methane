"""Source-identity audit and baseline-only source-centred eligibility proxy.

DBSCAN is diagnostic here. Reject extended/overlapping clusters rather than
letting chains silently combine unrelated flares. Cloud negatives use the
nearest site pixel, an explicitly imperfect surrogate for source-pixel cloud.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.special import betaln
from sklearn.cluster import DBSCAN

sys.path.insert(0,".")
from regions import region_of
from r2_evidence import load
from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")
CACHE=Path("outputs/flareout_research/scripts_opus/cache")


def pair_audit(s,c):
    rows=[]
    base=c[c.night.between("2026-08-10","2026-09-07")].copy()
    for a,b in [(4582,4583),(3656,3669),(928,1035),(4497,4582)]:
        ca=base[base.assigned_site==a]
        cb=base[base.assigned_site==b]
        la,lo=ca.centroid_lat.median(),ca.centroid_lon.median()
        lb,ln=cb.centroid_lat.median(),cb.centroid_lon.median()
        near=base[(dist(la,lo,base.centroid_lat,base.centroid_lon)<=1000)|
                  (dist(lb,ln,base.centroid_lat,base.centroid_lon)<=1000)].copy()
        near=near.sort_values(["satellite","t"])
        gap=near.groupby("satellite").t.diff()
        near["passid"]=(gap.isna()|(gap>=pd.Timedelta(seconds=10))).cumsum()
        # Distinct groups in one pass are evidence of at least two resolved peaks;
        # conversely, no such groups does not prove only one physical flare.
        ng=near.groupby("passid").size()
        bright=near[near.group_rh_mw>=3].groupby("passid").size().reindex(ng.index,fill_value=0)
        if a==4582 and b==4583:
            near[near.passid.isin(ng.index[ng>=2])].to_csv(OUT/"r2_4582_4583_multigroup_passes.csv",index=False)
        rows.append(dict(site_a=a,site_b=b,platform_m=float(dist(s.loc[a,"lat"],s.loc[a,"lon"],s.loc[b,"lat"],s.loc[b,"lon"])),
                         source_center_m=float(dist(la,lo,lb,ln)),a_groups=len(ca),b_groups=len(cb),
                         a_groups_in_b_1km=int((dist(lb,ln,ca.centroid_lat,ca.centroid_lon)<=1000).sum()),
                         b_groups_in_a_1km=int((dist(la,lo,cb.centroid_lat,cb.centroid_lon)<=1000).sum()),
                         union_groups=len(near),passes=len(ng),passes_with_multiple_groups=int((ng>=2).sum()),
                         passes_with_two_groups_ge3MW=int((bright>=2).sum())))
    pd.DataFrame(rows).to_csv(OUT/"r2_source_pairs.csv",index=False)
    return rows


def entity_cohort(p,s,c):
    base=c[c.night.between("2026-08-10","2026-09-07")].copy()
    base["label"]=DBSCAN(eps=750/6371000,min_samples=5,metric="haversine",algorithm="ball_tree").fit_predict(
        np.radians(base[["centroid_lat","centroid_lon"]]))
    centers=[]
    for label,g in base[base.label>=0].groupby("label"):
        la,lo=g.centroid_lat.median(),g.centroid_lon.median()
        pos=np.column_stack([np.radians(g.centroid_lon-lo)*np.cos(np.radians(la))*6371000,
                             np.radians(g.centroid_lat-la)*6371000])
        diameter=pdist(pos).max() if len(g)>1 else 0
        dd=dist(la,lo,s.lat,s.lon)
        sid=int(dd.idxmin())
        clear=g[g.cloud_mask_peak.isin([0,1])]
        centers.append(dict(entity=int(label),lat=la,lon=lo,diameter_m=diameter,groups=len(g),
                            r95_m=float(np.quantile(dist(la,lo,g.centroid_lat,g.centroid_lon),.95)),
                            baseline_nights=g.night.nunique(),
                            assignees=";".join(str(int(i)) for i in sorted(g.assigned_site.dropna().unique())),
                            n_assignees=g.assigned_site.nunique(),nearest_site=sid,nearest_site_m=float(dd.loc[sid]),
                            clear_groups=len(clear),med_rh=float(clear.group_rh_mw.median())))
    centers=pd.DataFrame(centers).set_index("entity")
    centers["nearest_entity_m"]=np.inf
    for k,a in centers.iterrows():
        ds=dist(a.lat,a.lon,centers.lat,centers.lon)
        ds.loc[k]=np.inf
        centers.loc[k,"nearest_entity_m"]=ds.min()
    # Use one predetermined site pixel as an exposure proxy; never choose the
    # clearest of many sites after seeing that pass's outcome.
    p=p.assign(raw=p.cloud_mask.isin([0,1]))
    nn=p.groupby(["site_id","night"]).agg(nraw=("raw","sum")).reset_index()
    complete=pd.DataFrame({"night":pd.date_range("2026-08-09","2026-11-04").strftime("%Y-%m-%d")})
    records=[]
    runs=[]
    for k,a in centers.iterrows():
        in1=c[dist(a.lat,a.lon,c.centroid_lat,c.centroid_lon)<=1000]
        h=complete.merge(nn[nn.site_id==a.nearest_site],on="night",how="left").fillna({"nraw":0})
        h["lit"]=h.night.isin(in1.night)
        baseline=h[h.night.between("2026-08-10","2026-09-07")]
        for nq in [1,2]:
            q=baseline[baseline.nraw>=nq]
            n,success=len(q),int(q.lit.sum())
            usable=a.diameter_m<=2000 and a.nearest_entity_m>=2000 and a.nearest_site_m<=2000
            eligible=usable and n>=12 and success/n>=.8 and a.med_rh>=3
            robust_usable=a.r95_m<=1000 and a.nearest_entity_m>=2000 and a.nearest_site_m<=2000
            robust_eligible=robust_usable and n>=12 and success/n>=.8 and a.med_rh>=3
            records.append(dict(entity=int(k),nearest_site=int(a.nearest_site),quality=nq,n=n,s=success,
                                fraction=success/n if n else None,med_rh=a.med_rh,eligible=eligible,
                                compact=a.diameter_m<=2000,separated=a.nearest_entity_m>=2000,
                                robust_eligible=robust_eligible,r95_m=a.r95_m))
            if eligible or robust_eligible:
                gap=[]
                def finish(recovery):
                    if not gap:return
                    z=pd.DataFrame(gap)
                    misses=int((z.nraw>=nq).sum())
                    tail=float(np.exp(betaln(success+1,n-success+1+misses)-betaln(success+1,n-success+1)))
                    runs.append(dict(entity=int(k),nearest_site=int(a.nearest_site),quality=nq,start=z.night.min(),
                                     end=z.night.max(),k=misses,n=n,s=success,tail=tail,recovery=recovery,
                                     eligible=eligible,robust_eligible=robust_eligible))
                for row in h[h.night>="2026-09-08"].itertuples(index=False):
                    if row.lit:
                        finish(row.night)
                        gap=[]
                    else:gap.append(row._asdict())
                finish(None)
    centers.to_csv(OUT/"r2_entities.csv")
    records=pd.DataFrame(records)
    records.to_csv(OUT/"r2_source_eligibility.csv",index=False)
    # Guards are an exploratory sensitivity grid, not tuned production rules.
    joined=records.merge(centers[["diameter_m","nearest_entity_m","nearest_site_m"]],
                         left_on="entity",right_index=True)
    guards=[]
    for quality in [1,2]:
        for diameter in [2000,3000,4000]:
            for radius in [750,1000,1600]:
                z=joined[(joined.quality==quality)&(joined.n>=12)&(joined.fraction>=.8)&
                         (joined.med_rh>=3)&(joined.diameter_m<=diameter)&(joined.r95_m<=radius)&
                         (joined.nearest_entity_m>=2*radius)&(joined.nearest_site_m<=2000)]
                guards.append(dict(quality=quality,max_diameter_m=diameter,max_r95_m=radius,
                                   eligible=len(z),has927=927 in z.nearest_site.values,
                                   has4582=4582 in z.nearest_site.values))
    pd.DataFrame(guards).to_csv(OUT/"r2_source_guard_sensitivity.csv",index=False)
    runs=pd.DataFrame(runs)
    for nq in [1,2]:
        m=int(records[(records.quality==nq)&records.eligible].shape[0])
        mr=int(records[(records.quality==nq)&records.robust_eligible].shape[0])
        mask=(runs.quality==nq)&runs.eligible
        rmask=(runs.quality==nq)&runs.robust_eligible
        runs.loc[mask,"nominal_bonf"]=(runs.loc[mask,"tail"]*m*58).clip(upper=1)
        runs.loc[rmask,"robust_nominal_bonf"]=(runs.loc[rmask,"tail"]*mr*58).clip(upper=1)
    runs.to_csv(OUT/"r2_source_runs.csv",index=False)
    nominal=runs[runs.nominal_bonf<=.05]
    robust=runs[runs.robust_nominal_bonf<=.05]
    result={"training_groups":len(base),"dbscan_entities":len(centers),
            "dbscan_noise_groups":int((base.label<0).sum()),
            "entities_diameter_gt2km":int((centers.diameter_m>2000).sum()),
            "entities_with_neighbor_lt2km":int((centers.nearest_entity_m<2000).sum()),
            "q1_eligible":int(records[(records.quality==1)&records.eligible].shape[0]),
            "q2_eligible":int(records[(records.quality==2)&records.eligible].shape[0]),
            "q1_robust95_eligible":int(records[(records.quality==1)&records.robust_eligible].shape[0]),
            "q2_robust95_eligible":int(records[(records.quality==2)&records.robust_eligible].shape[0]),
            "q1_nominal_events":int(((runs.quality==1)&(runs.nominal_bonf<=.05)).sum()),
            "q2_nominal_events":int(((runs.quality==2)&(runs.nominal_bonf<=.05)).sum()),
            "all_nominal_events":nominal.astype(object).where(nominal.notna(),None).to_dict("records"),
            "robust95_nominal_events":robust.astype(object).where(robust.notna(),None).to_dict("records")}
    return result


def main():
    p,s,c=load()
    result={"baseline_source_pairs":pair_audit(s,c),"cohort":entity_cohort(p,s,c)}
    (OUT/"r2_entities.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
