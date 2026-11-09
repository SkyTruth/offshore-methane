"""Night-level serial-dependence sensitivity for Sol's frozen 21-anchor screen.

The Markov stress nulls preserve each site's posterior mean miss fraction;
pooled persistence is a sensitivity assumption, not an identified benign null.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.stats import beta

from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")
CACHE=Path("outputs/flareout_research/scripts_opus/cache")


def main():
    p=pd.read_pickle(CACHE/"passes.pkl")
    p["raw"]=p.cloud_mask.isin([0,1])
    n=p.groupby(["site_id","night"]).agg(nraw=("raw","sum"),lit=("detected","max")).reset_index()
    anchors=pd.read_csv(OUT/"anchors.csv").set_index("site_id")
    histories=[]
    for sid in anchors.index:
        z=n[(n.site_id==sid)&n.night.between("2026-08-10","2026-09-07")&(n.nraw>=2)]
        histories.append(z.sort_values("night").lit.astype(int).to_numpy())
    observed=sum(((z[:-1]==0)&(z[1:]==0)).sum() for z in histories)
    expected=sum((z==0).sum()*((z==0).sum()-1)/len(z) for z in histories)
    rng=np.random.default_rng(54619)
    permutations=20000
    null=np.zeros(permutations,int)
    for z in histories:
        zz=np.array([rng.permutation(z) for _ in range(permutations)])
        null+=((zz[:,:-1]==0)&(zz[:,1:]==0)).sum(axis=1)
    result={"baseline_q_nights":sum(map(len,histories)),
            "baseline_q_misses":int(sum((z==0).sum() for z in histories)),
            "observed_adjacent_qualified_misses":int(observed),
            "within_site_permutation_expected":float(expected),"permutations":permutations,
            "permutation_ge_observed":int((null>=observed).sum()),
            "permutation_p_plus_one":float(((null>=observed).sum()+1)/(permutations+1)),
            "pooled_r_beta8_13_interval95":beta.ppf([.025,.975],8,13).tolist(),"stress_nulls":{}}
    repetitions=20000
    for label,rho in [("independent",None),("r_0.15",.15),("r_7of19",7/19),("r_0.60",.60)]:
        rng=np.random.default_rng(919067)
        events=np.zeros(repetitions,int)
        for sid,a in anchors.iterrows():
            q=n[(n.site_id==sid)&(n.night>="2026-09-08")].sort_values("night").nraw.to_numpy()>=2
            pdetection=rng.beta(a.q2_s+1,a.q2_n-a.q2_s+1,size=repetitions)
            pmiss=1-pdetection
            persist=pmiss if rho is None else np.full(repetitions,rho)
            switch=pmiss*(1-persist)/(1-pmiss)
            prev_miss=rng.random(repetitions)<pmiss
            run=np.zeros(repetitions,int)
            flagged=np.zeros(repetitions,bool)
            for j,valid in enumerate(q[q]):
                miss=prev_miss if j==0 else rng.random(repetitions)<np.where(prev_miss,persist,switch)
                run=np.where(miss,run+1,0)
                tail=np.exp(betaln(a.q2_s+1,a.q2_n-a.q2_s+1+run)-betaln(a.q2_s+1,a.q2_n-a.q2_s+1))
                hit=tail<=.05/(21*58)
                events+=hit&~flagged
                flagged=np.where(miss,flagged|hit,False)
                prev_miss=miss
        result["stress_nulls"][label]={"replicates":repetitions,"total_events":int(events.sum()),
            "mean_alert_events":float(events.mean()),"familywise_alert_fraction":float((events>0).mean()),
            "mc_se_mean":float(events.std()/np.sqrt(repetitions))}
    # An indicative tail using the pooled q-sequence transition rate.
    rows=[]
    runs=pd.read_csv(OUT/"runs.csv")
    for r in runs[runs.bonf<=.05].itertuples(index=False):
        a=anchors.loc[r.site_id]
        pi=(a.q2_n-a.q2_s+1)/(a.q2_n+2)
        mt=pi*(7/19)**(r.k-1)
        rows.append(dict(site_id=r.site_id,start=r.start,k=r.k,independent_bonf=r.bonf,
                         markov_pooled_tail=mt,markov_pooled_bonf=min(1,mt*21*58)))
    pd.DataFrame(rows).to_csv(OUT/"r2_markov_candidates.csv",index=False)
    # Check whether a fixed-radius source proxy alone removes baseline serial
    # clustering, on the same frozen anchor/night sample. No claim of a fitted
    # entity model: a hard aperture can itself create localization misses.
    copies=pd.concat([pd.read_csv(f"outputs/{folder}/copies_{tag}.csv",low_memory=False)
                      for folder,tag in [("v5_backfill","2026-08-09_2026-10-05"),
                                         ("v5_30n","2026-10-06_2026-11-04")]],ignore_index=True)
    copies=copies.drop_duplicates(["satellite","granule","group_id"]).copy()
    copies["night"]=(pd.to_datetime(copies.time_utc)+pd.to_timedelta(copies.centroid_lon/15-12,"h")).dt.strftime("%Y-%m-%d")
    aperture=[]
    for radius in [750,1000,1600]:
        for sid in anchors.index:
            base=copies[(copies.assigned_site==sid)&copies.night.between("2026-08-10","2026-09-07")]
            la,lo=base.centroid_lat.median(),base.centroid_lon.median()
            close=copies[dist(la,lo,copies.centroid_lat,copies.centroid_lon)<=radius]
            h=n[(n.site_id==sid)&n.night.between("2026-08-10","2026-09-07")&(n.nraw>=2)].sort_values("night")
            y=h.night.isin(close.night).to_numpy()
            m=int((~y).sum())
            aperture.append(dict(radius_m=radius,site_id=sid,n=len(y),misses=m,
                                 n00=int(((~y[:-1])&(~y[1:])).sum()),expected00=m*(m-1)/len(y)))
    ap=pd.DataFrame(aperture)
    ap.to_csv(OUT/"r2_aperture_baseline_dependence.csv",index=False)
    result["fixed_anchor_source_aperture_sensitivity"]=ap.groupby("radius_m")[["n","misses","n00","expected00"]].sum().reset_index().to_dict("records")
    # Reproduce the persistence-tail sensitivity table without changing the
    # frozen independent-night screen's thresholds or cohort.
    sensitivity=runs[runs.bonf<=.05][["site_id","start","k","bonf"]].copy()
    for rho in [.15,7/19,float(beta.ppf(.975,8,13)),.60]:
        vals=[]
        for rr in sensitivity.itertuples(index=False):
            aa=anchors.loc[rr.site_id]
            pi=(aa.q2_n-aa.q2_s+1)/(aa.q2_n+2)
            vals.append(min(1,1218*pi*rho**(rr.k-1)))
        sensitivity[f"markov_{rho:.3f}"]=vals
    sensitivity.to_csv(OUT/"r2_markov_sensitivity.csv",index=False)
    (OUT/"r2_dependence.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
