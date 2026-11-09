"""Audit how cohort, opportunity definition and pooling affect Markov scores.

None of these exploratory scores is a validated real-world false-alert rate.
Raw cloud 0/1 opportunities, any-weather positive resets, and source apertures
are the same as in r3_checks.py. No new input data are downloaded or generated.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import betaln, betaincinv

from r3_checks import transition_counts

OUT=Path("outputs/flareout_research/scripts_sol")


def main():
    t=pd.read_csv(OUT/"r3_transitions.csv")
    src=pd.read_csv(OUT/"r3_source_transitions.csv")
    p=pd.read_pickle("outputs/flareout_research/scripts_opus/cache/passes.pkl")
    p["raw"]=p.cloud_mask.isin([0,1])
    n=p.groupby(["site_id","night"]).agg(n01=("raw","sum"),lit=("detected","max")).reset_index()
    baseline=n[n.night.between("2026-08-10","2026-09-07")]
    obs=baseline.groupby("site_id").agg(observed=("night","size"),lit=("lit","sum"))
    obs["fraction"]=obs.lit/obs.observed
    pp=p[p.night.between("2026-08-10","2026-09-07")]
    obs["med_rh"]=pp[pp.raw&pp.detected].groupby("site_id").rh_mw.median()
    for q in (1,2):
        z=baseline[baseline.n01>=q].groupby("site_id").agg(n=("lit","size"),s=("lit","sum"))
        obs[f"q{q}_n"],obs[f"q{q}_s"]=z.n,z.s
    obs["eligible_lit_observed"]=(obs.observed>=12)&(obs.fraction>=.8)&(obs.med_rh>=3)
    obs.to_csv(OUT/"r3_observed_eligibility.csv")
    ids=obs.index[obs.eligible_lit_observed]
    (OUT/"r3_observed_eligibility.json").write_text(json.dumps(dict(
        eligible=len(ids),ids=[int(i) for i in ids],scan_family=int(len(ids)*58)),indent=2))
    new=transition_counts(n,ids,"Q_only")
    new.to_csv(OUT/"r3_observed_cohort_transitions.csv",index=False)
    variants=[("21 site Q",t[(t.cohort=="bright21")&(t.convention=="Q_only")],1218),
              ("21 site Q positive resets",t[(t.cohort=="bright21")&(t.convention=="Q_with_positive_resets")],1218),
              ("21 1km-source Q proxy",src,1218),
              (f"{len(ids)} observed-eligible site Q",new,len(ids)*58)]
    records=[]
    for label,z,mult in variants:
        z=z.set_index("site_id")
        total=z[["HH","HM","MH","MM"]].sum()
        ba,bb=int(total.MM+1),int(total.MH+1)
        for sid,k in [(4582,14),(3669,10),(4497,5)]:
            sc=z.loc[sid]
            pool_a=(total.HM+1)/(total.HH+total.HM+2)
            site_a=(sc.HM+1)/(sc.HH+sc.HM+2)
            eb=np.exp(betaln(ba+k-1,bb)-betaln(ba,bb))
            b_hi=betaincinv(ba,bb,.975)
            records.append(dict(variant=label,site_id=sid,k=k,family=mult,
                HH=int(total.HH),HM=int(total.HM),MH=int(total.MH),MM=int(total.MM),
                b_mean=float(ba/(ba+bb)),b_975=float(b_hi),
                pooled_a_integrated_b_score=float(mult*pool_a*eb),
                site_a_integrated_b_score=float(mult*site_a*eb),
                site_a_fixed_b975_stress=float(mult*site_a*b_hi**(k-1))))
    table=pd.DataFrame(records)
    table.to_csv(OUT/"r3_calibration_audit.csv",index=False)
    (OUT/"r3_calibration_audit.json").write_text(json.dumps(records,indent=2))
    print(table.to_string(index=False))


if __name__=="__main__":
    main()
