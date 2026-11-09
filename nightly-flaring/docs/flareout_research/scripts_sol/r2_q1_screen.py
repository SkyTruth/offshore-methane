"""One-clear-pass nightly variant, retaining the full 24-site family size.

This is a pre-baseline eligibility change examined retrospectively; nominal
independent-night probabilities are descriptive, not deployed calibration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaln

OUT=Path("outputs/flareout_research/scripts_sol")
CACHE=Path("outputs/flareout_research/scripts_opus/cache")


def main():
    p=pd.read_pickle(CACHE/"passes.pkl")
    p["raw"]=p.cloud_mask.isin([0,1])
    n=p.groupby(["site_id","night"]).agg(nraw=("raw","sum"),lit=("detected","max")).reset_index()
    a=pd.read_csv(OUT/"r2_eligibility_q1.csv").set_index("site_id")
    records=[]
    for sid,z in a.iterrows():
        h=n[(n.site_id==sid)&(n.night>="2026-09-08")].sort_values("night")
        gap=[]
        def finish(recovery):
            if not gap:return
            g=pd.DataFrame(gap)
            k=int((g.nraw>=1).sum())
            tail=float(np.exp(betaln(z.s+1,z.n-z.s+1+k)-betaln(z.s+1,z.n-z.s+1)))
            records.append(dict(site_id=sid,start=g.night.min(),end=g.night.max(),k=k,
                                n=z.n,s=z.s,tail=tail,nominal_bonf=min(1,tail*len(a)*58),recovery=recovery))
        for row in h.itertuples(index=False):
            if row.lit:
                finish(row.night)
                gap=[]
            else:gap.append(row._asdict())
        finish(None)
    r=pd.DataFrame(records)
    r.to_csv(OUT/"r2_q1_runs.csv",index=False)
    print("Cohort",len(a),"nominal flags",int((r.nominal_bonf<=.05).sum()),flush=True)
    print(r[(r.nominal_bonf<=.05)|(r.site_id==927)].to_string(index=False),flush=True)


if __name__=="__main__":main()
