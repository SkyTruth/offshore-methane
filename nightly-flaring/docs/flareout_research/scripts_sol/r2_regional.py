"""Geometry-adjusted Gulf of Mexico descriptions, without weather attribution.

Source identities and observational cloud gaps remain limitations. This is a
retrospective diagnostic, not a fitted storm detector or a holdout test.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0,".")
from regions import region_of
from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")
CACHE=Path("outputs/flareout_research/scripts_opus/cache")


def main():
    p=pd.read_pickle(CACHE/"passes.pkl")
    s=pd.read_csv("outputs/gfw_oil_sites.csv").set_index("site_id")
    s["region"]=[region_of(a,b) for a,b in zip(s.lat,s.lon)]
    st=pd.read_pickle(CACHE/"site_stats.pkl")
    mon=st[(st.nd>=6)&(st.f>=.3)].index
    ids=mon.intersection(s.index[s.region=="Gulf of Mexico"])
    g=p[p.site_id.isin(ids)&p.cloud_mask.isin([0,1])].copy()
    X=pd.get_dummies(g.site_id.astype(str),prefix="site",dtype=float)
    X["zone1"]=(g.zone==1).astype(float)
    X["zone2"]=(g.zone==2).astype(float)
    X["n21"]=(g.satellite=="NOAA-21").astype(float)
    X["snpp"]=(g.satellite=="SNPP").astype(float)
    X["neighbors_lt8"]=(g.cloud_clear_neighbors<8).astype(float)
    model=LogisticRegression(C=100,max_iter=2000,fit_intercept=False).fit(X,g.detected)
    g["expected"]=model.predict_proba(X)[:,1]
    g["resid"]=g.detected-g.expected
    g["v"]=g.expected*(1-g.expected)
    g["subset"]=np.where(g.site_id.map(s.lat)<22,"south_of_22N","north_of_22N")
    # Also use observation-night outcomes so repeated satellite misses cannot
    # masquerade as independent site-night evidence.
    gn=p[p.site_id.isin(ids)].assign(raw=lambda d:d.cloud_mask.isin([0,1])).groupby(["site_id","night"]).agg(
        nraw=("raw","sum"),lit=("detected","max"),detraw=("out",lambda z:(z=="D").sum())).reset_index()
    summaries=[]
    for subset,group in [("all",g),("south_of_22N",g[g.subset=="south_of_22N"]),
                         ("north_of_22N",g[g.subset=="north_of_22N"])]:
        by=group.groupby("night").agg(passes=("detected","size"),sites=("site_id","nunique"),
                                      detected=("detected","sum"),resid=("resid","mean"),v=("v","sum"))
        by["binomial_sd"]=np.sqrt(by.v)/by.passes
        by["seven_night_resid"]=by.resid.rolling(7,min_periods=5).mean()
        by.to_csv(OUT/f"r2_gom_{subset}_nightly.csv")
        weekly=group.assign(week=pd.to_datetime(group.night).dt.to_period("W").astype(str)).groupby("week").agg(
            passes=("detected","size"),detections=("detected","sum"),residual=("resid","mean"))
        weekly["rate"]=weekly.detections/weekly.passes
        weekly.to_csv(OUT/f"r2_gom_{subset}_weekly.csv")
        summaries.append(dict(subset=subset,sites=group.site_id.nunique(),passes=len(group),
                              detections=int(group.detected.sum()),
                              minimum_seven_night_resid=float(by.seven_night_resid.min()),
                              minimum_seven_night_ending=by.seven_night_resid.idxmin(),
                              lowest_week=weekly.rate.idxmin(),lowest_week_rate=float(weekly.rate.min())))
    # Inspect the original paired GoM run starts directly, including any cloudy
    # positive observations that reset the supposed blackout.
    details=[]
    for start,end in [("2026-10-05","2026-10-15"),("2026-10-29","2026-11-04")]:
        for sid in ids:
            z=gn[(gn.site_id==sid)&gn.night.between(start,end)&(gn.nraw>=1)]
            allz=gn[(gn.site_id==sid)&gn.night.between(start,end)]
            pre=gn[(gn.site_id==sid)&gn.night.between("2026-08-10","2026-09-07")&(gn.nraw>=1)]
            details.append(dict(site_id=int(sid),lat=s.loc[sid,"lat"],lon=s.loc[sid,"lon"],
                                start=start,end=end,n_clear_nights=len(z),n_clear_lit=int(z.lit.sum()),
                                all_nights=len(allz),all_lit_nights=int(allz.lit.sum()),
                                baseline_clear_nights=len(pre),baseline_clear_lit=int(pre.lit.sum())))
    pd.DataFrame(details).to_csv(OUT/"r2_gom_windows.csv",index=False)
    result={"region_monitored_sites":len(ids),"geometry_adjusted_summaries":summaries,
            "co_stop_928_1035_distance_m":float(dist(s.loc[928,"lat"],s.loc[928,"lon"],s.loc[1035,"lat"],s.loc[1035,"lon"]))}
    (OUT/"r2_regional.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    print(pd.DataFrame(details).query('site_id in [927,928,1035]').to_string(index=False),flush=True)


if __name__=="__main__":main()
