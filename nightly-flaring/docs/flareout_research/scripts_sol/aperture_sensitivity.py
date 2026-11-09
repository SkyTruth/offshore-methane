"""Audit source-aperture size and brightness before site-level gaps."""

from pathlib import Path
import json

import pandas as pd

from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")


def main():
    n=pd.read_pickle(OUT/"night_features.pkl")
    o=pd.read_pickle(OUT/"obs.pkl")
    c=pd.read_pickle(OUT/"copy_groups.pkl")
    anchors=pd.read_csv(OUT/"anchors.csv").set_index("site_id")
    runs=pd.read_csv(OUT/"runs.csv")
    rows=[]
    for r in runs[runs.bonf<=.05].itertuples(index=False):
        base=c[(c.assigned_site==r.site_id)&c.night.between("2026-08-10","2026-09-07")]
        clat,clon=base.centroid_lat.median(),base.centroid_lon.median()
        near=c.copy()
        near["dcenter"]=dist(clat,clon,c.centroid_lat,c.centroid_lon)
        for radius in (500,750,1000,1500,2000,3000):
            gap=near[near.night.between(r.start,r.end)&(near.dcenter<=radius)]
            pre7=(pd.Timestamp(r.start)-pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            pre=near[near.night.between(pre7,r.start,inclusive="left")&(near.dcenter<=radius)]
            # Peak cloud rule is kept outcome-independent here (0/1 only).
            pre_clear=pre[pre.cloud_mask_peak.isin([0,1])]
            rows.append(dict(site_id=r.site_id,start=r.start,end=r.end,radius=radius,
                gap_groups=len(gap),gap_lit_nights=gap.night.nunique(),
                gap_rh_median=gap.group_rh_mw.median(),
                pre7_groups=len(pre),pre7_clear_groups=len(pre_clear),
                pre7_clear_median=pre_clear.group_rh_mw.median(),
                pre7_all_median=pre.group_rh_mw.median()))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/"aperture_sensitivity.csv",index=False)
    print(out.to_string(index=False))
    # Exact quality and multisatellite count for the source-level 4582 interval.
    z=o[(o.site_id==4582)&o.night.between("2026-10-21","2026-11-03")&o.cloud01]
    print("4582 14-NIGHT QUALITY",len(z),int(z.high_clear.sum()),int(z.central.sum()),z.groupby("satellite").agg(n=("site_id","size"),nights=("night","nunique"),high_clear=("high_clear","sum")).to_dict("index"))
    sites=pd.read_pickle(OUT/"sites.pkl").set_index("site_id")
    print("4497/4485 site distance",dist(sites.loc[4497,"lat"],sites.loc[4497,"lon"],sites.loc[4485,"lat"],sites.loc[4485,"lon"]))
    # Quantiles of positive clear detections describe observed sizes, not a
    # detection-completeness curve: there are no denominator truths below L.
    print("CLEAR POSITIVE RH QUANTILES",o[o.cloud01&o.detected].rh_mw.quantile([0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).to_dict())
    print("positive below .3 / 1MW",int((o.cloud01&o.detected&(o.rh_mw<.3)).sum()),int((o.cloud01&o.detected&(o.rh_mw<1)).sum()),int((o.cloud01&o.detected).sum()))


if __name__=="__main__":
    main()
