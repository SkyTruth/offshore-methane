"""Summaries of early heat losses and of controls at different spatial scales."""

from pathlib import Path
import json

import pandas as pd

from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")


def main():
    n=pd.read_pickle(OUT/"night_features.pkl")
    c=pd.read_pickle(OUT/"copy_groups.pkl")
    o=pd.read_pickle(OUT/"obs.pkl")
    anchors=pd.read_csv(OUT/"anchors.csv").set_index("site_id")
    sites=pd.read_pickle(OUT/"sites.pkl").set_index("site_id")
    rows=[]
    for sid,split in ((3669,"2026-08-29"),(4497,"2026-09-13"),(4582,"2026-10-21")):
        base=c[(c.assigned_site==sid)&c.night.between("2026-08-10","2026-09-07")]
        clat,clon=base.centroid_lat.median(),base.centroid_lon.median()
        q=c[(dist(clat,clon,c.centroid_lat,c.centroid_lon)<=1000)&c.cloud_mask_peak.isin([0,1])]
        row=dict(site_id=sid,split=split,source_site_m=float(dist(clat,clon,sites.loc[sid,"lat"],sites.loc[sid,"lon"])))
        for label,start,end in (("pre14",(pd.Timestamp(split)-pd.Timedelta(days=14)).strftime("%Y-%m-%d"),(pd.Timestamp(split)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")),("post14",split,(pd.Timestamp(split)+pd.Timedelta(days=13)).strftime("%Y-%m-%d"))):
            z=q[q.night.between(start,end)]
            row[label+"_groups"]=len(z)
            row[label+"_median_mw"]=z.group_rh_mw.median()
            row[label+"_nights"]=z.night.nunique()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"heat_loss_onsets.csv",index=False)
    print("HEAT LOSS ONSETS",pd.DataFrame(rows).to_string(index=False))
    result=[]
    r=pd.read_csv(OUT/"runs.csv")
    for row in r[r.bonf<=.05].itertuples(index=False):
        a=anchors.loc[row.site_id]
        for radius in (50000,100000,250000):
            ds=dist(a.lat,a.lon,anchors.lat,anchors.lon)
            controls=anchors[(ds<=radius)&(anchors.cluster_2km!=a.cluster_2km)]
            z=n[n.site_id.isin(controls.index)&n.q2&n.night.between(row.start,row.end)].merge(controls[["cluster_2km"]],left_on="site_id",right_index=True)
            group=z.groupby(["cluster_2km","night"]).lit.max()
            result.append(dict(site_id=row.site_id,start=row.start,radius_km=radius/1000,
                donor_sites=len(controls),donor_clusters=int(controls.cluster_2km.nunique()),
                n=len(group),lit=int(group.sum()),rate=group.mean()))
    pd.DataFrame(result).to_csv(OUT/"local_donors.csv",index=False)
    print("LOCAL DONORS",pd.DataFrame(result).to_string(index=False))
    z=o[(o.site_id==3669)&o.cloud01&o.detected&o.night.between("2026-08-29","2026-10-22")]
    print("3669 RESIDUAL DETECTIONS",len(z),z.night.nunique(),z.rh_mw.median())


if __name__=="__main__":
    main()
