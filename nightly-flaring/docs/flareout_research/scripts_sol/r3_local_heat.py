"""Audit the independent 4500 source's local heat change and fit/scan controls."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from r2_evidence import load
from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")


def main():
    _,s,c=load()
    base=c[(c.assigned_site==4500)&c.night.between("2026-08-10","2026-09-07")]
    la,lo=base.centroid_lat.median(),base.centroid_lon.median()
    close=c[dist(la,lo,c.centroid_lat,c.centroid_lon)<=1000].copy()
    close["distance_m"]=dist(la,lo,close.centroid_lat,close.centroid_lon)
    col=close.peak_col
    close["zone"]=np.select([col.between(1008,2191),col.between(640,2559)],["nadir","mid"],"edge")
    windows=[("pre","2026-10-08","2026-10-21"),
             ("dim","2026-10-22","2026-11-01"),
             ("return","2026-11-02","2026-11-04")]
    records=[]
    for label,start,end in windows:
        z=close[close.night.between(start,end)&close.cloud_mask_peak.isin([0,1])].copy()
        for quality in ("all_clear","nadir_neighbors6"):
            q=z if quality=="all_clear" else z[(z.zone=="nadir")&(z.clear_neighbors_peak>=6)]
            for sat in ("ALL","SNPP","NOAA-20","NOAA-21"):
                a=q if sat=="ALL" else q[q.satellite==sat]
                if not len(a):continue
                records.append(dict(phase=label,start=start,end=end,quality=quality,satellite=sat,
                    groups=len(a),nights=a.night.nunique(),median_mw=float(a.group_rh_mw.median()),
                    min_mw=float(a.group_rh_mw.min()),max_mw=float(a.group_rh_mw.max()),
                    median_temperature_k=float(a.temp_k.median()),
                    r95_from_baseline_m=float(a.distance_m.quantile(.95)),
                    median_lat=float(a.centroid_lat.median()),median_lon=float(a.centroid_lon.median()),
                    fit_flags=a.fit_flag.value_counts().to_dict(),
                    band_rule_changed=int(a.band_rule_changed.sum())))
    table=pd.DataFrame(records)
    table.to_csv(OUT/"r3_4500_heat_controls.csv",index=False)
    late=close[close.night.between("2026-10-08","2026-11-04")]
    late.to_csv(OUT/"r3_4500_groups.csv",index=False)
    weekly=[]
    for start in pd.date_range("2026-08-10","2026-11-04",freq="7D"):
        end=min(start+pd.Timedelta(days=6),pd.Timestamp("2026-11-04"))
        z=close[close.night.between(str(start.date()),str(end.date()))]
        clear=z[z.cloud_mask_peak.isin([0,1])]
        weekly.append(dict(start=str(start.date()),end=str(end.date()),groups=len(z),
            nights=z.night.nunique(),clear_groups=len(clear),median_mw=float(clear.group_rh_mw.median())))
    pd.DataFrame(weekly).to_csv(OUT/"r3_4500_weekly.csv",index=False)
    center_distances=[]
    for sid in (4582,4583,4497,4485):
        b=c[(c.assigned_site==sid)&c.night.between("2026-08-10","2026-09-07")]
        center_distances.append(dict(other_site=sid,
            platform_m=float(dist(s.loc[4500,"lat"],s.loc[4500,"lon"],s.loc[sid,"lat"],s.loc[sid,"lon"])),
            source_center_m=float(dist(la,lo,b.centroid_lat.median(),b.centroid_lon.median()))))
    # Source aperture guard: narrow radii must retain the drop and the return.
    radius=[]
    for rr in (500,750,1000,1500):
        z=c[dist(la,lo,c.centroid_lat,c.centroid_lon)<=rr]
        for label,start,end in windows:
            zz=z[z.night.between(start,end)&z.cloud_mask_peak.isin([0,1])]
            radius.append(dict(radius_m=rr,phase=label,groups=len(zz),nights=zz.night.nunique(),
                median_mw=float(zz.group_rh_mw.median())))
    result=dict(source_lat=float(la),source_lon=float(lo),
                controls=records,source_distances=center_distances,
                radius_sensitivity=radius,weekly=weekly)
    (OUT/"r3_local_heat.json").write_text(json.dumps(result,indent=2))
    print(table[["phase","quality","satellite","groups","nights","median_mw", "r95_from_baseline_m"]].to_string(index=False))
    print(json.dumps(center_distances,indent=2))
    print(pd.DataFrame(weekly).to_string(index=False))


if __name__=="__main__":
    main()
