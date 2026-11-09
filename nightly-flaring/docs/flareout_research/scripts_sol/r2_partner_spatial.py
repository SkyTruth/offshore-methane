"""Recheck Opus's 84 flagged runs against original group-deduped copy CSVs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from r2_evidence import load
from spatial_and_peers import dist

OUT=Path("outputs/flareout_research/scripts_sol")
CACHE=Path("outputs/flareout_research/scripts_opus/cache")


def main():
    _,s,c=load()
    r=pd.read_pickle(CACHE/"runs.pkl")
    r=r[r.S>=np.log(1e4)].copy()
    rows=[]
    for a in r.itertuples(index=False):
        z=c[(c.t>=a.t0)&(c.t<=a.t1)]
        d=dist(s.loc[a.site_id,"lat"],s.loc[a.site_id,"lon"],z.centroid_lat,z.centroid_lon)
        rows.append(dict(site_id=a.site_id,start=a.night0,end=a.night1,
                         groups_lt2km=int((d<2000).sum()),groups_2_4km=int(((d>=2000)&(d<4000)).sum()),
                         unassigned_lt4km=int(((d<4000)&z.assigned_site.isna()).sum())))
    t=pd.DataFrame(rows)
    t.to_csv(OUT/"r2_partner_spatial.csv",index=False)
    result={"runs":len(t),"with_group_lt2km":int((t.groups_lt2km>0).sum()),
            "with_group_2_4km":int((t.groups_2_4km>0).sum()),
            "with_unassigned_lt4km":int((t.unassigned_lt4km>0).sum()),
            "note":"Original copy CSVs; groups deduped by satellite/granule/group_id; no angular bounding-box truncation"}
    (OUT/"r2_partner_spatial.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
