"""Stricter observation-stratum shuffles, without a serial-dependence model."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,"outputs/flareout_research/scripts_opus")
from s02_runs import H,SOFF,ZOFF,scan_site

OUT=Path("outputs/flareout_research/scripts_sol")
CACHE=Path("outputs/flareout_research/scripts_opus/cache")


def main():
    p=pd.read_pickle(CACHE/"passes.pkl")
    nd=p[p.out=="D"].groupby("site_id").size()
    p=p[p.site_id.isin(nd[nd>=6].index)].copy()
    p["off"]=p.zone.map(ZOFF)+p.satellite.map(SOFF)
    p["cell"]=p.zone.astype(str)+":"+p.satellite+":"+(p.cloud_clear_neighbors<8).astype(str)
    arrays=[]
    for sid,g in p.groupby("site_id",sort=False):
        out=g.out.to_numpy()
        cells=g.cell.to_numpy()
        ix=[np.flatnonzero(np.isin(out,["D","Z"])&(cells==v)) for v in np.unique(cells)]
        arrays.append((out,g.off.to_numpy(),g.night.to_numpy(),ix))
    rng=np.random.default_rng(50712)
    totals=[]
    for rep in range(100):
        counts=np.zeros(3,int)
        for original,off,night,indices in arrays:
            out=original.copy()
            for ix in indices:out[ix]=rng.permutation(out[ix])
            scores=np.array([r[2] for r in scan_site(out,off,night)])
            counts+=np.array([(scores>=h).sum() for h in H])
        totals.append(counts)
    t=np.array(totals)
    result={"shuffle":"site x zone x satellite x (clear_neighbors<8); C/U fixed",
            "replicates":100,"mean_runs":t.mean(axis=0).tolist(),
            "q95_runs":np.quantile(t,.95,axis=0).tolist(),"max_runs":t.max(axis=0).tolist()}
    (OUT/"r2_stationary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
