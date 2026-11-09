"""Flare-centred entities: DBSCAN (750 m) on detection peak positions; how many
entities draw detections assigned to more than one GFW site."""
# %%
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import passes  # noqa: E402


def main():
    p = passes()
    d = p[p.out.isin(["D", "C"]) & p.peak_lat.notna()].copy()
    X = np.radians(d[["peak_lat", "peak_lon"]].to_numpy())
    lab = DBSCAN(eps=750 / 6_371_000, min_samples=5, metric="haversine",
                 algorithm="ball_tree").fit_predict(X)
    d["ent"] = lab
    e = d[d.ent >= 0].groupby("ent").agg(n=("site_id", "size"), nsite=("site_id", "nunique"),
                                         top=("site_id", lambda x: x.value_counts(normalize=True).iloc[0]))
    print("detections", len(d), "in entities", int((lab >= 0).sum()), "entities", len(e),
          "flaring sites", d.site_id.nunique())
    print("entities whose detections are assigned to >1 site:", int((e.nsite > 1).sum()),
          "; with top site share < 0.9:", int((e.top < 0.9).sum()))
    print("sites drawing detections from >1 entity:",
          int((d[d.ent >= 0].groupby("site_id").ent.nunique() > 1).sum()))


main()
