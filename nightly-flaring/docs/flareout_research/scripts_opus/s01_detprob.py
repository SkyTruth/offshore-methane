"""Per-pass detection probability at steady flares by view zone, satellite, size.

Steady = >= 40 clear passes and clear-pass detection fraction >= 0.5 over 88 nights.
"""
# %%
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes  # noqa: E402


def main():
    p = passes()
    print("passes", len(p), p.out.value_counts().to_dict())
    clear = p[p.out.isin(["D", "Z"])]
    st = clear.groupby("site_id").agg(nc=("out", "size"), nd=("detected", "sum"))
    st["f"] = st.nd / st.nc
    medrh = clear[clear.out == "D"].groupby("site_id").rh_mw.median()
    st["med_rh"] = medrh
    st["n_cloudydet"] = p[p.out == "C"].groupby("site_id").size()
    st = st.fillna({"n_cloudydet": 0})
    st.to_pickle(CACHE / "site_stats.pkl")
    print("sites with clear passes", len(st), "median clear passes",
          st.nc.median(), "sites with >=1 clear det", (st.nd > 0).sum())
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    print(pd.cut(st[st.nd > 0].f, bins, right=False).value_counts().sort_index())
    steady = st[(st.nc >= 40) & (st.f >= 0.5)]
    print("steady sites", len(steady))
    c = clear[clear.site_id.isin(steady.index)].merge(
        steady[["med_rh"]], left_on="site_id", right_index=True)
    c["size"] = pd.cut(c.med_rh, [0, 1, 3, 10, 1e4], labels=["<1", "1-3", "3-10", ">10"])
    print("\nper-pass detection at steady sites, by zone (0 nadir, 2 edge)")
    print(c.groupby("zone").detected.agg(["mean", "size"]))
    print(c.groupby("satellite").detected.agg(["mean", "size"]))
    t = c.groupby(["size", "zone"], observed=True).detected.agg(["mean", "size"]).unstack()
    print(t.round(3))
    print(c.groupby("size", observed=True).site_id.nunique())
    # cloud-neighbour effect among clear passes (thin cloud near the pixel)
    c["cn"] = pd.cut(c.cloud_clear_neighbors, [-1, 3, 5, 7, 8])
    print(c.groupby("cn", observed=True).detected.agg(["mean", "size"]))
    # logistic model: site effect + zone + satellite + clear neighbours
    from sklearn.linear_model import LogisticRegression

    X = pd.get_dummies(c[["site_id"]].astype(str), drop_first=False)
    X["zone1"] = (c.zone == 1).astype(float)
    X["zone2"] = (c.zone == 2).astype(float)
    X["n21"] = (c.satellite == "NOAA-21").astype(float)
    X["snpp"] = (c.satellite == "SNPP").astype(float)
    X["cn_lt8"] = (c.cloud_clear_neighbors < 8).astype(float)
    m = LogisticRegression(C=100, max_iter=2000, fit_intercept=False).fit(X, c.detected)
    coef = pd.Series(m.coef_[0], X.columns)
    print("\nlogit effects (vs nadir, NOAA-20, 8 clear neighbours):")
    print(coef[["zone1", "zone2", "n21", "snpp", "cn_lt8"]].round(3))
    steady.to_pickle(CACHE / "steady.pkl")


main()
