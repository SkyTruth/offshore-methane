"""Annotate high-score dark runs with confounder evidence."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes, region, sites  # noqa: E402

R_NEIGH = 2000.0


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def main(h=np.log(1e4)):
    p = passes()
    s = sites().set_index("site_id")
    runs = pd.read_pickle(CACHE / "runs.pkl")
    myc = pd.read_pickle(CACHE / "myc_match.pkl").set_index("site_id")
    r = runs[runs.S >= h].copy()
    tree = cKDTree(xyz(s.lat, s.lon))
    rad = 2 * np.sin(R_NEIGH / 2 / 6_371_000)
    nb = {sid: [s.index[k] for k in tree.query_ball_point(xyz([s.lat[sid]], [s.lon[sid]])[0], rad)
                if s.index[k] != sid] for sid in r.site_id.unique()}
    lit = p[p.out.isin(["D", "C"])]
    litg = {k: g.t.to_numpy() for k, g in lit.groupby("site_id")}
    det = p[p.out == "D"]
    rows = []
    for _, e in r.iterrows():
        sid = e.site_id
        n_nb = len(nb[sid])
        nbl = 0
        for o in nb[sid]:
            t = litg.get(o)
            if t is not None:
                nbl += ((t >= np.datetime64(e.t0)) & (t <= np.datetime64(e.t1))).sum()
        d = det[det.site_id == sid]
        before = d[d.t < e.t0]
        after = d[d.t > e.t1]
        rows.append(dict(
            n_neigh=n_nb, neigh_lit=nbl,
            med_rh_before=before.rh_mw.median(),
            last3_rh=before.rh_mw.tail(3).median(),
            first3_rh_after=after.rh_mw.head(3).median() if len(after) else np.nan,
            n_det_after=len(after),
        ))
    r = pd.concat([r.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    r["nights"] = (pd.to_datetime(r.night1) - pd.to_datetime(r.night0)).dt.days + 1
    r["region"] = region(s.lat[r.site_id].to_numpy(), s.lon[r.site_id].to_numpy())
    r["myc_regime"] = myc.myc_regime.reindex(r.site_id).to_numpy()
    r.to_pickle(CACHE / "runs_flagged.pkl")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(len(r), "runs at", r.site_id.nunique(), "sites; ongoing", r.ongoing.sum())
    print("neighbour within 2 km:", (r.n_neigh > 0).sum(), "; neighbour lit during run:",
          (r.neigh_lit > 0).sum())
    print("nights: median", r.nights.median(), "IQR", r.nights.quantile([.25, .75]).tolist())
    print("clear passes in run: median", r.n_clear.median())
    print(r.region.value_counts())
    print(r.myc_regime.value_counts(dropna=False))
    print("fade ratio last3/median before:", (r.last3_rh / r.med_rh_before).describe())
    print("return ratio first3 after/median before:",
          (r.first3_rh_after / r.med_rh_before).describe())
    cols = ["site_id", "region", "night0", "night1", "nights", "n_clear", "S", "base_f",
            "med_rh_before", "last3_rh", "first3_rh_after", "n_neigh", "neigh_lit",
            "ongoing", "myc_regime"]
    print(r.sort_values("S", ascending=False)[cols].round(2).to_string())
    # calendar clustering: run starts per night by region
    r["start"] = pd.to_datetime(r.night0)
    print(r.groupby(["region", r.start.dt.to_period("W")]).size().loc[lambda x: x >= 3])


main()
