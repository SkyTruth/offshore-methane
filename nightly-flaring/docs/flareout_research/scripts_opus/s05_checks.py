"""Confounder checks on flagged runs: cloud-mask quality inside runs, hot sources
near the site during runs (any assignment), example night strings, sojourn stats."""
# %%
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes, sites  # noqa: E402

CODE = {"flaring": "F", "flaring_cloud": "f", "clear_no_flare": ".", "cloudy": "c",
        "not_observed": " "}


def hav(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def main():
    p = passes()
    s = sites().set_index("site_id")
    runs = pd.read_pickle(CACHE / "runs_flagged.pkl")
    # 1. cloud-mask quality of clear non-detections inside vs outside runs
    z = p[(p.out == "Z") & p.site_id.isin(runs.site_id)].copy()
    inrun = np.zeros(len(z), bool)
    for _, e in runs.iterrows():
        inrun |= ((z.site_id == e.site_id) & (z.t >= e.t0) & (z.t <= e.t1)).to_numpy()
    z["inrun"] = inrun
    print("clear non-detections at flagged sites: in runs", inrun.sum(), "outside", (~inrun).sum())
    print(z.groupby("inrun").agg(prob_clear=("cloud_mask", lambda x: (x == 1).mean()),
                                 nbr_lt8=("cloud_clear_neighbors", lambda x: (x < 8).mean()),
                                 edge=("zone", lambda x: (x == 2).mean()),
                                 snpp=("satellite", lambda x: (x == "SNPP").mean()),
                                 n_hot_m10=("n_hot_m10", lambda x: (x > 0).mean())).round(3))
    # 2. hot sources within 4 km of the site during the run, any assignment
    cp = pd.read_pickle(CACHE / "copies.pkl")
    cp["t"] = pd.to_datetime(cp.time_utc)
    near = []
    for _, e in runs.iterrows():
        la, lo = s.lat[e.site_id], s.lon[e.site_id]
        c = cp[(cp.t >= e.t0) & (cp.t <= e.t1) & (cp.centroid_lat.sub(la).abs() < 0.05)
               & (cp.centroid_lon.sub(lo).abs() < 0.06)]
        d = hav(la, lo, c.centroid_lat, c.centroid_lon)
        near.append(dict(src_lt2km=int((d < 2000).sum()), src_2_4km=int(((d >= 2000) & (d < 4000)).sum()),
                         src_unassigned_lt4=int(((d < 4000) & c.assigned_site.isna()).sum())))
    near = pd.DataFrame(near)
    r = pd.concat([runs.reset_index(drop=True), near], axis=1)
    print("runs with any hot source <2 km during run:", (r.src_lt2km > 0).sum(),
          "; 2-4 km:", (r.src_2_4km > 0).sum(), "; unassigned <4 km:", (r.src_unassigned_lt4 > 0).sum(),
          "of", len(r))
    r.to_pickle(CACHE / "runs_flagged2.pkl")
    # 3. night strings for examples
    n = pd.read_pickle(CACHE / "nights.pkl")
    n["night"] = n.night.astype(str)
    ex = [3662, 3644, 6011, 927, 4069, 4497, 3105, 6779, 4582, 928]
    for sid in ex:
        x = n[n.site_id == sid].sort_values("night")
        st = "".join(CODE[v] for v in x.status)
        rh = x[x.status == "flaring"].rh_mw
        print(f"{sid:5d} {st}  med flaring-night RH {rh.median():.2f} MW (n={len(rh)})")
    print("927-928 distance m:", round(hav(s.lat[927], s.lon[927], s.lat[928], s.lon[928])))
    # 4. sojourn: longest dark span (nights) with >= 10 clear passes, monitored sites
    runs_all = pd.read_pickle(CACHE / "runs.pkl")
    runs_all["nights"] = (pd.to_datetime(runs_all.night1) - pd.to_datetime(runs_all.night0)).dt.days + 1
    st = pd.read_pickle(CACHE / "site_stats.pkl")
    mon = st[(st.nd >= 6) & (st.f >= 0.3)].index
    ra = runs_all[runs_all.site_id.isin(mon)]
    lng = ra[(ra.n_clear >= 10)].groupby("site_id").nights.max().reindex(mon).fillna(0)
    for k in (3, 7, 14, 28):
        print(f"monitored sites with a dark run >= {k} nights and >= 10 clear passes:",
              int((lng >= k).sum()), "of", len(mon))
    x = r[r.ongoing]
    print("ongoing flagged runs:\n", x[["site_id", "region", "night0", "nights", "n_clear", "S",
                                        "base_f", "med_rh_before", "n_neigh", "neigh_lit",
                                        "src_lt2km", "src_2_4km"]].round(2).to_string())


main()
