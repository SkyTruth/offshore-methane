"""Space-time co-stops among flagged runs; satellites inside runs; 4497 zones."""
# %%
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes, sites  # noqa: E402
from s05_checks import hav  # noqa: E402


def main():
    p = passes()
    s = sites().set_index("site_id")
    r = pd.read_pickle(CACHE / "runs_flagged2.pkl")
    r["d0"] = pd.to_datetime(r.night0)
    la, lo = s.lat[r.site_id].to_numpy(), s.lon[r.site_id].to_numpy()
    co = []
    for i in range(len(r)):
        d = hav(la[i], lo[i], la, lo)
        dt = (r.d0 - r.d0.iloc[i]).abs().dt.days.to_numpy()
        m = (d < 10_000) & (dt <= 3) & (r.site_id.to_numpy() != r.site_id.iloc[i])
        co.append(r.site_id.to_numpy()[m].tolist())
    r["co_stop"] = co
    print("flagged runs with another site's flagged run starting within 3 nights and 10 km:",
          int((r.co_stop.str.len() > 0).sum()), "of", len(r))
    # null: shift start dates of other runs by random offsets
    rng = np.random.default_rng(0)
    cnt = []
    for _ in range(200):
        d0 = r.d0 + pd.to_timedelta(rng.integers(-40, 41, len(r)), "D")
        c = 0
        for i in range(len(r)):
            d = hav(la[i], lo[i], la, lo)
            dt = (d0 - d0.iloc[i]).abs().dt.days.to_numpy()
            c += bool(((d < 10_000) & (dt <= 3) & (r.site_id.to_numpy() != r.site_id.iloc[i])).any())
        cnt.append(c)
    print("  null (start dates jittered +-40 nights): mean", np.mean(cnt), "95th", np.percentile(cnt, 95))
    print(r[r.co_stop.str.len() > 0][["site_id", "region", "night0", "nights", "co_stop"]].to_string())
    print("3662-3644 m:", round(hav(s.lat[3662], s.lon[3662], s.lat[3644], s.lon[3644])),
          "; 4582-4583 m:", round(hav(s.lat[4582], s.lon[4582], s.lat[4583], s.lon[4583])))
    # satellites inside runs
    z = p[p.out == "Z"]
    nsat = []
    for _, e in r.iterrows():
        x = z[(z.site_id == e.site_id) & (z.t >= e.t0) & (z.t <= e.t1)].satellite.value_counts()
        nsat.append(int((x >= 3).sum()))
    r["n_sat3"] = nsat
    print("runs with >=3 clear non-detections from >=2 satellites:", int((r.n_sat3 >= 2).sum()), "of", len(r))
    # 4497: clear detections by zone before and after 2026-09-08
    g = p[(p.site_id == 4497) & p.out.isin(["D", "Z"])].copy()
    g["late"] = g.night >= "2026-09-08"
    print("4497 detection rate by zone (early / late):")
    print(g.groupby(["late", "zone"]).detected.agg(["mean", "size"]).round(2))
    print("4497 median RH of clear detections early/late:",
          g[g.detected].groupby("late").rh_mw.median().round(2).to_dict())
    # turndown signature among all flagged runs: median RH before vs after
    rr = r.dropna(subset=["first3_rh_after"])
    ratio = rr.first3_rh_after / rr.med_rh_before
    print("closed runs: return/before RH ratio <0.5:", int((ratio < 0.5).sum()), "of", len(rr))
    r.to_pickle(CACHE / "runs_flagged3.pkl")


main()
