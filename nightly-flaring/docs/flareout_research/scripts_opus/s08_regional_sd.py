"""Regional nightly residual sd vs binomial expectation (monitored sites)."""
# %%
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes, region, sites  # noqa: E402


def main():
    p = passes()
    s = sites().set_index("site_id")
    st = pd.read_pickle(CACHE / "site_stats.pkl")
    mon = st[(st.nd >= 6) & (st.f >= 0.3)].index
    c = p[p.site_id.isin(mon) & p.out.isin(["D", "Z"])].copy()
    c["region"] = pd.Series(region(s.lat, s.lon), index=s.index).reindex(c.site_id).to_numpy()
    c["pm"] = c.groupby("site_id").detected.transform("mean")
    c["v"] = c.pm * (1 - c.pm)
    c["resid"] = c.detected - c.pm
    g = c.groupby(["region", "night"]).agg(n=("resid", "size"), ns=("site_id", "nunique"),
                                         r=("resid", "mean"), v=("v", "sum"))
    g = g[g.ns >= 8]
    g["sd_exp"] = np.sqrt(g.v) / g.n
    g["z"] = g.r / g.sd_exp
    for reg in ["Persian Gulf", "Gulf of Mexico"]:
        x = g.loc[reg]
        print(reg, "obs sd", round(x.r.std(), 3), "binomial sd (rms)",
              round(np.sqrt((x.sd_exp ** 2).mean()), 3))
    print("z<-3:", (g.z < -3).sum(), "of", len(g), "expected", round(0.00135 * len(g), 1),
          "; z>3:", (g.z > 3).sum())


main()
