"""(a) regional common-mode detection fraction; (b) repeat runs per site;
(c) synthetic flare-outs injected into steady-site histories."""
# %%
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes, region, sites  # noqa: E402
from s02_runs import SOFF, ZOFF, expit, logit  # noqa: E402

pd.set_option("display.width", 250, "display.max_rows", 200)


def regional(p, mon):
    s = sites().set_index("site_id")
    c = p[p.site_id.isin(mon) & p.out.isin(["D", "Z"])].copy()
    c["region"] = pd.Series(region(s.lat, s.lon), index=s.index).reindex(c.site_id).to_numpy()
    # site-standardised: detection minus site mean, averaged per region-night
    c["resid"] = c.detected - c.groupby("site_id").detected.transform("mean")
    g = c.groupby(["region", "night"]).agg(n=("resid", "size"), ns=("site_id", "nunique"),
                                         r=("resid", "mean"), f=("detected", "mean"))
    g = g[g.ns >= 8]
    # z-score of nightly regional residual vs binomial-ish sd
    g["z"] = g.r / (0.45 / np.sqrt(g.n))
    for reg in ["Persian Gulf", "Gulf of Mexico", "Bohai & Yellow Sea"]:
        x = g.loc[reg]
        w = x.r.rolling(7, center=True, min_periods=4).mean()
        print(reg, "nights", len(x), "sites/night median", x.ns.median(),
              "nightly resid sd", round(x.r.std(), 3), "7-night min", round(w.min(), 3),
              "at", w.idxmin(), "z<-3 nights", (x.z < -3).sum())
        wk = x.assign(week=pd.to_datetime(x.index).to_period("W")).groupby("week").f.mean()
        print("  weekly det fraction:", " ".join(f"{v:.2f}" for v in wk.to_numpy()))
    print("region-nights with z < -3:", (g.z < -3).sum(), "of", len(g),
          "; expected under N(0,1)", round(0.00135 * len(g), 1))
    return g


def inject(p, steady, h_list, n_rep=3, seed=1):
    rng = np.random.default_rng(seed)
    p = p.copy()
    p["off"] = p.zone.map(ZOFF) + p.satellite.map(SOFF)
    res = []
    for sid, g in p[p.site_id.isin(steady.index)].groupby("site_id"):
        out = g.out.to_numpy()
        off = g.off.to_numpy()
        clr = np.isin(out, ["D", "Z"])
        ccum = np.cumsum(clr)
        nights = pd.to_datetime(g.night).to_numpy()
        ok = np.where((ccum >= 40) & (nights <= np.datetime64("2026-10-15")))[0]
        if len(ok) == 0:
            continue
        for _ in range(n_rep):
            i0 = rng.choice(ok)
            base = out[:i0]
            bc = np.isin(base, ["D", "Z"])
            f = min((base == "D").sum() / bc.sum(), 0.98)
            boff = off[:i0][bc].mean()
            S = 0.0
            hit = {h: None for h in h_list}
            for j in range(i0, len(out)):
                if out[j] in ("D", "Z"):  # flare off: every clear pass is a non-detection
                    S += -np.log1p(-expit(logit(f) + off[j] - boff))
                for h in h_list:
                    if hit[h] is None and S >= h:
                        hit[h] = (nights[j] - nights[i0]) / np.timedelta64(1, "D") + 1
            res.append(dict(site_id=sid, med_rh=steady.med_rh[sid], f=f,
                            **{f"d{int(round(np.exp(h)))}": hit[h] for h in h_list}))
    return pd.DataFrame(res)


def main():
    p = passes()
    st = pd.read_pickle(CACHE / "site_stats.pkl")
    steady = pd.read_pickle(CACHE / "steady.pkl")
    mon = st[(st.nd >= 6) & (st.f >= 0.3)].index
    print("monitored sites (>=6 clear det, f>=0.3):", len(mon))
    regional(p, mon)
    runs = pd.read_pickle(CACHE / "runs_flagged.pkl")
    k = runs.groupby("site_id").size()
    print("flagged runs per site:", k.value_counts().sort_index().to_dict())
    print("runs at sites with >=2 flagged runs:", k[k >= 2].sum(), "of", k.sum())
    H = [np.log(1e3), np.log(1e4), np.log(1e5)]
    inj = inject(p, steady, H)
    inj["size"] = pd.cut(inj.med_rh, [0, 1, 3, 1e4], labels=["<1", "1-3", ">3"])
    for c in ["d1000", "d10000", "d100000"]:
        q = inj.groupby("size", observed=True)[c].agg(
            n="size", found=lambda x: x.notna().mean(),
            med="median", p90=lambda x: x.quantile(0.9))
        print(c, "\n", q.round(2))
    inj.to_pickle(CACHE / "inject.pkl")


main()
