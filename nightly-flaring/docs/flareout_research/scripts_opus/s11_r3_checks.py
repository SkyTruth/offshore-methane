"""Round-3 checks.

(a) What drives clustered qualified-night misses: geometry-stratified
    permutation and same-night co-misses between nearby sites.
(b) Empirical detection curve from same-night pass pairs, and radiant-heat
    upper bounds for dark spells (an identifiable alternative to "off").
(c) 4497 vs the 4582/4583 complex: lit-status association across the record.
"""
# %%
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import passes, sites  # noqa: E402
from s09_r2_checks import BASE0, BASE1, hav  # noqa: E402

SOL = "outputs/flareout_research/scripts_sol"


def nights(p):
    p = p.copy()
    p["c01"] = p.cloud_mask.isin([0, 1])
    p["zc"] = np.where(p.c01, p.zone, 9)
    n = p.groupby(["site_id", "night"]).agg(
        n01=("c01", "sum"), lit=("detected", "max"), bestz=("zc", "min")).reset_index()
    n["q"] = n.n01 >= 2
    n["geo"] = n.bestz.astype(str) + np.where(n.n01 >= 3, "+", "")
    return n


def adjacent(n, ids, period, strat, rng, n_perm=2000):
    x = n[n.site_id.isin(ids) & (n.night >= period[0]) & (n.night <= period[1]) & (n.q | n.lit)]
    x = x.sort_values(["site_id", "night"]).copy()
    x["miss"] = x.q & ~x.lit
    obs = 0
    seqs = []
    for sid, g in x.groupby("site_id"):
        m = g.miss.to_numpy()
        obs += int((m[1:] & m[:-1]).sum())
        seqs.append((m, g[strat].to_numpy() if strat else np.zeros(len(m))))
    sims = []
    for _ in range(n_perm):
        c = 0
        for m, s in seqs:
            mm = m.copy()
            for v in np.unique(s):
                k = s == v
                mm[k] = rng.permutation(mm[k])
            c += int((mm[1:] & mm[:-1]).sum())
        sims.append(c)
    sims = np.array(sims)
    return obs, sims.mean(), (sims >= obs).mean(), int(x.miss.sum()), len(x)


def co_miss(n, ids, period, s, lo=3000, hi=50000):
    x = n[n.site_id.isin(ids) & (n.night >= period[0]) & (n.night <= period[1]) & n.q].copy()
    x["miss"] = ~x.lit
    piv = x.pivot(index="night", columns="site_id", values="miss")
    rate = x.groupby("site_id").miss.mean()
    obs = exp = pairs = shared = 0
    ids = [i for i in ids if i in piv.columns]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            d = hav(s.lat[i], s.lon[i], s.lat[j], s.lon[j])
            if not (lo <= d < hi):
                continue
            both = piv[[i, j]].dropna()
            if both.empty:
                continue
            pairs += 1
            shared += len(both)
            obs += int((both[i].astype(bool) & both[j].astype(bool)).sum())
            exp += len(both) * rate[i] * rate[j]
    return pairs, shared, obs, exp


def detection_curve(p):
    p = p[p.cloud_mask.isin([0, 1])]
    g = p.groupby(["site_id", "night"])
    rows = []
    for (sid, nt), x in g:
        if len(x) < 2 or not x.detected.any():
            continue
        for i in np.where(x.detected.to_numpy())[0]:
            rh = x.rh_mw.iloc[i]
            others = x.drop(x.index[i])
            for _, o in others.iterrows():
                rows.append((rh, o.detected, o.zone))
    d = pd.DataFrame(rows, columns=["rh", "det", "zone"])
    d["bin"] = pd.cut(d.rh, [0, 0.2, 0.3, 0.5, 1, 2, 5, 10, 1e4])
    t = d.groupby("bin", observed=True).det.agg(["mean", "size"])
    t["nadir"] = d[d.zone == 0].groupby("bin", observed=True).det.mean()
    t["edge"] = d[d.zone == 2].groupby("bin", observed=True).det.mean()
    return t


def main():
    p = passes()
    s = sites().set_index("site_id")
    n = nights(p)
    anchors = pd.read_csv(f"{SOL}/anchors.csv").site_id.tolist()
    bl = n[(n.night >= BASE0) & (n.night <= BASE1)].groupby("site_id").agg(
        qn=("q", "sum"), qs=("lit", lambda v: 0))
    qs = n[(n.night >= BASE0) & (n.night <= BASE1) & n.q].groupby("site_id").lit.sum()
    bl["qs"] = qs.reindex(bl.index).fillna(0)
    wide = bl[(bl.qn >= 12) & (bl.qs / bl.qn >= 0.8)].index.tolist()
    rng = np.random.default_rng(3)
    print("(a) adjacent qualified-night miss pairs, baseline 08-10..09-07")
    for lab, ids in (("Sol anchors", anchors), (f"wide set ({len(wide)})", wide)):
        for strat in (None, "geo"):
            o, e, pv, nm, nn = adjacent(n, ids, (BASE0, BASE1), strat, rng)
            print(f"   {lab}, permute within site{' x geometry' if strat else ''}: "
                  f"observed {o}, expected {e:.2f}, P(sim>=obs) {pv:.4f}  (misses {nm}, nights {nn})")
    for lab, ids, per in (("wide set, baseline", wide, (BASE0, BASE1)),
                          ("wide set, all nights", wide, ("2026-08-10", "2026-11-04"))):
        pr, sh, o, e = co_miss(n, ids, per, s)
        print(f"   same-night co-misses, site pairs 3-50 km, {lab}: pairs {pr}, shared Q nights {sh}, "
              f"observed {o}, expected under independence {e:.1f}")
    print("(b) detection curve: P(other clear pass same night detects | a pass detected at RH)")
    t = detection_curve(p)
    print(t.round(3))
    for sid, ncl in ((4582, 45), (927, 15), (3662, 148), (3644, 159), (6011, 126), (3669, 34)):
        need = 1 - 0.05 ** (1 / ncl)
        ok = t[t["mean"] >= need]
        bound = ok.index[0].left if len(ok) else np.nan
        print(f"   {sid}: {ncl} clear misses -> 95% bound needs p >= {need:.3f}; "
              f"lowest RH bin with pooled p above it starts at {bound} MW")
    print("(c) 4497 vs 4582/4583 complex: nightly lit on nights both have >=1 cloud-0/1 pass")
    n1 = p.assign(c01=p.cloud_mask.isin([0, 1])).groupby(["site_id", "night"]).agg(
        c=("c01", "sum"), lit=("detected", "max")).reset_index()
    a = n1[n1.site_id == 4497].set_index("night")
    b = n1[n1.site_id.isin([4582, 4583])].groupby("night").agg(c=("c", "max"), lit=("lit", "max"))
    ref = n1[n1.site_id == 4500].set_index("night")
    for per in (("2026-08-10", "2026-09-07"), ("2026-09-08", "2026-11-04")):
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j.c_a > 0) & (j.c_b > 0) & (j.index >= per[0]) & (j.index <= per[1])]
        ct = pd.crosstab(j.lit_a, j.lit_b)
        print(f"   {per}: 4497 lit rows x complex lit cols\n{ct}")
    wk = p[p.detected & p.cloud_mask.isin([0, 1]) & p.site_id.isin([4497, 4582, 4583, 4500])].copy()
    wk["grp"] = wk.site_id.map({4497: "4497", 4582: "cx", 4583: "cx", 4500: "4500"})
    wk["w"] = pd.to_datetime(wk.night).dt.to_period("W")
    print(wk.groupby(["w", "grp"]).rh_mw.median().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
