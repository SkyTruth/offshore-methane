"""Round-2 checks against Sol's round-1 claims.

(a) within-night dependence of misses while lit (sandwiched nights);
(b) outcome-dependent clearing in strict_clear;
(c) first-order Markov night-level null for Sol's five gaps;
(d) site 927 under Sol's qualified-night rule;
(e) 4582 neighbourhood (10 km) status strings;
(f) flagged-run rates by flare size.
"""
# %%
import sys

import numpy as np
import pandas as pd
from scipy.special import betaln

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes, sites  # noqa: E402

SOL = "outputs/flareout_research/scripts_sol"
BASE0, BASE1 = "2026-08-10", "2026-09-07"


def hav(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def night_table(p):
    p = p.copy()
    p["c01"] = p.cloud_mask.isin([0, 1])
    p["d01"] = p.c01 & p.detected
    n = p.groupby(["site_id", "night"]).agg(n01=("c01", "sum"), d01=("d01", "sum"),
                                           lit=("detected", "max")).reset_index()
    n["q"] = n.n01 >= 2
    return n


def within_night(p, site_ids, label):
    """Nights inside lit spells: previous and next informative night both lit."""
    p = p[p.site_id.isin(site_ids)].copy()
    p["c01"] = p.cloud_mask.isin([0, 1])
    n = night_table(p)
    rate = p[p.c01].groupby("site_id").detected.mean()
    obs = exp = tot = 0
    exp_pass = []
    for sid, g in n.groupby("site_id"):
        g = g[(g.q) | (g.lit)].sort_values("night").reset_index(drop=True)
        lit = g.lit.to_numpy()
        for i in range(1, len(g) - 1):
            if lit[i - 1] and lit[i + 1] and g.q[i]:
                m = int(g.n01[i])
                tot += 1
                obs += int(g.d01[i] == 0)
                exp += (1 - rate[sid]) ** m
    print(f"(a) {label}: sandwiched qualified nights {tot}; all cloud-0/1 passes missed "
          f"observed {obs}, expected under independence {exp:.1f}, ratio {obs / exp:.2f}")


def markov_null(n, site_ids, period, label):
    hh = hm = mh = mm = 0
    for sid, g in n[n.site_id.isin(site_ids)].groupby("site_id"):
        g = g[(g.night >= period[0]) & (g.night <= period[1]) & (g.q | g.lit)].sort_values("night")
        s = np.where(g.lit, "H", "M")
        for a, b in zip(s[:-1], s[1:]):
            hh += (a == "H") & (b == "H")
            hm += (a == "H") & (b == "M")
            mh += (a == "M") & (b == "H")
            mm += (a == "M") & (b == "M")
    p_hm = hm / (hh + hm)
    p_mm = mm / (mh + mm) if (mh + mm) else np.nan
    print(f"(c) {label}: transitions HH {hh} HM {hm} MH {mh} MM {mm}; P(M|H) {p_hm:.3f}, "
          f"P(M|M) {p_mm:.3f} (marginal miss rate {(hm + mm) / (hh + hm + mh + mm):.3f})")
    return p_hm, p_mm, (mh, mm)


def main():
    p = passes()
    s = sites().set_index("site_id")
    anchors = pd.read_csv(f"{SOL}/anchors.csv").site_id.tolist()
    steady = pd.read_pickle(CACHE / "steady.pkl").index.tolist()
    n = night_table(p)
    # (a)
    within_night(p, anchors, "Sol anchors (21)")
    within_night(p, steady, "Opus steady sites (126)")
    # (b)
    d = p[p.detected]
    print("(b) detections counted clear by strict_clear with cloud_mask 2/3:",
          int((d.clear & ~d.cloud_mask.isin([0, 1])).sum()), "of", int(d.clear.sum()),
          "clear detections; at anchors:",
          int((d.clear & ~d.cloud_mask.isin([0, 1]) & d.site_id.isin(anchors)).sum()), "of",
          int((d.clear & d.site_id.isin(anchors)).sum()))
    # (c)
    p_hm_b, p_mm_b, cnt_b = markov_null(n, anchors, (BASE0, BASE1), "anchors, baseline")
    p_hm_a, p_mm_a, cnt_a = markov_null(n, anchors, ("2026-08-10", "2026-11-04"), "anchors, all nights")
    bl = n[(n.night >= BASE0) & (n.night <= BASE1)].groupby("site_id").apply(
        lambda g: pd.Series(dict(qn=(g.q).sum(), qs=(g.q & g.lit).sum())))
    bl["f"] = bl.qs / bl.qn
    wide = bl[(bl.qn >= 12) & (bl.f >= 0.8)].index
    p_hm_w, p_mm_w, cnt_w = markov_null(n, wide, (BASE0, BASE1),
                                        f"all sites with >=12 Q nights, lit >=0.8 in baseline ({len(wide)})")
    gaps = pd.read_csv(f"{SOL}/final_candidate_gates.csv")
    M = 21 * 58
    print("(c) Sol gaps under a first-order Markov null (Beta-posterior P(M|M) from "
          "anchors' baseline transitions), x M=1218:")
    a_, b_ = cnt_b[1] + 1, cnt_b[0] + 1  # Beta(mm+1, mh+1)
    for _, g in gaps.iterrows():
        k = int(g.k)
        # P(M|H) from the site's own baseline; P(M|M)^(k-1) integrated over Beta posterior
        pm = np.exp(betaln(a_ + k - 1, b_) - betaln(a_, b_))
        print(f"   {g.site_id} {g.start} k={k}: P(k-run) = P(M|H) {p_hm_b:.3f} x E[P(M|M)^(k-1)] "
              f"{pm:.2e} -> x M {M * p_hm_b * pm:.2e}")
    # (d) 927
    x = n[n.site_id == 927]
    xb = x[(x.night >= BASE0) & (x.night <= BASE1)]
    print("(d) 927 baseline: Q nights", int(xb.q.sum()), "lit on Q", int((xb.q & xb.lit).sum()),
          "; lit nights", int(xb.lit.sum()), "of", len(xb))
    xs = x[x.night >= "2026-10-30"]
    print("    927 from 10-30: nights", len(xs), "Q nights", int(xs.q.sum()), "lit", int(xs.lit.sum()),
          "cloud-0/1 passes", int(xs.n01.sum()))
    # (e) 4582 neighbourhood
    near = s[hav(s.lat[4582], s.lon[4582], s.lat, s.lon) < 10_000].index
    nn = pd.read_pickle(CACHE / "nights.pkl")
    nn["night"] = nn.night.astype(str)
    w = nn[nn.site_id.isin(near) & (nn.night >= "2026-10-08")]
    act = w.groupby("site_id").status.apply(lambda v: v.isin(["flaring", "flaring_cloud"]).sum())
    code = {"flaring": "F", "flaring_cloud": "f", "clear_no_flare": ".", "cloudy": "c", "not_observed": " "}
    print(f"(e) sites within 10 km of 4582: {len(near)}; with a detection 10-08..11-04: {int((act > 0).sum())}")
    for sid in act[act > 0].index:
        st = "".join(code[v] for v in w[w.site_id == sid].sort_values("night").status)
        dkm = hav(s.lat[4582], s.lon[4582], s.lat[sid], s.lon[sid]) / 1000
        print(f"    {sid:5d} {dkm:4.1f} km  {st}")
    # (f) size classes
    st = pd.read_pickle(CACHE / "site_stats.pkl")
    mon = st[(st.nd >= 6) & (st.f >= 0.3)]
    runs = pd.read_pickle(CACHE / "runs_flagged.pkl")
    mon = mon.assign(size=pd.cut(mon.med_rh, [0, 1, 3, 1e4], labels=["<1", "1-3", ">=3"]),
                     runs=runs.groupby("site_id").size().reindex(mon.index).fillna(0))
    print("(f) flagged runs (P<1e-4) per monitored site by median clear RH:")
    print(mon.groupby("size", observed=True).agg(sites=("runs", "size"), runs=("runs", "sum"),
                                                 sites_flagged=("runs", lambda r: int((r > 0).sum()))))


if __name__ == "__main__":
    main()
