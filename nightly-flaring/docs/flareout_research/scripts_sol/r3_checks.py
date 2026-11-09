"""Final-round checks using only local nightly-flaring inputs.

Run from nightly-flaring with PYTHONDONTWRITEBYTECODE=1 .venv/bin/python.
All output files are research artifacts, not fitted production alert models.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import betaln, betaincinv

from r2_evidence import load
from spatial_and_peers import dist

OUT = Path("outputs/flareout_research/scripts_sol")
BASE0, BASE1, LAST = "2026-08-10", "2026-09-07", "2026-11-04"
RNG_SEED = 319471


def transition_counts(n, ids, convention):
    records = []
    for sid in ids:
        g = n[(n.site_id == sid) & n.night.between(BASE0, BASE1)]
        g = g.sort_values("night")
        counts = np.zeros((2, 2), dtype=int)  # 0 lit, 1 miss
        if convention in ("Q_only", "Q_or_lit"):
            g = g[(g.n01 >= 2) | (g.lit if convention == "Q_or_lit" else False)]
            y = (~g.lit).astype(int).to_numpy()
            for prev, curr in zip(y[:-1], y[1:]):
                counts[prev, curr] += 1
        else:
            # Non-Q positives reset the state but are not transition trials.
            prev = None
            for row in g.itertuples(index=False):
                if row.n01 >= 2:
                    curr = int(not row.lit)
                    if prev is not None:
                        counts[prev, curr] += 1
                    prev = curr
                elif row.lit:
                    prev = 0
        records.append(dict(site_id=int(sid), convention=convention,
                            HH=int(counts[0, 0]), HM=int(counts[0, 1]),
                            MH=int(counts[1, 0]), MM=int(counts[1, 1])))
    return pd.DataFrame(records)


def min_k(a):
    for k in range(1, 101):
        tail = np.exp(betaln(a.q2_s + 1, a.q2_n - a.q2_s + 1 + k)
                      - betaln(a.q2_s + 1, a.q2_n - a.q2_s + 1))
        if tail <= .05 / (21 * 58):
            return k
    raise ValueError("No threshold in 100 opportunities")


def count_alerts(states, required):
    """states[replicate, calendar night, site]: 0 unknown, 1 lit, 2 Q miss."""
    run = np.zeros((len(states), len(required)), dtype=np.int16)
    flagged = np.zeros_like(run, dtype=bool)
    events = np.zeros(len(states), dtype=np.int16)
    for t in range(states.shape[1]):
        lit = states[:, t, :] == 1
        miss = states[:, t, :] == 2
        run = np.where(lit, 0, run + miss)
        flagged &= ~lit
        hit = (run >= required) & ~flagged
        events += hit.sum(axis=1)
        flagged |= hit
    return dict(mean_events=float(events.mean()),
                familywise_fraction=float((events > 0).mean()),
                mc_se_mean=float(events.std() / np.sqrt(len(events))),
                quantile95=float(np.quantile(events, .95)),
                replicates=len(events))


def null_checks(p, c):
    n = p.groupby(["site_id", "night"]).agg(
        n01=("raw", "sum"), lit=("detected", "max")
    ).reset_index()
    anchors = pd.read_csv(OUT / "anchors.csv").set_index("site_id")
    baseline = n[n.night.between(BASE0, BASE1)]
    elig = baseline[baseline.n01 >= 2].groupby("site_id").agg(
        nq=("lit", "size"), s=("lit", "sum"))
    wide = elig[(elig.nq >= 12) & (elig.s / elig.nq >= .8)].index
    tables = []
    result = dict(transition_totals=[], baseline_resampling={}, markov_tails=[])
    for label, ids in [("bright21", anchors.index), ("wide70", wide)]:
        for convention in ("Q_only", "Q_or_lit", "Q_with_positive_resets"):
            t = transition_counts(n, ids, convention)
            t["cohort"] = label
            tables.append(t)
            totals = t[["HH", "HM", "MH", "MM"]].sum()
            result["transition_totals"].append(dict(
                cohort=label, sites=len(ids), convention=convention,
                **{k: int(v) for k, v in totals.items()},
                p_M_given_H=float(totals.HM / (totals.HH + totals.HM)),
                p_MM=float(totals.MM / (totals.MH + totals.MM)),
                b_interval95=betaincinv(totals.MM+1, totals.MH+1,
                                       np.array([.025, .975])).tolist()))
    pd.concat(tables, ignore_index=True).to_csv(OUT / "r3_transitions.csv", index=False)
    duration=[]
    for label,ids in [("bright21",anchors.index),("wide70",wide)]:
        counts={k:[0,0] for k in (1,2,3,4)}
        sample=baseline[baseline.site_id.isin(ids)&(baseline.n01>=2)]
        for _,g in sample.groupby("site_id"):
            run=0
            for miss in (~g.sort_values("night").lit).to_numpy():
                if run:
                    key=min(run,4)
                    counts[key][0]+=int(miss)
                    counts[key][1]+=1
                run=run+1 if miss else 0
        for k,(cont,total) in counts.items():
            duration.append(dict(cohort=label,preceding_qualified_miss_run=k,
                                 continuations=cont,trials=total,
                                 rate=cont/total if total else None))
    result["duration_diagnostic"]=duration
    pd.DataFrame(duration).to_csv(OUT/"r3_duration_diagnostic.csv",index=False)

    # Source-level sensitivity uses fixed baseline centres and raw grouped copies.
    source_n = n[n.site_id.isin(anchors.index)].copy()
    for sid in anchors.index:
        base = c[(c.assigned_site == sid) & c.night.between(BASE0, BASE1)]
        la, lo = base.centroid_lat.median(), base.centroid_lon.median()
        close = c[dist(la, lo, c.centroid_lat, c.centroid_lon) <= 1000]
        ix = source_n.site_id == sid
        source_n.loc[ix, "lit"] = source_n.loc[ix, "night"].isin(close.night)
    src = transition_counts(source_n, anchors.index, "Q_only")
    src.to_csv(OUT / "r3_source_transitions.csv", index=False)
    result["source_aperture_Q_transitions"] = {
        k: int(v) for k, v in src[["HH", "HM", "MH", "MM"]].sum().items()}

    # These all refer to the same 21-site, 58-night exploratory family.
    # The source-corrected 4582 interval ends Nov 3, hence k=14.
    candidates = [(4582, 14), (4582, 15), (3669, 10), (4497, 8), (4497, 5), (4286, 4)]
    for sid, k in candidates:
        anchor = anchors.loc[sid]
        m, s = anchor.q2_n - anchor.q2_s, anchor.q2_s
        for total in result["transition_totals"]:
            if total["convention"] != "Q_only":
                continue
            aa, bb = total["MM"] + 1, total["MH"] + 1
            eb = np.exp(betaln(aa+k-1, bb) - betaln(aa, bb))
            # Independent Beta posteriors for a=P(M|H), b=P(M|M).
            pool_a = (total["HM"] + 1) / (total["HH"] + total["HM"] + 2)
            # Site-specific a matters: 4582 has 3/26 Q misses, unlike most anchors.
            site_counts = tables[0].set_index("site_id").loc[sid]
            site_a = (site_counts.HM+1)/(site_counts.HH+site_counts.HM+2)
            # Stationary per-site pi and pooled b: a=pi(1-b)/(1-pi).
            # pi~Beta(m+1,s+1), so E[pi/(1-pi)]=(m+1)/s.
            stationary_start = ((m+1)/s) * np.exp(
                betaln(aa+k-1, bb+1) - betaln(aa, bb))
            result["markov_tails"].append(dict(
                site_id=sid, k=k, pool=total["cohort"],
                pooled_a_integrated_b_times1218=float(1218*pool_a*eb),
                site_a_integrated_b_times1218=float(1218*site_a*eb),
                stationary_pi_start_times1218=float(1218*stationary_start),
                stationary_pi_boundary_times1218=float(1218*(m+1)/(m+s+2)*eb),
                fixed_b_7of19_pi_times1218=float(1218*(m+1)/(m+s+2)*(7/19)**(k-1))))

    # Circular moving calendar-night blocks from the 29-night baseline.
    # Whole fleet-day vectors stay together. Selection/estimation is frozen;
    # real baseline operating changes stay in the bundles. NOT a benign null.
    dates = pd.date_range(BASE0, BASE1).strftime("%Y-%m-%d")
    states = np.zeros((len(dates), len(anchors)), dtype=np.int8)
    runs = []
    for j, sid in enumerate(anchors.index):
        h = baseline[baseline.site_id == sid].set_index("night").reindex(dates)
        lit = h.lit.fillna(False).to_numpy(dtype=bool)
        q = h.n01.fillna(0).to_numpy() >= 2
        states[:, j] = np.where(lit, 1, np.where(q, 2, 0))
        seq = (~h.loc[q, "lit"]).astype(int).to_numpy()
        length = best = 0
        for value in seq:
            length = length+1 if value else 0
            best = max(best, length)
        runs.append(dict(site_id=int(sid), max_baseline_Q_miss_run=best))
    result["baseline_max_Q_miss_run"] = max(r["max_baseline_Q_miss_run"] for r in runs)
    pd.DataFrame(runs).to_csv(OUT / "r3_baseline_runs.csv", index=False)
    required = np.array([min_k(a) for _, a in anchors.iterrows()])
    repetitions = 20000
    for block in (1, 3, 7):
        rng = np.random.default_rng(RNG_SEED + block)
        starts = rng.integers(0, len(dates), (repetitions, int(np.ceil(58/block))))
        ix = ((starts[:, :, None] + np.arange(block)) % len(dates)).reshape(repetitions, -1)[:, :58]
        result["baseline_resampling"][str(block)] = count_alerts(states[ix], required)
    pd.DataFrame(result["markov_tails"]).to_csv(OUT / "r3_markov_tails.csv", index=False)
    return result


def pass_clusters(z):
    z = z.sort_values(["satellite", "t"]).copy()
    gap = z.groupby("satellite").t.diff()
    z["passid"] = (gap.isna() | (gap >= pd.Timedelta(seconds=10))).cumsum()
    return z


def source_checks(p, s, c):
    centers, source = {}, {}
    for sid in (4582, 4583, 4497, 4485, 4500):
        base = c[(c.assigned_site == sid) & c.night.between(BASE0, BASE1)]
        centers[sid] = (float(base.centroid_lat.median()), float(base.centroid_lon.median()))
        source[sid] = c[dist(*centers[sid], c.centroid_lat, c.centroid_lon) <= 1000].copy()
    result = dict(centers={str(k): v for k, v in centers.items()}, pairs=[], windows=[], radius_gaps=[])
    base = c[c.night.between(BASE0, BASE1)].copy()
    for a, b in ((4582, 4583), (4582, 4497), (4497, 4485), (4497, 4500)):
        da = dist(*centers[a], base.centroid_lat, base.centroid_lon)
        db = dist(*centers[b], base.centroid_lat, base.centroid_lon)
        near = base[(da <= 1000) | (db <= 1000)].copy()
        near["a"] = da[(da <= 1000) | (db <= 1000)] <= 1000
        near["b"] = db[(da <= 1000) | (db <= 1000)] <= 1000
        near["bright_a"] = near.a & (near.group_rh_mw >= 3)
        near["bright_b"] = near.b & (near.group_rh_mw >= 3)
        near = pass_clusters(near)
        ag = near.groupby("passid").agg(a=("a", "max"), b=("b", "max"),
                                       bright_a=("bright_a", "max"), bright_b=("bright_b", "max"),
                                       groups=("group_id", "size"))
        # Overlapping apertures cannot establish independent sources.
        result["pairs"].append(dict(
            a=a, b=b, platform_m=float(dist(s.loc[a,"lat"],s.loc[a,"lon"],s.loc[b,"lat"],s.loc[b,"lon"])),
            source_center_m=float(dist(*centers[a], *centers[b])), passes=len(ag),
            passes_multiple_groups=int((ag.groups >= 2).sum()),
            both_apertures_bright=int((ag.bright_a & ag.bright_b).sum()),
            apertures_overlap=bool(dist(*centers[a],*centers[b]) < 2000)))
    windows = [
        (4497, "2026-08-30", "2026-09-12"), (4497, "2026-09-13", "2026-09-26"),
        (4497, "2026-10-14", "2026-10-20"), (4497, "2026-10-21", "2026-11-03"),
        (4497, "2026-10-23", "2026-10-30"),
        (4582, "2026-10-07", "2026-10-20"), (4582, "2026-10-14", "2026-10-20"),
        (4582, "2026-10-21", "2026-11-03"),
        (4500, "2026-10-21", "2026-11-03")]
    for sid, start, end in windows:
        z = source[sid][source[sid].night.between(start, end)]
        clear = z[z.cloud_mask_peak.isin([0,1])]
        pp = p[(p.site_id == sid) & p.night.between(start,end)]
        result["windows"].append(dict(site_id=sid,start=start,end=end,groups=len(z),
            source_lit_nights=z.night.nunique(),clear_groups=len(clear),
            clear_median_mw=float(clear.group_rh_mw.median()) if len(clear) else None,
            clear_passes=int(pp.raw.sum()),satellites=pp[pp.raw].satellite.value_counts().to_dict()))
    daily, groups = [], []
    for sid in (4582,4583,4497,4485,4500):
        z = source[sid]
        for night in pd.date_range("2026-10-08",LAST).strftime("%Y-%m-%d"):
            row = z[z.night == night]
            clear = row[row.cloud_mask_peak.isin([0,1])]
            pp = p[(p.site_id==sid)&(p.night==night)]
            daily.append(dict(site_id=sid,night=night,groups=len(row),clear_groups=len(clear),
                clear_med_mw=clear.group_rh_mw.median(),max_mw=row.group_rh_mw.max(),
                assignees=";".join(str(int(i)) for i in sorted(row.assigned_site.dropna().unique())),
                n01=int(pp.raw.sum()),site_lit=bool(pp.detected.any())))
        late=z[z.night>="2026-10-19"].copy()
        late["source_label"]=sid
        late["distance_m"]=dist(*centers[sid],late.centroid_lat,late.centroid_lon)
        groups.append(late)
        for radius in (750,1000,1500,2000,3000):
            close = c[dist(*centers[sid],c.centroid_lat,c.centroid_lon)<=radius]
            for start,end in [("2026-10-21","2026-11-03"),("2026-10-23","2026-10-30")]:
                row=close[close.night.between(start,end)]
                result["radius_gaps"].append(dict(site_id=sid,radius_m=radius,start=start,end=end,
                    groups=len(row),lit_nights=row.night.nunique(),
                    lit_dates=";".join(sorted(row.night.unique()))))
    pd.DataFrame(daily).to_csv(OUT / "r3_source_daily.csv",index=False)
    pd.concat(groups).to_csv(OUT / "r3_late_source_groups.csv",index=False)
    pd.DataFrame(result["windows"]).to_csv(OUT / "r3_source_windows.csv",index=False)
    pd.DataFrame(result["radius_gaps"]).to_csv(OUT / "r3_radius_gaps.csv",index=False)
    return result


def state_check():
    rows=[]
    epsilon=.002
    for n in (15,29,45,94,573):
        for low_p in (.15,.05,.01,.0021,.002):
            rows.append(dict(clear_misses=n,epsilon=epsilon,low_p=low_p,
                conditional_independent_BF=float(((1-epsilon)/(1-low_p))**n)))
    pd.DataFrame(rows).to_csv(OUT / "r3_off_low_sensitivity.csv",index=False)
    return rows


def main():
    p,s,c=load()
    print("Loaded local passes and original copy groups",flush=True)
    result=dict(provenance=dict(pass_rows=len(p),group_rows=len(c),seed=RNG_SEED),
                null_checks=null_checks(p,c),source_checks=source_checks(p,s,c),
                off_low_sensitivity=state_check())
    (OUT / "r3_checks.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
