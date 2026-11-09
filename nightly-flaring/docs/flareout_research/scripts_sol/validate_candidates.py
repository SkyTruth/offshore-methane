"""Source-radius runs, brightness changes, and explicit null simulations."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import betaln

from spatial_and_peers import dist

OUT = Path("outputs/flareout_research/scripts_sol")


def source_runs(n, c, anchors):
    out = []
    for sid, a in anchors.iterrows():
        base = c[(c.assigned_site==sid) & c.night.between("2026-08-10", "2026-09-07")]
        clat, clon = base.centroid_lat.median(), base.centroid_lon.median()
        close = c[dist(clat, clon, c.centroid_lat, c.centroid_lon)<=1000]
        g = n[n.site_id==sid].sort_values("night").copy()
        # Proxy only: site clearing retained, but any source in this fixed aperture
        # interrupts a supposed blackout, whether assigned or unassigned/cloudy.
        g["source_lit"] = g.night.isin(close.night)
        train = g[g.night.between("2026-08-10", "2026-09-07") & g.q2]
        bn, bs = len(train), int(train.source_lit.sum())
        last_lit = g.loc[(g.night<"2026-09-08") & g.source_lit, "night"].max()
        gap = []
        def finish(gap, recovery):
            if not gap:
                return
            z = pd.DataFrame(gap)
            k = int(z.q2.sum())
            tail = np.exp(betaln(bs+1,bn-bs+1+k)-betaln(bs+1,bn-bs+1))
            out.append(dict(site_id=sid,start=z.night.min(),end=z.night.max(),before=last_lit,recovery=recovery,
                days=len(z),k=k,n=bn,s=bs,tail=tail,bonf=min(1,tail*len(anchors)*58)))
        for row in g[g.night>="2026-09-08"].itertuples(index=False):
            if row.source_lit:
                finish(gap,row.night)
                gap=[]
                last_lit=row.night
            else:
                gap.append(row._asdict())
        finish(gap,None)
    out = pd.DataFrame(out).sort_values("tail")
    out.to_csv(OUT/"source_radius_runs.csv",index=False)
    print("SOURCE-APERTURE RUNS",out[out.k>=3].to_string(index=False))
    return out


def simulate_null(n, anchors, repeats=20000):
    # Bayesian posterior predictive: one latent p per site, Bernoulli nights.
    # Non-Q nights cannot score; set their lit state false for a conservative
    # upper envelope because the real rule lets their detections reset a run.
    rng=np.random.default_rng(919067)
    hits = np.zeros(repeats,dtype=int)
    events = np.zeros(repeats,dtype=int)
    maxk = np.zeros(repeats,dtype=int)
    cutoff=.05/(len(anchors)*58)
    for sid,a in anchors.iterrows():
        q=n[(n.site_id==sid)&(n.night>="2026-09-08")].sort_values("night").q2.to_numpy()
        p=rng.beta(a.q2_s+1,a.q2_n-a.q2_s+1,size=repeats)
        k=np.zeros(repeats,dtype=int)
        seen=np.zeros(repeats,dtype=bool)
        for valid in q:
            if not valid:
                continue
            lit=rng.random(repeats)<p
            k=np.where(lit,0,k+1)
            tail=np.exp(betaln(a.q2_s+1,a.q2_n-a.q2_s+1+k)-betaln(a.q2_s+1,a.q2_n-a.q2_s+1))
            flag=(tail<=cutoff)
            # Count first crossing of each run, avoiding repeated daily alerts.
            events += flag & ~seen
            hits += flag & ~seen
            seen=np.where(lit,False,seen|flag)
            maxk=np.maximum(maxk,k)
    result=dict(replicates=repeats, familywise_alert_probability=float((hits>0).mean()),
        mean_alert_events=float(events.mean()), total_alerts=int(events.sum()), null="frozen independent-night beta posterior predictive; non-Q nights never reset")
    print("NULL SIMULATION",result)
    return result


def main():
    n=pd.read_pickle(OUT/"night_features.pkl")
    o=pd.read_pickle(OUT/"obs.pkl")
    c=pd.read_pickle(OUT/"copy_groups.pkl")
    anchors=pd.read_csv(OUT/"anchors.csv").set_index("site_id")
    r=pd.read_csv(OUT/"runs.csv")
    sr=source_runs(n,c,anchors)
    sim=simulate_null(n,anchors)
    brightness=[]
    for row in r[r.bonf<=.05].itertuples(index=False):
        g=o[(o.site_id==row.site_id)&o.cloud01&o.detected]
        pre7=(pd.Timestamp(row.start)-pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        pre14=(pd.Timestamp(row.start)-pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        before=g[g.night.between(pre7,row.start,inclusive="left")]
        earlier=g[g.night.between(pre14,pre7,inclusive="left")]
        after=g[(g.night>row.end)&(g.night<=(pd.Timestamp(row.end)+pd.Timedelta(days=7)).strftime("%Y-%m-%d"))]
        zero=o[(o.site_id==row.site_id)&o.night.between(row.start,row.end)&o.cloud01]
        brightness.append(dict(site_id=row.site_id,start=row.start,end=row.end,baseline_rh=row.median_rh,
            pre14_7_n=len(earlier),pre14_7_med=earlier.rh_mw.median(),pre7_n=len(before),pre7_med=before.rh_mw.median(),
            post7_n=len(after),post7_med=after.rh_mw.median(),
            gap_clear=len(zero),gap_hq=int(zero.high_clear.sum()),gap_central=int(zero.central.sum()),
            gap_satellite_counts=zero.satellite.value_counts().to_dict(),
            gap_satellite_nights=zero.groupby("satellite").night.nunique().to_dict()))
    pd.DataFrame(brightness).to_json(OUT/"brightness.json",orient="records",indent=2)
    print("BRIGHTNESS",json.dumps(brightness,indent=2))
    # A pre/post 5-Q-night sandwich around each gap; diagnostic threshold 4/5.
    sand=[]
    for row in r[r.k>=3].itertuples(index=False):
        g=n[(n.site_id==row.site_id)&n.q2].sort_values("night")
        pre=g[g.night<row.start].tail(5)
        post=g[g.night>row.end].head(5)
        if len(pre)==5 and len(post)==5 and pre.lit.sum()>=4 and post.lit.sum()>=4:
            sand.append(dict(site_id=row.site_id,start=row.start,end=row.end,k=row.k,bonf=row.bonf,
                pre_s=int(pre.lit.sum()),post_s=int(post.lit.sum())))
    pd.DataFrame(sand).to_csv(OUT/"sandwich.csv",index=False)
    print("SANDWICH EPISODES",sand)
    summary=dict(simulation=sim,source_radius_nominal_events=int((sr.bonf<=.05).sum()),
        source_radius_nominal_sites=int(sr.loc[sr.bonf<=.05,"site_id"].nunique()),sandwich_events=len(sand))
    (OUT/"candidate_validation.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
