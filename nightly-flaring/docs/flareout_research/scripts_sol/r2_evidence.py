"""Round 2 candidate, false-detection, heat and qualification diagnostics.

Reads only the local nightly-flaring data/caches. Writes only scripts_sol/.
Uses no annual catalogue. Copy counts collapse satellite/granule/group_id.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import betaln

sys.path.insert(0, ".")
from regions import region_of
from spatial_and_peers import dist

OUT = Path("outputs/flareout_research/scripts_sol")
CACHE = Path("outputs/flareout_research/scripts_opus/cache")
FIRST, TRAIN_END, LAST = "2026-08-10", "2026-09-07", "2026-11-04"


def load():
    p = pd.read_pickle(CACHE / "passes.pkl")
    p["raw"] = p.cloud_mask.isin([0, 1])
    p["detraw"] = p.raw & p.detected
    s = pd.read_csv("outputs/gfw_oil_sites.csv").set_index("site_id")
    s["region"] = [region_of(a, b) for a, b in zip(s.lat, s.lon)]
    cp = pd.concat([pd.read_csv(f"outputs/{folder}/copies_{tag}.csv",low_memory=False)
                    for folder,tag in [("v5_backfill","2026-08-09_2026-10-05"),
                                       ("v5_30n","2026-10-06_2026-11-04")]],ignore_index=True)
    c = cp.drop_duplicates(["satellite", "granule", "group_id"]).copy()
    c["t"] = pd.to_datetime(c.time_utc)
    c["night"] = (c.t + pd.to_timedelta(c.centroid_lon/15-12, "h")).dt.strftime("%Y-%m-%d")
    c = c[c.night.between("2026-08-09", LAST)].copy()
    return p, s, c


def candidates(p, s, c):
    runs = pd.read_pickle(CACHE / "runs.pkl")
    ids = [3662, 3644, 6011, 927, 4582, 4583, 4497]
    rows, bands, trajectories = [], [], []
    for sid in ids:
        g = p[p.site_id == sid]
        r = runs[(runs.site_id == sid) & (runs.S >= np.log(1e4))].sort_values("t0").iloc[-1]
        pre = c[(c.assigned_site == sid) & c.night.between(FIRST, TRAIN_END)]
        clat, clon = pre.centroid_lat.median(), pre.centroid_lon.median()
        d = dist(clat, clon, c.centroid_lat, c.centroid_lon)
        own = c[d <= 1000]
        gap = c[(c.t >= r.t0) & (c.t <= r.t1)].copy()
        gap["d_site"] = dist(s.loc[sid, "lat"], s.loc[sid, "lon"], gap.centroid_lat, gap.centroid_lon)
        gap["d_source"] = dist(clat, clon, gap.centroid_lat, gap.centroid_lon)
        clear = g[(g.t >= r.t0) & (g.t <= r.t1) & g.raw]
        pre7 = own[(own.night >= (pd.Timestamp(r.night0)-pd.Timedelta(days=7)).strftime("%Y-%m-%d")) &
                   (own.night < r.night0) & own.cloud_mask_peak.isin([0, 1])]
        first_full=r.night0
        if g[g.night==r.night0].detected.any():
            first_full=(pd.Timestamp(r.night0)+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        full_pre7=own[(own.night>=(pd.Timestamp(first_full)-pd.Timedelta(days=7)).strftime("%Y-%m-%d"))&
                      (own.night<first_full)&own.cloud_mask_peak.isin([0,1])]
        z = dict(site_id=sid, start=r.night0, end=r.night1, site_clear_passes=len(clear),
                 first_fully_dark_site_night=first_full,
                 pre7_full_site_clear_source_groups=len(full_pre7),
                 pre7_full_site_clear_source_median=float(full_pre7.group_rh_mw.median()),
                 high_clear_passes=int((clear.cloud_clear_neighbors>=6).sum()),
                 satellites=clear.satellite.value_counts().to_dict(),
                 nadir_clear=int((clear.zone==0).sum()), baseline_source_groups=len(pre),
                 source_lat=clat, source_lon=clon,
                 source_site_m=float(dist(clat, clon, s.loc[sid,"lat"], s.loc[sid,"lon"])),
                 source_baseline_r95=float(np.quantile(dist(clat,clon,pre.centroid_lat,pre.centroid_lon),.95)),
                 pre7_clear_source_groups=len(pre7),pre7_clear_source_median=float(pre7.group_rh_mw.median()),
                 gap_nearest_source_m=float(gap.d_source.min()),
                 gap_source_1km=int((gap.d_source<=1000).sum()),
                 gap_source_2km=int((gap.d_source<=2000).sum()),
                 gap_source_4km=int((gap.d_source<=4000).sum()),
                 gap_site_2km=int((gap.d_site<=2000).sum()),gap_site_4km=int((gap.d_site<=4000).sum()))
        rows.append(z)
        gap["target_site"] = sid
        bands.append(gap[(gap.d_source<=10000)|(gap.d_site<=10000)])
        # Fixed seven-night bins, source heat rather than assigned-site sums.
        for start in pd.date_range(FIRST, LAST, freq="7D"):
            end = min(start+pd.Timedelta(days=6),pd.Timestamp(LAST))
            co = own[own.night.between(start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))]
            cs = co[co.cloud_mask_peak.isin([0,1])]
            po = g[g.night.between(start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))]
            trajectories.append(dict(site_id=sid,start=str(start.date()),end=str(end.date()),
                                     source_groups=len(co),source_lit_nights=co.night.nunique(),
                                     clear_source_groups=len(cs),clear_source_med=cs.group_rh_mw.median(),
                                     raw_passes=int(po.raw.sum()),raw_detections=int(po.detraw.sum())))
    pd.DataFrame(rows).to_json(OUT/"r2_candidates.json",orient="records",indent=2)
    pd.concat(bands,ignore_index=True).to_csv(OUT/"r2_candidate_nearby_groups.csv",index=False)
    pd.DataFrame(trajectories).to_csv(OUT/"r2_candidate_weekly.csv",index=False)
    return rows


def epsilon(p):
    st = p[p.raw].groupby("site_id").agg(n=("detected","size"),d=("detected","sum"))
    quiet = st[st.d<=1]
    pre = p[p.night.between(FIRST,TRAIN_END)]
    everpre = pre.groupby("site_id").detected.any()
    quiet_pre = everpre.index[~everpre]
    later = p[(p.night>TRAIN_END)&p.site_id.isin(quiet_pre)]
    laterraw = later[later.raw]
    return {"retrospective_quiet_sites":len(quiet),"retrospective_clear_passes":int(quiet.n.sum()),
            "retrospective_clear_detections":int(quiet.d.sum()),
            "retrospective_rate":float(quiet.d.sum()/quiet.n.sum()),
            "training_quiet_sites":len(quiet_pre),"evaluation_clear_passes":len(laterraw),
            "evaluation_clear_detections":int(laterraw.detected.sum()),
            "evaluation_clear_rate":float(laterraw.detected.mean()),
            "evaluation_newly_detected_sites":int(later[later.detected].site_id.nunique())}


def bright_misses(p, c):
    # Match Opus's selected cohort before considering quality refinements.
    st = pd.read_pickle(CACHE/"steady.pkl")
    ids = st.index[st.med_rh>10]
    g = p[p.site_id.isin(ids)&(p.zone==0)&p.out.isin(["D","Z"])].copy()
    z = g[g.out=="Z"].copy()
    centers={}
    for sid in ids:
        pre=c[(c.assigned_site==sid)&c.night.between(FIRST,TRAIN_END)]
        centers[sid]=(pre.centroid_lat.median(),pre.centroid_lon.median())
    rows=[]
    for a in z.itertuples(index=False):
        near = c[(c.satellite==a.satellite)&((c.t-a.t).abs()<=pd.Timedelta(seconds=10))]
        # Non-detections lack peaks: source search uses the actual site's lat/lon.
        site = sites_for_misses.loc[a.site_id]
        ds = dist(site.lat,site.lon,near.centroid_lat,near.centroid_lon)
        dsource=dist(*centers[a.site_id],near.centroid_lat,near.centroid_lon)
        # A +/-2 night D/C only shows a nearby-night return, not simultaneous truth.
        same=p[(p.site_id==a.site_id)&((p.t-a.t).abs()<=pd.Timedelta(days=2))]
        rows.append(dict(site_id=a.site_id,night=a.night,satellite=a.satellite,
                         raw=bool(a.raw),neighbors=int(a.cloud_clear_neighbors),
                         n_hot_m10=int(a.n_hot_m10),
                         source2km=int((ds<=2000).sum()),source4km=int((ds<=4000).sum()),
                         source_center1km=int((dsource<=1000).sum()),
                         source2km_same_assignee=int(((ds<=2000)&(near.assigned_site==a.site_id)).sum()),
                         lit_within2days=bool(same.detected.any())))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/"r2_bright_misses.csv",index=False)
    quality=[]
    for name,m in [("all_strict",np.ones(len(g),bool)),("raw01",g.raw),
                   ("raw01_neighbors8",g.raw&(g.cloud_clear_neighbors==8)),
                   ("raw01_neighbors6",g.raw&(g.cloud_clear_neighbors>=6))]:
        q=g[m]
        quality.append(dict(quality=name,passes=len(q),misses=int((~q.detected).sum()),
                            miss_rate=float((~q.detected).mean())))
    return {"sites":len(ids),"quality":quality,"misses":len(z),
            "misses_at_sites":z.site_id.nunique(),
            "misses_with_source2km":int((out.source2km>0).sum()),
            "misses_with_source4km":int((out.source4km>0).sum()),
            "misses_with_source_center1km":int((out.source_center1km>0).sum()),
            "misses_with_source2km_same_assignee":int((out.source2km_same_assignee>0).sum()),
            "misses_lit_within2days":int(out.lit_within2days.sum()),
            "all_clear_misses":int((p.out=="Z").sum()),
            "all_clear_misses_hot_positive":int(((p.out=="Z")&(p.n_hot_m10>0)).sum())}


def qualifications(p):
    n=p.groupby(["site_id","night"]).agg(nraw=("raw","sum"),lit=("detected","max")).reset_index()
    pre=n[n.night.between(FIRST,TRAIN_END)]
    rh=p[p.night.between(FIRST,TRAIN_END)&p.detraw].groupby("site_id").rh_mw.median()
    rows=[]
    for nq in [1,2]:
        q=pre[pre.nraw>=nq].groupby("site_id").agg(n=("lit","size"),s=("lit","sum"))
        q["p"]=q.s/q.n
        q["med_rh"]=rh
        elig=q[(q.n>=12)&(q.p>=.8)&(q.med_rh>=3)]
        elig.to_csv(OUT/f"r2_eligibility_q{nq}.csv")
        a=q.loc[927]
        rows.append(dict(min_raw_clear_passes=nq,sites=len(elig),site927_n=int(a.n),
                         site927_s=int(a.s),site927_medrh=float(a.med_rh)))
    # Pool empirical transitions over qualified nights in the original 21 cohort.
    anchors=pd.read_csv(OUT/"anchors.csv").set_index("site_id")
    trans=np.zeros((2,2),int)
    per=[]
    for sid in anchors.index:
        q=pre[(pre.site_id==sid)&(pre.nraw>=2)].sort_values("night").lit.astype(int).to_numpy()
        tt=np.zeros((2,2),int)
        for a,b in zip(q[:-1],q[1:]): tt[a,b]+=1
        trans+=tt
        per.append(dict(site_id=sid,n00=int(tt[0,0]),n01=int(tt[0,1]),n10=int(tt[1,0]),n11=int(tt[1,1])))
    pd.DataFrame(per).to_csv(OUT/"r2_anchor_transitions.csv",index=False)
    return {"cohorts":rows,"baseline_Q_transitions_0miss_1lit":trans.tolist(),
            "pooled_miss_given_miss":float(trans[0,0]/trans[0].sum()),
            "pooled_miss_given_lit":float(trans[1,0]/trans[1].sum())}


def main():
    global sites_for_misses
    p,s,c=load()
    sites_for_misses=s
    result={"copy_source":"original two v5 copy CSVs, not the Opus copy cache","groups":len(c),
            "epsilon_descriptive_only":epsilon(p),"qualifications":qualifications(p),
            "bright_misses":bright_misses(p,c)}
    result["candidates"]=candidates(p,s,c)
    (OUT/"r2_evidence.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
