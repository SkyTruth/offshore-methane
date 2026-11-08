# %%
"""Evaluate the backfill gates (docs/reviews.md, review 4) for one run directory.

Compares the run with the v3 30-night files at the top of ``outputs/`` and
writes ``gates.json``, ``gates.md`` and per-gate lists under ``<run>/gates/``.
Network-free. Detection sets follow ``site_nights``: deduplicate, keep solar
zenith >= 97 degrees, then keep local nights inside the tag's interval (review
5, S1: padding nights had been counted). Raw records stay unfiltered for the
coverage and dedupe checks.

Definitions calibrated on v3 (they reproduce the review-2/3 baselines):
- gate 6 pairs: two sites < 2 km apart, both detected in one granule; edge by
  the mean site-pixel column (v3: 65.5% of 203 pairs; all detections 35.2%).
- gate 8: per (night, site, satellite) mean RH of strict-clear detections;
  share of satellite pairs whose ratio is outside 0.5-2 (v3: 22-31% under
  this definition; the review-2 figure of 26-34% used another aggregation).
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import granule as g
import nightfire as nf
import pipeline as p
import validate_myc25 as v

TAG = "2026-10-06_2026-11-04"
NADIR_ZENITH = (0.0, 36.0)  # satellite zenith span of columns 1,008-2,191
EDGE_ZENITH = (53.0, 70.0)  # span of columns < 640 or >= 2,560 (review 5, O9)
MIN_GEOMETRY_PASSES = 3  # clear passes per geometry for rate subsets (S2)
V3_ROWS = 828_621
NADIR_COLS = (1008, 2192)
EDGE_COLS = (640, 2560)
SWIR = {"M10", "M11", "M12", "M13"}
GROUP = ["satellite", "granule", "group_id"]


def is_edge(col):
    return (col < EDGE_COLS[0]) | (col >= EDGE_COLS[1])


def is_nadir(col):
    return (col >= NADIR_COLS[0]) & (col < NADIR_COLS[1])


def km(chord):
    return 2 * nf.EARTH_RADIUS_M * np.arcsin(np.minimum(np.asarray(chord) / 2, 1.0))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(folder: Path, tag: str, sites: pd.DataFrame) -> dict:
    """Raw records, deduplicated night-time records, and detections."""
    over = pd.read_csv(folder / f"overpasses_{tag}.csv")
    dd = p.dedupe_overpasses(over)
    dd = dd[dd.sza >= p.SZA_CLEAR].copy()
    lon = dd.site_id.map(sites.set_index("site_id").lon)
    local = pd.to_datetime(dd.time_utc) + pd.to_timedelta(lon / 15, "h")
    dd["night"] = (local - pd.Timedelta(hours=12)).dt.date.astype(str)
    dd["clear"] = p.strict_clear(dd)
    start, end = tag.split("_")
    dd = dd[(dd.night >= start) & (dd.night <= end)].copy()
    return dict(over=over, dd=dd, det=dd[dd.detected].copy())


def observations_detected(over: pd.DataFrame) -> int:
    """Dedupe observations (same <10 s rule) that hold at least one detection."""
    o = over[["satellite", "site_id", "time_utc", "detected"]].copy()
    o["_t"] = pd.to_datetime(o.time_utc)
    o = o.sort_values(["satellite", "site_id", "_t"], kind="stable")
    gap = o.groupby(["satellite", "site_id"])._t.diff()
    o["_obs"] = (gap.isna() | (gap >= pd.Timedelta(seconds=10))).cumsum()
    return int(o.groupby("_obs").detected.any().sum())


def band_ok(bands) -> bool:
    s = set(str(bands).split("+"))
    return len(s & SWIR) >= 2 and bool(s & {"M10", "M11"})


def accepted(flag: pd.Series) -> pd.Series:
    f = flag.fillna("unfitted")
    return ~f.str.startswith("pinned") & (f != "unfitted")


def site_tree(sites: pd.DataFrame) -> cKDTree:
    return cKDTree(g._unit_xyz(sites.lat.values, sites.lon.values))


def record_rh_variants(det: pd.DataFrame, copies: pd.DataFrame) -> pd.DataFrame:
    """Per detection: RH under mean (as run), max and sum of bow-tie copies."""
    groups = copies.dropna(subset=["assigned_site"]).drop_duplicates(GROUP).copy()
    groups["site_id"] = groups.assigned_site.astype(int)
    groups["rh_sum"] = groups.group_rh_mw * groups.group_size
    agg = groups.groupby(["satellite", "granule", "site_id"]).agg(
        rh_max=("group_max_rh_mw", "sum"), rh_sum=("rh_sum", "sum")
    )
    return det.join(agg, on=["satellite", "granule", "site_id"])


# --------------------------------------------------------------------------- gates


def gate1(base: Path) -> dict:
    files = {}
    for line in (base / "v3_30n_checksums.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            files[name] = sha256(base / name) == digest
    return dict(files=files, unchanged=all(files.values()))


def gate2(run: Path, tag: str, new: dict, copies: pd.DataFrame, lists: dict) -> dict:
    out = {}
    manifest = run / f"jobs_{tag}.csv"
    if manifest.exists():
        jobs = pd.read_csv(manifest)
        out["manifest"] = json.loads((run / f"run_summary_{tag}.json").read_text())
        lists["missing_jobs"] = jobs[~jobs.complete.astype(bool)]
        jobs["date"] = jobs.granule.str[1:9]
        out["jobs_by_satellite"] = jobs.groupby("satellite").size().to_dict()
        per_day = jobs.groupby(["date", "satellite"]).size().unstack(fill_value=0)
        out["days_without_jobs"] = {
            s: per_day.index[per_day[s] == 0].tolist() for s in per_day
        }
    log = run / "run.log"
    if log.exists():
        out["skipped_coverage"] = [
            line.split(": ", 1)[1]
            for line in log.read_text().splitlines()
            if line.startswith("Skipped empty coverage")
        ]
    over = new["over"]
    out["site_rows"] = len(over)
    out["site_rows_vs_v3"] = len(over) / V3_ROWS - 1
    key = ["satellite", "granule", "site_id"]
    out["duplicate_site_records"] = int(over.duplicated(key).sum())
    groups = copies.dropna(subset=["assigned_site"]).drop_duplicates(GROUP).copy()
    groups["site_id"] = groups.assigned_site.astype(int)
    agg = (
        groups.groupby(key)
        .agg(n_groups=("group_id", "size"), group_rh=("group_rh_mw", "sum"))
        .reset_index()
    )
    det = over[over.detected]
    m = det.merge(agg, on=key, how="outer", indicator=True)
    both = m[m._merge == "both"]
    bad = both[
        (both.n_sources != both.n_groups)
        | ((both.rh_mw - both.group_rh).abs() > 1e-3 * np.maximum(1, both.group_rh))
    ]
    lists["reconcile_mismatch"] = pd.concat(
        [bad, m[m._merge != "both"]], ignore_index=True
    )
    out["reconcile"] = dict(
        detections=len(det),
        assigned_site_groups=len(agg),
        matched=len(both),
        detections_without_groups=int((m._merge == "left_only").sum()),
        groups_without_detection=int((m._merge == "right_only").sum()),
        source_or_rh_mismatch=len(bad),
    )
    return out


def gate3(det: pd.DataFrame, copies: pd.DataFrame, lists: dict) -> dict:
    out = {}
    for name, f in (("copies", copies), ("detections", det)):
        acc = accepted(f.fit_flag)
        two = f.bands.fillna("").str.count(r"\+") == 1
        out[name] = dict(
            n=len(f),
            fit_flags=f.fit_flag.fillna("unfitted").value_counts().to_dict(),
            accepted_t_out_of_range=int(
                (acc & ((f.temp_k < nf.FIT_T_MIN) | (f.temp_k > nf.FIT_T_MAX))).sum()
            ),
            accepted_two_band_out_of_range=int(
                (
                    acc
                    & two
                    & ((f.temp_k < nf.TWO_BAND_T_MIN) | (f.temp_k > nf.TWO_BAND_T_MAX))
                ).sum()
            ),
            accepted_band_rule_failures=int((acc & ~f.bands.map(band_ok)).sum()),
            pinned_share=float(f.fit_flag.fillna("").str.startswith("pinned").mean()),
            eliminated_m10=int(
                f.fit_flag.fillna("").str.match(r"eliminated:.*M10(\+|$)").sum()
            ),
        )
        rh = f.rh_mw if "rh_mw" in f else f.rh75_mw
        out[name]["pinned_rh_share"] = float(
            rh[f.fit_flag.fillna("").str.startswith("pinned")].sum() / rh.sum()
        )
    low = copies[copies.fit_flag == "pinned:low_t"]
    lists["pinned_low_t_copies"] = low
    out["pinned_low_t_copies"] = dict(
        n=len(low),
        rh_share=float(low.rh75_mw.sum() / copies.rh75_mw.sum()),
        t_fit_rejected_p50=float(low.t_fit_rejected.median()) if len(low) else None,
    )
    if "band_rule_changed" in copies:
        out["band_rule_changed_copies"] = int(
            copies.band_rule_changed.fillna(False).astype(bool).sum()
        )
    return out


def gate4(
    run_nights: pd.DataFrame, det: pd.DataFrame, copies: pd.DataFrame, lists: dict
) -> dict:
    ratio = copies.rh75_mw / copies.rh_peak_mw
    tail = copies[(ratio > 3) | (copies.n_rh_cap_excluded > 0)]
    lists["rh_tail_copies"] = tail
    top = run_nights.nlargest(20, "rh_mw")[["night", "site_id", "rh_mw", "n_clear"]]
    rows = []
    for t in top.itertuples():
        recs = det[(det.night == t.night) & (det.site_id == t.site_id) & det.clear]
        for r in recs.itertuples():
            cs = copies[
                (copies.satellite == r.satellite)
                & (copies.granule == r.granule)
                & (copies.assigned_site == r.site_id)
            ]
            for c in cs.itertuples():
                rows.append(
                    dict(
                        night=t.night,
                        site_id=t.site_id,
                        night_rh_mw=t.rh_mw,
                        n_clear=t.n_clear,
                        satellite=r.satellite,
                        granule=r.granule,
                        record_rh_mw=r.rh_mw,
                        group_id=c.group_id,
                        group_size=c.group_size,
                        copy_rh75_mw=c.rh75_mw,
                        rh75_over_peak=c.rh75_mw / c.rh_peak_mw,
                        n_rh_cap_excluded=c.n_rh_cap_excluded,
                        temp_k=c.temp_k,
                        bands=c.bands,
                        fit_flag=c.fit_flag,
                        peak_col=c.peak_col,
                        cloud_mask_peak=c.cloud_mask_peak,
                    )
                )
    table = pd.DataFrame(rows)
    lists["top20_site_nights"] = table
    flagged = (
        table[
            (table.rh75_over_peak > 3)
            | (table.n_rh_cap_excluded > 0)
            | ~accepted(table.fit_flag)
        ]
        if len(table)
        else table
    )
    return dict(
        copies=len(copies),
        rh75_over_peak_gt3=int((ratio > 3).sum()),
        cap_excluded=int((copies.n_rh_cap_excluded > 0).sum()),
        tail_share=len(tail) / len(copies),
        top20_copy_rows=len(table),
        top20_flagged_copy_rows=len(flagged),
        top20_sites=top.site_id.tolist(),
    )


def candidate_pairs(copies: pd.DataFrame, max_px: float = 3.0) -> pd.DataFrame:
    """Same-granule copies in adjacent scans with |dcol| <= 1 and < max_px."""
    rows = []
    cols = [
        "scan",
        "peak_col",
        "peak_lat",
        "peak_lon",
        "footprint_m2",
        "rh75_mw",
        "group_id",
        "assigned_site",
    ]
    for (sat, gid), f in copies.groupby(["satellite", "granule"]):
        if len(f) < 2:
            continue
        recs = f[cols].to_dict("records")
        for a, b in itertools.combinations(recs, 2):
            if (
                abs(a["scan"] - b["scan"]) != 1
                or abs(a["peak_col"] - b["peak_col"]) > 1
            ):
                continue
            d = float(
                nf.haversine_m(
                    a["peak_lat"], a["peak_lon"], b["peak_lat"], b["peak_lon"]
                )
            )
            px = d / np.sqrt(min(a["footprint_m2"], b["footprint_m2"]))
            if px >= max_px:
                continue
            rows.append(
                dict(
                    satellite=sat,
                    granule=gid,
                    distance_px=px,
                    rh_ratio=max(a["rh75_mw"], b["rh75_mw"])
                    / min(a["rh75_mw"], b["rh75_mw"]),
                    matched=a["group_id"] == b["group_id"],
                    site_a=a["assigned_site"],
                    site_b=b["assigned_site"],
                    col=(a["peak_col"] + b["peak_col"]) / 2,
                )
            )
    return pd.DataFrame(rows)


def cross_granule_pairs(copies: pd.DataFrame) -> pd.DataFrame:
    """Copies across consecutive granules of one satellite that would pair.

    Applies the D36 guards (|dcol| <= 1, < 1.5 px, RH ratio < 3) and requires
    one copy in scan 0 and the other in scan >= 46 of the earlier granule.
    """
    rows = []
    t = pd.to_datetime(copies.time_utc)
    for sat, f in copies.assign(_t=t).groupby("satellite"):
        f = f.reset_index(drop=True)
        xyz = g._unit_xyz(f.peak_lat.values, f.peak_lon.values)
        radius = 1.5 * np.sqrt(f.footprint_m2.max()) / nf.EARTH_RADIUS_M
        for i, j in cKDTree(xyz).query_pairs(radius):
            a, b = f.iloc[i], f.iloc[j]
            if a.granule == b.granule or abs((a._t - b._t).total_seconds()) > 200:
                continue
            if abs(a.peak_col - b.peak_col) > 1:
                continue
            d = float(nf.haversine_m(a.peak_lat, a.peak_lon, b.peak_lat, b.peak_lon))
            px = d / np.sqrt(min(a.footprint_m2, b.footprint_m2))
            ratio = max(a.rh75_mw, b.rh75_mw) / min(a.rh75_mw, b.rh75_mw)
            first, second = (a, b) if a.granule < b.granule else (b, a)
            if px >= 1.5 or ratio >= 3 or second.scan != 0 or first.scan < 46:
                continue
            rows.append(
                dict(
                    satellite=sat,
                    granule_a=a.granule,
                    granule_b=b.granule,
                    scan_a=a.scan,
                    scan_b=b.scan,
                    distance_px=px,
                    rh_a=a.rh75_mw,
                    rh_b=b.rh75_mw,
                    rh_ratio=ratio,
                    site_a=a.assigned_site,
                    site_b=b.assigned_site,
                )
            )
    return pd.DataFrame(rows)


def gate5(copies: pd.DataFrame, lists: dict) -> dict:
    out = {}
    grp = copies.groupby(GROUP)
    out["groups"] = int(grp.ngroups)
    out["groups_repeating_scan"] = int(
        grp.scan.apply(lambda s: s.duplicated().any()).sum()
    )
    sizes = grp.size()
    out["group_size_counts"] = {int(k): int(n) for k, n in sizes.value_counts().items()}
    big = sizes[sizes >= 3].index
    lists["groups_3plus"] = (
        copies.set_index(GROUP).loc[big].reset_index() if len(big) else copies.iloc[:0]
    )
    merged = copies[copies.group_size > 1]
    out["edge_guard_violations"] = int(
        (
            (merged.pair_distance_px >= g.BOWTIE_MERGE_PIXELS)
            | (merged.pair_dcol > 1)
            | (merged.pair_rh_ratio >= 3)
        ).sum()
    )
    scans = merged.groupby(GROUP).scan.agg(lambda s: s.max() - s.min() + 1 == len(s))
    out["merged_groups_non_adjacent_scans"] = int((~scans).sum())
    mean = grp.rh75_mw.mean()
    mx = grp.rh75_mw.max()
    first = grp.first()
    out["group_mean_mismatch"] = int(
        ((mean - first.group_rh_mw).abs() > 1e-6 * mean).sum()
    )
    out["group_max_mismatch"] = int(
        ((mx - first.group_max_rh_mw).abs() > 1e-6 * mx).sum()
    )
    xyz = (
        g._unit_xyz(copies.peak_lat.values, copies.peak_lon.values)
        * copies.rh75_mw.values[:, None]
    )
    w = pd.DataFrame(xyz, columns=list("xyz"), index=copies.index).join(copies[GROUP])
    s = w.groupby(GROUP)[list("xyz")].sum()
    s = s.div(np.linalg.norm(s.values, axis=1), axis=0)
    lat = np.degrees(np.arcsin(s.z))
    lon = np.degrees(np.arctan2(s.y, s.x))
    err = nf.haversine_m(
        lat.values, lon.values, first.centroid_lat.values, first.centroid_lon.values
    )
    out["centroid_mismatch_gt_1m"] = int((np.asarray(err) > 1).sum())

    pairs = candidate_pairs(copies)
    lists["pair_candidates"] = pairs
    bins = [0, 0.5, 1, 1.32, 1.5, 2, 3]
    if len(pairs):
        pairs["bin"] = pd.cut(pairs.distance_px, bins, right=False).astype(str)
        pairs["ratio_class"] = np.select(
            [pairs.rh_ratio < 3, pairs.rh_ratio < 10], ["<3", "3-10"], ">=10"
        )
        hist = pd.crosstab([pairs.bin, pairs.matched], pairs.ratio_class).reset_index()
        lists["pair_histogram"] = hist
        out["pair_histogram"] = hist.to_dict("records")
        band = (pairs.distance_px >= 1.32) & (pairs.distance_px < 1.5)
        out["candidates"] = len(pairs)
        out["candidates_1p32_1p5"] = int(band.sum())
        out["candidates_1p32_1p5_share"] = float(band.mean())
        out["matched_pairs"] = int(pairs.matched.sum())
        out["matched_1p32_1p5"] = int((band & pairs.matched).sum())
        out["matched_1p32_1p5_share"] = float(
            (band & pairs.matched).sum() / max(1, pairs.matched.sum())
        )
        un = pairs[(pairs.distance_px < 1) & ~pairs.matched]
        lists["unmerged_under_1px"] = un
        out["unmerged_under_1px"] = len(un)
        out["unmerged_under_1px_ratio_ge3"] = int((un.rh_ratio >= 3).sum())
        diff = pairs[
            (pairs.distance_px < 2)
            & (pairs.rh_ratio >= 10)
            & (pairs.site_a.fillna(-1) != pairs.site_b.fillna(-1))
        ]
        lists["ratio10_different_sites"] = diff
        out["under_2px_ratio_ge10_different_sites"] = len(diff)
    cross = cross_granule_pairs(copies)
    lists["cross_granule_pairs"] = cross
    out["cross_granule_pairs"] = len(cross)
    out["cross_granule_pairs_same_site"] = int(
        (cross.site_a == cross.site_b).sum() if len(cross) else 0
    )
    return out


def same_pass_pairs(
    det: pd.DataFrame, sites: pd.DataFrame, col: str = "col"
) -> pd.DataFrame:
    loc = sites.set_index("site_id")[["lat", "lon"]]
    rows = []
    for (s, gid), f in det.groupby(["satellite", "granule"]):
        if len(f) < 2:
            continue
        recs = f[["site_id", col]].values
        for a, b in itertools.combinations(recs, 2):
            la, lb = loc.loc[int(a[0])], loc.loc[int(b[0])]
            if nf.haversine_m(la.lat, la.lon, lb.lat, lb.lon) < 2000:
                rows.append(
                    dict(
                        satellite=s,
                        granule=gid,
                        site_a=a[0],
                        site_b=b[0],
                        col=(a[1] + b[1]) / 2,
                    )
                )
    return pd.DataFrame(
        rows, columns=["satellite", "granule", "site_a", "site_b", "col"]
    )


def site_switches(
    det: pd.DataFrame, sites: pd.DataFrame, myc: pd.DataFrame
) -> pd.DataFrame:
    """MYC flares with >= 2 GFW sites within 2 km: which site takes each pass."""
    tree = site_tree(sites)
    near = tree.query_ball_point(
        g._unit_xyz(myc.lat.values, myc.lon.values), 2000 / nf.EARTH_RADIUS_M
    )
    rows = []
    ids = sites.site_id.values
    for flare, idx in zip(myc.flare_id, near):
        if len(idx) < 2:
            continue
        own = set(ids[idx])
        f = det[det.site_id.isin(own)]
        per_pass = f.groupby(["satellite", "granule"]).site_id.agg(
            lambda s: tuple(sorted(set(s)))
        )
        single = per_pass[per_pass.map(len) == 1].map(lambda t: t[0])
        counts = single.value_counts()
        rows.append(
            dict(
                flare_id=flare,
                n_sites=len(own),
                passes=len(per_pass),
                multi_site_passes=int((per_pass.map(len) > 1).sum()),
                single_site_passes=len(single),
                sites_taking_passes=len(counts),
                modal_share=float(counts.iloc[0] / counts.sum())
                if len(counts)
                else np.nan,
            )
        )
    return pd.DataFrame(rows)


def gate6(
    new: dict,
    old: dict,
    copies: pd.DataFrame,
    sites: pd.DataFrame,
    myc: pd.DataFrame,
    lists: dict,
) -> dict:
    out = {}
    for name, d in (("v3", old["det"]), ("run", new["det"])):
        pr = same_pass_pairs(d, sites)
        out[name] = dict(
            detections=len(d),
            all_detection_edge_share=float(is_edge(d.col).mean()),
            same_pass_pairs=len(pr),
            same_pass_pair_edge_share=float(is_edge(pr.col).mean())
            if len(pr)
            else None,
        )
        lists[f"same_pass_pairs_{name}"] = pr
        sw = site_switches(d, sites, myc)
        lists[f"site_switches_{name}"] = sw
        act = sw[sw.passes > 0]
        out[name]["switch_flares"] = dict(
            flares_with_2plus_sites=len(sw),
            with_detections=len(act),
            with_switches=int((act.sites_taking_passes > 1).sum()),
            multi_site_passes=int(act.multi_site_passes.sum()),
            single_site_passes=int(act.single_site_passes.sum()),
            off_modal_single_passes=int(
                (act.single_site_passes * (1 - act.modal_share.fillna(1))).round().sum()
            ),
        )
    d = new["det"]
    pr = same_pass_pairs(d, sites, "peak_col")
    out["run"]["same_pass_pair_edge_share_peak_col"] = (
        float(is_edge(pr.col).mean()) if len(pr) else None
    )
    multi = d[d.n_sources > 1]
    out["run"]["multi_source"] = len(multi)
    out["run"]["multi_source_edge_share"] = (
        float(is_edge(multi.peak_col).mean()) if len(multi) else None
    )
    lists["multi_source_records"] = multi

    # Centroid vs strongest-copy site for merged groups.
    tree = site_tree(sites)
    ids = sites.site_id.values
    merged = copies[copies.group_size > 1]
    best = merged.loc[merged.groupby(GROUP).rh75_mw.idxmax()]
    chord, j = tree.query(g._unit_xyz(best.peak_lat.values, best.peak_lon.values))
    best = best.assign(
        peak_site=np.where(km(chord) <= g.ASSOC_M, ids[j], np.nan),
    )
    changed = best[best.peak_site.fillna(-1) != best.assigned_site.fillna(-1)].copy()
    mtree = cKDTree(g._unit_xyz(myc.lat.values, myc.lon.values))
    chord, k = mtree.query(
        g._unit_xyz(changed.centroid_lat.values, changed.centroid_lon.values)
    )
    changed["myc_flare"] = myc.flare_id.values[k]
    changed["myc_dist_centroid_m"] = km(chord)
    loc = sites.set_index("site_id")[["lat", "lon"]]

    def dist(site, i):
        if pd.isna(site):
            return np.nan
        s = loc.loc[int(site)]
        return float(nf.haversine_m(myc.lat.values[i], myc.lon.values[i], s.lat, s.lon))

    changed["myc_to_centroid_site_m"] = [
        dist(s, i) for s, i in zip(changed.assigned_site, k)
    ]
    changed["myc_to_peak_site_m"] = [dist(s, i) for s, i in zip(changed.peak_site, k)]
    lists["centroid_site_changes"] = changed
    near = changed[changed.myc_dist_centroid_m <= 2000]
    sides = near.myc_to_centroid_site_m < near.myc_to_peak_site_m.fillna(np.inf)
    out["run"]["centroid_site_changes"] = dict(
        merged_groups=int(best.shape[0]),
        changed=len(changed),
        with_myc_within_2km=len(near),
        myc_sides_with_centroid=int(sides.sum()),
        share=float(sides.mean()) if len(near) else None,
    )
    return out


def flame_area_ratio(zenith_deg: float, h_over_r: float = 1.6) -> float:
    """Apparent/nadir area of an upright ellipsoidal flame (Elvidge 2016 eq. 3)."""
    c = np.cos(2 * np.radians(zenith_deg))
    return float(np.sqrt((h_over_r**2 + 1 - (h_over_r**2 - 1) * c) / 2))


def flame_zone_ratio(h_over_r: float) -> float:
    """Edge-zone/nadir-zone mean apparent area, uniform in zenith per zone."""

    def zone(lo, hi):
        return np.mean(
            [flame_area_ratio(z, h_over_r) for z in np.linspace(lo, hi, 181)]
        )

    return float(zone(*EDGE_ZENITH) / zone(*NADIR_ZENITH))


def implied_h_over_r(ratio: float) -> float | None:
    """H/R at which the zone-mean flame model equals an observed ratio."""
    from scipy.optimize import brentq

    if ratio is None or not 1.0 < ratio < flame_zone_ratio(6.0):
        return None
    return float(brentq(lambda h: flame_zone_ratio(h) - ratio, 1.0, 6.0))


def gate7(
    new: dict, old: dict, copies: pd.DataFrame, run: Path, tag: str, lists: dict
) -> dict:
    """Edge/nadir RH at persistent MYC sites, with selection-limited subsets.

    Edge detections are conditioned on detection, and small flares escape
    detection at the edge, so the all-detection ratio is biased upward. Sites
    detected on most clear passes at both geometries limit that bias. The
    flame-shape model averaged over each zone gives 1.36 for H/R 1.6 (review
    5, O9); with H/R free (published range 1-4) it fits most ratios, so the
    implied H/R is reported instead of a pass band.
    """
    val = pd.read_csv(run / f"validation_myc25_{tag}.csv")
    pers = set(val.loc[val.myc_regime == "persistent", "site_id"])
    out = dict(
        persistent_sites=len(pers),
        flame_zone_ratio={
            h: flame_zone_ratio(h) for h in (1.0, 1.4, 1.6, 1.8, 2.0, 2.5)
        },
    )

    def ratios(d, col, rh_cols):
        d = d[d.site_id.isin(pers) & d.clear].copy()
        d["geo"] = np.select(
            [is_nadir(d[col]), is_edge(d[col])], ["nadir", "edge"], "mid"
        )
        d = d[d.geo != "mid"]
        res = {}
        for rh in rh_cols:
            d["rh0"] = d[rh].where(d.detected, 0.0)
            s = (
                d.groupby(["site_id", "geo"])
                .agg(
                    n=("rh0", "size"),
                    rate=("detected", "mean"),
                    mean0=("rh0", "mean"),
                    mean_det=(rh, lambda x: x[d.loc[x.index, "detected"]].mean()),
                )
                .unstack()
            )
            enough = (s[("n", "edge")] >= MIN_GEOMETRY_PASSES) & (
                s[("n", "nadir")] >= MIN_GEOMETRY_PASSES
            )
            subsets = {
                "all_detected": s[("mean_det", "edge")].notna()
                & s[("mean_det", "nadir")].notna(),
                "rate_ge_0.5": enough
                & (s[("rate", "edge")] >= 0.5)
                & (s[("rate", "nadir")] >= 0.5),
                "rate_ge_0.8": enough
                & (s[("rate", "edge")] >= 0.8)
                & (s[("rate", "nadir")] >= 0.8),
            }
            res[rh] = {}
            for name, keep in subsets.items():
                r = (
                    s.loc[keep, ("mean_det", "edge")]
                    / s.loc[keep, ("mean_det", "nadir")]
                ).dropna()
                res[rh][name] = dict(
                    n_sites=len(r),
                    median=float(r.median()) if len(r) else None,
                    implied_h_over_r=implied_h_over_r(float(r.median()))
                    if len(r)
                    else None,
                    p25=float(r.quantile(0.25)) if len(r) else None,
                    p75=float(r.quantile(0.75)) if len(r) else None,
                )
            both = (
                (s[("n", "edge")] >= 3)
                & (s[("n", "nadir")] >= 3)
                & (s[("mean0", "nadir")] > 0)
            )
            r0 = (
                s.loc[both, ("mean0", "edge")] / s.loc[both, ("mean0", "nadir")]
            ).dropna()
            res[rh]["zero_inclusive_3plus_clear"] = dict(
                n_sites=len(r0), median=float(r0.median()) if len(r0) else None
            )
            pooled = d.groupby("geo").rh0.mean()
            res[rh]["pooled_zero_inclusive"] = float(
                pooled.get("edge", np.nan) / pooled.get("nadir", np.nan)
            )
        return res

    out["v3_site_col"] = ratios(old["dd"], "col", ["rh_mw"])
    rec = record_rh_variants(new["dd"], copies)
    rec["rh_max"] = rec.rh_max.fillna(0.0)
    rec["rh_sum"] = rec.rh_sum.fillna(0.0)
    out["run_site_col"] = ratios(rec, "col", ["rh_mw", "rh_max", "rh_sum"])
    rec["peak_col"] = rec.peak_col.fillna(rec.col)
    out["run_peak_col"] = ratios(rec, "peak_col", ["rh_mw", "rh_max", "rh_sum"])
    return out


def gate8(d: pd.DataFrame) -> dict:
    x = d[d.clear].groupby(["night", "site_id", "satellite"]).rh_mw.mean().unstack()
    res = {}
    for a, b in itertools.combinations(sorted(x.columns), 2):
        q = x[[a, b]].dropna()
        r = q[a] / q[b]
        res[f"{a}/{b}"] = dict(
            n=len(q),
            differ_gt_2x=float(((r > 2) | (r < 0.5)).mean()),
            median_ratio=float(r.median()),
            log_r=float(np.corrcoef(np.log(q[a]), np.log(q[b]))[0, 1]),
        )
    return res


def gate9(det: pd.DataFrame) -> dict:
    """Split components (the gate's unit) and affected records (review 5, O10)."""
    split = det[det.n_components_split > 0]
    components = {
        (r.satellite, r.granule, c)
        for r in split.itertuples()
        for c in str(r.split_component_ids).split(",")
        if c and c != "nan"
    }
    return dict(
        detections=len(det),
        split_components=len(components),
        component_share=len(components) / len(det),
        split_records=len(split),
        record_share=len(split) / len(det),
        record_cloud_share=float((~split.clear).mean()) if len(split) else None,
    )


def site_mean_total(nights: pd.DataFrame) -> float:
    return float(nights.groupby("site_id").rh_mw.mean().sum())


def gate10(new: dict, old: dict, run_nights, v3_nights) -> dict:
    out = {}
    for name, d, nights in (("v3", old, v3_nights), ("run", new, run_nights)):
        det = d["det"]
        out[name] = dict(
            detections=len(det),
            detected_rh_mw=float(det.rh_mw.sum()),
            strict_clear_detections=int(det.clear.sum()),
            strict_clear_rh_mw=float(det.rh_mw[det.clear].sum()),
            by_satellite=det.groupby("satellite").size().to_dict(),
            sum_site_means_mw=site_mean_total(nights),
            sum_site_means_bcm=site_mean_total(nights) * p.BCM_PER_MW_YEAR,
            observations_with_detection=observations_detected(d["over"]),
            deduped_detections_all_sza=int(
                p.dedupe_overpasses(d["over"]).detected.sum()
            ),
        )
        out[name]["dedupe_lost"] = (
            out[name]["observations_with_detection"]
            - out[name]["deduped_detections_all_sza"]
        )
    for key in (
        "detections",
        "detected_rh_mw",
        "strict_clear_rh_mw",
        "sum_site_means_mw",
    ):
        out[f"{key}_change"] = out["run"][key] / out["v3"][key] - 1
    return out


def gate11(run: Path, tag: str, base: Path) -> dict:
    out = {}
    for name, folder in (("v3", base), ("run", run)):
        summary = json.loads((folder / f"validation_summary_{tag}.json").read_text())
        out[name] = {
            k: summary.get(k)
            for k in (
                "matched",
                "log_r",
                "n_both",
                "median_ratio",
                "det_freq",
                "det_freq_like_for_like",
                "det_freq_myc_strict",
                "det_freq_myc_detection_clears",
            )
            if k in summary
        }
    val = pd.read_csv(run / f"validation_myc25_{tag}.csv")
    checks = {"own": ("detections", "clear_overpasses")}
    for prefix in ("lf", "lf_strict"):
        if f"{prefix}_detections" in val:
            checks[prefix] = (f"{prefix}_detections", f"{prefix}_clear_passes")
    out["detections_exceed_clear"] = {
        k: int((val[a] > val[b]).sum()) for k, (a, b) in checks.items()
    }
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v_) for k, v_ in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v_) for v_ in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def main(
    run: Path = p.OUT_DIR / "v5_30n", tag: str = TAG, base: Path = p.OUT_DIR
) -> dict:
    sites = pd.read_csv(p.SITES_CSV)
    myc = v.load_myc(2025)
    new = load(run, tag, sites)
    old = load(base, tag, sites)
    copies = pd.read_csv(run / f"copies_{tag}.csv")
    run_nights = pd.read_csv(run / f"site_nights_{tag}.csv")
    v3_nights = pd.read_csv(base / f"site_nights_{tag}.csv")
    lists = {}
    result = dict(
        gate1_baseline=gate1(base),
        gate2_coverage=gate2(run, tag, new, copies, lists),
        gate3_fits=gate3(new["det"], copies, lists),
        gate4_rh_tail=gate4(run_nights, new["det"], copies, lists),
        gate5_pairs=gate5(copies, lists),
        gate6_attribution=gate6(new, old, copies, sites, myc, lists),
        gate7_edge_nadir=gate7(new, old, copies, run, tag, lists),
        gate8_cross_satellite=dict(v3=gate8(old["det"]), run=gate8(new["det"])),
        gate9_splits=gate9(new["det"]),
        gate10_totals=gate10(new, old, run_nights, v3_nights),
        gate11_myc=gate11(run, tag, base),
    )
    result = jsonable(result)
    folder = run / "gates"
    folder.mkdir(exist_ok=True)
    for name, frame in lists.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    (run / "gates.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# %%
if __name__ == "__main__":
    main()
