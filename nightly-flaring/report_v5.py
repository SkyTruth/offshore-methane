# %%
"""Read development artifacts and write review tables without network access."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import nightfire as nf
import pipeline as p
from dev_v5 import CASES

OUT = p.OUT_DIR / "dev_v5"
TAG = "2026-10-29_2026-10-29"


def markdown(frame: pd.DataFrame, index: bool = False) -> str:
    """Format a plain Markdown table without optional pandas dependencies."""
    if index:
        frame = frame.reset_index()
    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.values.tolist()]
    return "\n".join(
        "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
        for row in rows
    )


def pair_candidates(copies: pd.DataFrame) -> pd.DataFrame:
    """Adjacent scans, columns within one; retain rejected distance/RH pairs."""
    rows = []
    for (sat, gid), frame in copies.groupby(["satellite", "granule"]):
        cs = list(frame.itertuples())
        for i, a in enumerate(cs):
            for b in cs[:i]:
                if abs(a.scan - b.scan) != 1 or abs(a.peak_col - b.peak_col) > 1:
                    continue
                distance = float(
                    nf.haversine_m(a.peak_lat, a.peak_lon, b.peak_lat, b.peak_lon)
                )
                px = distance / np.sqrt(min(a.footprint_m2, b.footprint_m2))
                if px >= 3:
                    continue
                rows.append(
                    dict(
                        satellite=sat,
                        granule=gid,
                        distance_px=px,
                        drow=abs(a.peak_row - b.peak_row),
                        rh_ratio=max(a.rh75_mw, b.rh75_mw) / min(a.rh75_mw, b.rh75_mw),
                        matched=a.group_id == b.group_id,
                    )
                )
    return pd.DataFrame(rows)


def case_tables() -> pd.DataFrame:
    """Save every detected site's v4/v5 record and every v5 source group."""
    all_rows, groups, old_groups, compact = [], [], [], []
    named = [3802, 4279, 4281, 4197, 4202, 971, 957, 3801, 3804, 4284, 3752, 981, 4267]
    for sat, gids in CASES.items():
        for gid in gids:
            stem = OUT / "cases" / f"{sat}_{gid}"
            a, b = [pd.read_csv(f"{stem}_{v}.csv") for v in ("v4", "v5")]
            merged = a.merge(
                b, on=["site_id", "satellite", "granule"], suffixes=("_v4", "_v5")
            )
            selected = merged[merged.detected_v4 | merged.detected_v5]
            all_rows.append(selected)
            c = pd.read_csv(f"{stem}_v5_copies.csv")
            groups.append(
                c.sort_values("rh75_mw", ascending=False).drop_duplicates("group_id")
            )
            captured = json.loads(Path(f"{stem}_v4_groups.json").read_text())
            for capture in captured:
                for group_id, group in enumerate(capture["groups"]):
                    old_groups.append(
                        dict(
                            satellite=sat,
                            granule=gid,
                            group_id=group_id,
                            assigned_site=int(a.iloc[group["site"]].site_id),
                            rh_mw=group["rh_mw"],
                            n_clusters=group["n_clusters"],
                            peak_lat=group["peak_lat"],
                            peak_lon=group["peak_lon"],
                            peak_row=group["peak_row"],
                            peak_col=group["peak_col"],
                            temp_k=group["peak"]["temp_k"],
                            bands="+".join(group["peak"]["bands"]),
                            fit_flag=group["peak"].get("fit_flag"),
                        )
                    )
            ids = selected[selected.site_id.isin(named)]
            compact.append(
                dict(
                    satellite=sat,
                    granule=gid,
                    detections_v4=int(a.detected.sum()),
                    detections_v5=int(b.detected.sum()),
                    rh_v4=float(a.rh_mw.sum()),
                    rh_v5=float(b.rh_mw.sum()),
                    named_sites="; ".join(
                        f"{int(r.site_id)}: {r.rh_mw_v4:.2f} → {r.rh_mw_v5:.2f}"
                        for r in ids.itertuples()
                    ),
                )
            )
    pd.concat(all_rows).to_csv(OUT / "case_records.csv", index=False)
    pd.concat(groups).to_csv(OUT / "case_groups.csv", index=False)
    pd.DataFrame(old_groups).to_csv(OUT / "case_groups_v4.csv", index=False)
    result = pd.DataFrame(compact)
    result.to_csv(OUT / "case_summary.csv", index=False)
    return result


def main() -> dict:
    """Produce local-night metrics and all-processed-granule pair diagnostics."""
    cases = case_tables()
    comparison = json.loads((OUT / "comparison.json").read_text())
    sites = pd.read_csv(p.SITES_CSV)
    over = pd.read_csv(OUT / f"overpasses_{TAG}.csv").merge(
        sites[["site_id", "lon"]], on="site_id"
    )
    local = pd.to_datetime(over.time_utc) + pd.to_timedelta(over.lon / 15, "h")
    over = p.dedupe_overpasses(
        over[
            ((local - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d") == "2026-10-29")
            & (over.sza >= 97)
        ]
    )
    det = over[over.detected]
    copies = pd.read_csv(OUT / f"copies_{TAG}.csv")
    pairs = pair_candidates(copies)
    pairs.to_csv(OUT / "pair_candidates.csv", index=False)
    bins = [0, 0.5, 1, 1.32, 1.45, 1.5, 2, 3]
    pairs["bin"] = pd.cut(pairs.distance_px, bins, right=False)
    pairs["RH ratio"] = np.where(pairs.rh_ratio < 3, "<3", ">=3")
    hist = pd.crosstab(pairs["bin"], pairs["RH ratio"], dropna=False)
    hist.to_csv(OUT / "pair_histogram.csv")
    case_copies = pd.concat(
        [pd.read_csv(f) for f in (OUT / "cases").glob("*_v5_copies.csv")],
        ignore_index=True,
    )
    case_pairs = pair_candidates(case_copies)
    case_pairs.to_csv(OUT / "case_pair_candidates.csv", index=False)
    case_pairs["bin"] = pd.cut(case_pairs.distance_px, bins, right=False)
    case_pairs["RH ratio"] = np.where(case_pairs.rh_ratio < 3, "<3", ">=3")
    case_hist = pd.crosstab(case_pairs["bin"], case_pairs["RH ratio"], dropna=False)
    case_hist.to_csv(OUT / "case_pair_histogram.csv")
    multi = det[det.n_sources > 1]
    split = det[det.n_components_split > 0]

    def edge(f):
        return int(((f.peak_col < 640) | (f.peak_col >= 2560)).sum())

    def forbidden(f):
        return f.bands.fillna("").map(
            lambda x: "T1810" not in x
            and (
                len(set(x.split("+")) & {"M10", "M11", "M12", "M13"}) < 2
                or not set(x.split("+")) & {"M10", "M11"}
            )
        )

    old = pd.read_csv(p.OUT_DIR / "dev_v4" / f"overpasses_{TAG}.csv")
    old = old[old.detected]
    audit = dict(
        local_detections=len(det),
        fit_flags=det.fit_flag.value_counts().to_dict(),
        pinned_n=int(det.fit_flag.str.startswith("pinned").sum()),
        splits_n=len(split),
        split_cloud_n=int((~p.strict_clear(split)).sum()),
        multisource_n=len(multi),
        multisource_sites=int(multi.site_id.nunique()),
        multisource_edge_n=edge(multi),
        detection_edge_n=edge(det),
        copy_groups_n=int(
            copies.drop_duplicates(["satellite", "granule", "group_id"]).shape[0]
        ),
        merged_groups_n=int(
            copies[copies.group_size > 1]
            .drop_duplicates(["satellite", "granule", "group_id"])
            .shape[0]
        ),
        copy_pairs_n=int(pairs.matched.sum()),
        pairs_n=len(pairs),
        matched_drow=pairs.loc[pairs.matched, "drow"]
        .value_counts()
        .sort_index()
        .to_dict(),
        forbidden_v4_n=int(forbidden(old).sum()),
        v4_fits_n=len(old),
        forbidden_v5_n=int(forbidden(det).sum()),
        cap_excluded_n=int(copies.n_rh_cap_excluded.sum()),
        rh75_over_peak_gt3_n=int((copies.rh75_mw / copies.rh_peak_mw > 3).sum()),
        copy_fits_n=len(copies),
    )
    if "band_rule_changed" in copies:
        audit["band_rule_changed_copy_n"] = int(
            copies.band_rule_changed.fillna(False).sum()
        )
    if "n_band_rule_changed" in over:
        audit["band_rule_changed_pixel_n"] = int(
            over.groupby(["satellite", "granule"]).n_band_rule_changed.max().sum()
        )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    table = "| Measure | v3 | v4 | v5 |\n| --- | --- | --- | --- |\n"
    for label, key in [
        ("Detections", "raw_detections"),
        ("Detected RH, MW", "total_detected_rh"),
        ("Sum strict-clear site means, MW", "strict_rh_mw"),
        ("Strict-clear detections", "strict_detections"),
    ]:
        table += (
            "| "
            + label
            + " | "
            + " | ".join(f"{comparison[v][key]:.2f}" for v in ("v3", "v4", "v5"))
            + " |\n"
        )
    text = (
        table
        + "\n"
        + markdown(cases)
        + "\n\nAll processed granules, adjacent scans and |Δcol| ≤ 1, distance <3 pixels:\n\n"
        + markdown(hist, index=True)
        + "\n\n21 spot granules:\n\n"
        + markdown(case_hist, index=True)
        + "\n\n```json\n"
        + json.dumps(audit, indent=2)
        + "\n```\n"
    )
    (OUT / "report.md").write_text(text)
    print(text)
    return audit


# %%
if __name__ == "__main__":
    main()
