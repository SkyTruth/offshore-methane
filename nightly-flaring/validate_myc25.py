# %%
"""Compare a site-night run with EOG's MYC25Q4 annual upstream flare sites.

Downloads ``annual_flare_site_records.csv`` (CC BY-NC 4.0) from Zenodo into
``outputs/`` once, matches GFW and MYC25Q4 sites one-to-one within 2 km, and reports
pooled clear-overpass detection rates and volume agreement by annual MYC regime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
MYC_URL = (
    "https://zenodo.org/api/records/23091822/files/"
    "annual_flare_site_records.csv/content"
)
EARTH_RADIUS_M = 6_371_000.0


def load_myc(year: int = 2025, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    path = out_dir / "myc25q4_annual_flare_site_records.csv"
    if not path.exists():
        resp = requests.get(MYC_URL, timeout=300)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    myc = pd.read_csv(path)
    return myc[myc["year"] == year].reset_index(drop=True)


def _xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    )


def pooled_observations(nights: pd.DataFrame, over: pd.DataFrame) -> pd.DataFrame:
    """Pool deduplicated strict clear observations within the local-night period."""
    from pipeline import dedupe_overpasses, strict_clear

    sites = nights[["site_id", "lat", "lon"]].drop_duplicates("site_id")
    obs = dedupe_overpasses(over).merge(sites[["site_id", "lon"]], on="site_id")
    local = pd.to_datetime(obs.time_utc) + pd.to_timedelta(obs.lon / 15, "h")
    obs["night"] = (local - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    obs = obs.merge(
        nights[["site_id", "night"]].astype({"night": str}), on=["site_id", "night"]
    )
    obs = obs[strict_clear(obs)].copy()
    obs["rh_clear"] = obs.rh_mw.where(obs.detected, 0.0)
    pooled = obs.groupby("site_id").agg(
        clear_overpasses=("detected", "size"),
        detections=("detected", "sum"),
        rh_sum=("rh_clear", "sum"),
    )
    counts = (
        obs.groupby(["site_id", "night"])
        .detected.max()
        .groupby("site_id")
        .agg(nights_clear="size", nights_flaring="sum")
    )
    result = sites.merge(pooled.join(counts), on="site_id", how="left")
    for col in ("clear_overpasses", "detections", "nights_clear", "nights_flaring"):
        result[col] = result[col].fillna(0).astype(int)
    result["rh_mw"] = result.rh_sum / result.clear_overpasses.replace(0, np.nan)
    result["bcm_per_year"] = 0.0116 * result.rh_mw
    result["det_freq"] = result.detections / result.clear_overpasses.replace(0, np.nan)
    return result


def like_for_like_frequency(
    nights: pd.DataFrame,
    over: pd.DataFrame,
    lat: float,
    lon: float,
    copies: pd.DataFrame | None = None,
    sensitivity: bool = True,
) -> tuple[int, int]:
    """Count source-radius detections on strict or detection-cleared passes."""
    from pipeline import strict_clear, SZA_CLEAR
    from nightfire import haversine_m

    sites = nights[["site_id", "lat", "lon"]].drop_duplicates("site_id")
    nearby = sites[haversine_m(lat, lon, sites.lat, sites.lon) <= 2000]
    obs = over[over.site_id.isin(nearby.site_id)].copy()
    obs = obs.merge(sites[["site_id", "lon"]], on="site_id")
    local = pd.to_datetime(obs.time_utc) + pd.to_timedelta(obs.lon / 15, "h")
    obs["night"] = (local - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    obs = obs.merge(
        nights[["site_id", "night"]].astype({"night": str}), on=["site_id", "night"]
    )
    obs = obs[obs.sza >= SZA_CLEAR].copy()
    if copies is not None:
        sources = copies[
            haversine_m(lat, lon, copies.centroid_lat, copies.centroid_lon) <= 2000
        ]
        detected = []
        for row in obs.itertuples():
            hit = sources[
                (sources.satellite == row.satellite) & (sources.granule == row.granule)
            ]
            detected.append(
                bool(
                    (
                        abs(pd.to_datetime(hit.time_utc) - pd.Timestamp(row.time_utc))
                        < pd.Timedelta(seconds=10)
                    ).any()
                )
            )
        obs["detected"] = detected
    elif "peak_lat" in obs:
        obs["detected"] &= haversine_m(lat, lon, obs.peak_lat, obs.peak_lon) <= 2000
    obs["clear"] = strict_clear(obs)
    if sensitivity:
        obs["clear"] |= obs.detected
    obs["_time"] = pd.to_datetime(obs.time_utc)
    obs = obs.sort_values(["satellite", "_time"], kind="stable")
    gap = obs.groupby("satellite")._time.diff()
    obs["_pass"] = (gap.isna() | (gap >= pd.Timedelta(seconds=10))).cumsum()
    passes = obs.groupby("_pass").agg(
        detected=("detected", "max"), clear=("clear", "max")
    )
    return int((passes.detected & passes.clear).sum()), int(passes.clear.sum())


def match_sites(
    nights: pd.DataFrame,
    myc: pd.DataFrame,
    match_m: float = 2000.0,
    overpasses: pd.DataFrame | None = None,
    copies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Greedily accept nearest pairs within radius, using each site once."""
    if overpasses is None:
        raise ValueError("overpasses are required for pooled clear-sky validation")
    per_site = pooled_observations(nights, overpasses)
    tree = cKDTree(_xyz(myc.lat, myc.lon))
    xyz = _xyz(per_site.lat, per_site.lon)
    pairs = []
    radius = 2 * np.sin(match_m / (2 * EARTH_RADIUS_M))
    for i, candidates in enumerate(tree.query_ball_point(xyz, radius)):
        for j in candidates:
            chord = np.linalg.norm(xyz[i] - tree.data[j])
            pairs.append((2 * EARTH_RADIUS_M * np.arcsin(min(chord / 2, 1)), i, j))
    used_i, used_j, selected = set(), set(), {}
    for distance, i, j in sorted(pairs):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            selected[i] = (distance, j)
    per_site["myc_dist_m"] = [
        selected.get(i, (np.nan, None))[0] for i in range(len(per_site))
    ]
    for col in (
        "flare_id",
        "regime",
        "bcm",
        "upstream_class",
        "country_iso",
        "n_det",
        "n_clear",
    ):
        per_site[f"myc_{col}"] = [
            myc.iloc[selected[i][1]][col] if i in selected else None
            for i in range(len(per_site))
        ]
    per_site["myc_det_freq"] = pd.to_numeric(per_site.myc_n_det) / pd.to_numeric(
        per_site.myc_n_clear
    ).replace(0, np.nan)
    rates = {}
    for i, (_, j) in selected.items():
        flare = myc.iloc[j]
        rates[i] = like_for_like_frequency(
            nights, overpasses, flare.lat, flare.lon, copies
        )
    per_site["lf_detections"] = [rates.get(i, (0, 0))[0] for i in range(len(per_site))]
    per_site["lf_clear_passes"] = [
        rates.get(i, (0, 0))[1] for i in range(len(per_site))
    ]
    per_site["lf_det_freq"] = per_site.lf_detections / per_site.lf_clear_passes.replace(
        0, np.nan
    )
    strict_rates = {
        i: like_for_like_frequency(
            nights,
            overpasses,
            myc.iloc[j].lat,
            myc.iloc[j].lon,
            copies,
            sensitivity=False,
        )
        for i, (_, j) in selected.items()
    }
    per_site["lf_strict_detections"] = [
        strict_rates.get(i, (0, 0))[0] for i in range(len(per_site))
    ]
    per_site["lf_strict_clear_passes"] = [
        strict_rates.get(i, (0, 0))[1] for i in range(len(per_site))
    ]
    per_site["lf_strict_det_freq"] = (
        per_site.lf_strict_detections
        / per_site.lf_strict_clear_passes.replace(0, np.nan)
    )
    return per_site


def validation_summary(sites: pd.DataFrame, match_m: float = 2000.0) -> dict:
    """Comparable totals use matched sites observed on at least one clear night."""
    from regions import region_of

    observed = sites[sites.nights_clear > 0]
    matched = observed[observed.myc_flare_id.notna()].copy()
    both = matched[(matched.myc_bcm.astype(float) > 0) & (matched.bcm_per_year > 0)]
    persistent = matched[
        (matched.myc_regime == "persistent") & matched.myc_det_freq.notna()
    ]
    lf_persistent = sites[
        sites.myc_flare_id.notna()
        & (sites.myc_regime == "persistent")
        & sites.lf_det_freq.notna()
        & sites.myc_det_freq.notna()
    ]
    matched["region"] = [region_of(r.lat, r.lon) for r in matched.itertuples()]
    result = dict(
        matched=len(matched),
        observed=len(observed),
        match_m=match_m,
        by_regime={
            regime: [int((g.nights_flaring > 0).sum()), len(g)]
            for regime, g in matched.groupby("myc_regime")
        },
        log_r=float(
            np.corrcoef(
                np.log10(both.myc_bcm.astype(float)), np.log10(both.bcm_per_year)
            )[0, 1]
        )
        if len(both) > 2
        else None,
        n_both=len(both),
        median_ratio=float(np.median(both.bcm_per_year / both.myc_bcm.astype(float)))
        if len(both)
        else None,
        unmatched_flaring=int(
            ((observed.nights_flaring > 0) & observed.myc_flare_id.isna()).sum()
        ),
        det_freq=dict(
            ours=float(persistent.det_freq.median()) if len(persistent) else None,
            myc=float(persistent.myc_det_freq.median()) if len(persistent) else None,
            n=len(persistent),
        ),
        det_freq_myc_strict=dict(
            ours=float(lf_persistent.lf_strict_det_freq.median()),
            n=int(lf_persistent.lf_strict_det_freq.notna().sum()),
            detections=int(lf_persistent.lf_strict_detections.sum()),
            clear_passes=int(lf_persistent.lf_strict_clear_passes.sum()),
        ),
        det_freq_like_for_like=dict(
            ours=float(lf_persistent.lf_det_freq.median())
            if len(lf_persistent)
            else None,
            myc=float(lf_persistent.myc_det_freq.median())
            if len(lf_persistent)
            else None,
            n=len(lf_persistent),
            detections=int(lf_persistent.lf_detections.sum()),
            clear_passes=int(lf_persistent.lf_clear_passes.sum()),
        ),
        regions=[
            [
                name,
                float(g.bcm_per_year.sum()),
                float(g.myc_bcm.astype(float).sum()),
                len(g),
            ]
            for name, g in matched.groupby("region")
        ],
    )

    result["det_freq_myc_detection_clears"] = result["det_freq_like_for_like"].copy()
    return result


def main(
    tag: str = "2026-10-29_2026-10-29",
    out_dir: Path = OUT_DIR,
    year: int = 2025,
    match_m: float = 2000.0,
    site_nights_csv: Path | None = None,
    overpasses_csv: Path | None = None,
) -> pd.DataFrame:
    """Validate one run against annual MYC regimes (default last available year)."""
    import json

    nights = pd.read_csv(site_nights_csv or out_dir / f"site_nights_{tag}.csv")
    over = pd.read_csv(overpasses_csv or out_dir / f"overpasses_{tag}.csv")
    copies_path = out_dir / f"copies_{tag}.csv"
    copies = pd.read_csv(copies_path) if copies_path.exists() else None
    sites = match_sites(nights, load_myc(year), match_m, over, copies)
    summary = validation_summary(sites, match_m)
    print(f"MYC25Q4 {year}; run {tag} (short-period rates vs annual reference)")
    print(json.dumps(summary, indent=2))
    for regime, (detected, observed) in summary["by_regime"].items():
        print(f"{regime}: {detected}/{observed} = {detected / observed:.1%}")
    sites.to_csv(out_dir / f"validation_myc25_{tag}.csv", index=False)
    (out_dir / f"validation_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2)
    )
    return sites


# %%
if __name__ == "__main__":
    main()
