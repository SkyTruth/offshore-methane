# %%
"""Compare a site-night run with EOG's MYC25Q4 annual upstream flare sites.

Downloads ``annual_flare_site_records.csv`` (CC BY-NC 4.0) from Zenodo into
``outputs/`` once, matches each GFW site to the nearest MYC25Q4 site within
``match_m``, and reports detection rates and volume agreement by MYC regime.
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


def match_sites(
    nights: pd.DataFrame, myc: pd.DataFrame, match_m: float
) -> pd.DataFrame:
    """Per GFW site: 7-night summary plus the nearest MYC site within match_m."""
    per_site = (
        nights.groupby(["site_id", "lat", "lon"])
        .agg(
            nights_clear=("n_clear", lambda s: int((s > 0).sum())),
            nights_flaring=("n_detected", lambda s: int((s > 0).sum())),
            rh_sum=("rh_mw", lambda s: float(np.nansum(s))),
            nights_with_value=("rh_mw", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    per_site["bcm_per_year"] = (
        0.0116 * per_site["rh_sum"] / per_site["nights_with_value"].replace(0, np.nan)
    )
    chord, j = cKDTree(_xyz(myc.lat, myc.lon)).query(_xyz(per_site.lat, per_site.lon))
    dist = 2 * EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2, 0, 1))
    hit = dist <= match_m
    per_site["myc_dist_m"] = np.where(hit, dist, np.nan)
    for col in ("flare_id", "regime", "bcm", "upstream_class", "country_iso"):
        per_site[f"myc_{col}"] = np.where(hit, myc[col].to_numpy()[j], None)
    return per_site


def main(
    site_nights_csv: Path = OUT_DIR / "site_nights_2026-10-29_2026-11-04.csv",
    year: int = 2025,
    match_m: float = 1000.0,
) -> pd.DataFrame:
    nights = pd.read_csv(site_nights_csv)
    sites = match_sites(nights, load_myc(year), match_m)
    observed = sites[sites["nights_clear"] > 0]
    matched = observed[observed["myc_flare_id"].notna()]
    print(
        f"{len(sites)} GFW sites, {len(observed)} with >=1 clear night, "
        f"{len(matched)} of those within {match_m:.0f} m of a MYC25Q4 {year} site"
    )
    flaring = observed["nights_flaring"] > 0
    print(
        f"flaring on >=1 night: {int(flaring.sum())} sites; "
        f"{int((flaring & observed['myc_flare_id'].notna()).sum())} of them match MYC"
    )
    table = (
        matched.assign(detected=matched["nights_flaring"] > 0)
        .groupby("myc_regime")["detected"]
        .agg(["sum", "size"])
    )
    table["rate"] = table["sum"] / table["size"]
    print("MYC regime -> GFW sites with a detection on >=1 clear night")
    print(table.to_string())
    both = matched[
        (matched["myc_bcm"].astype(float) > 0) & (matched["bcm_per_year"] > 0)
    ]
    if len(both) > 2:
        x = np.log10(both["myc_bcm"].astype(float))
        y = np.log10(both["bcm_per_year"])
        print(
            f"volume, sites positive in both (n={len(both)}): "
            f"log10 r={np.corrcoef(x, y)[0, 1]:.2f}, "
            f"median ratio ours/MYC={np.median(10 ** (y - x)):.2f}"
        )
    sites.to_csv(Path(site_nights_csv).with_name("validation_myc25.csv"), index=False)
    return sites


# %%
if __name__ == "__main__":
    main()
