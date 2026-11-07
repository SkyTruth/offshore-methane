# %%
"""Group oil sites by the VIIRS night granules that cover them on one date.

Footprints come from NASA CMR for the 6-minute geolocation granules (VNP03MOD,
VJ103MOD, VJ203MOD); CMR search needs no login. Each footprint is split into
``N_SLICES`` along-track slices as a stand-in for NOAA's ~86 s SDR granules.
CMR lists the corners as start-A, start-B, end-B, end-A, so the slices
interpolate between the start edge (c0-c1) and the end edge (c3-c2).

Granules that cross a pole are not handled, so sites above ~80 degrees can be
missed. Needs ``outputs/gfw_oil_sites.csv`` from ``fetch_gfw_oil.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely
from shapely import STRtree
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
PRODUCTS = {"SNPP": "VNP03MOD", "NOAA-20": "VJ103MOD", "NOAA-21": "VJ203MOD"}
N_SLICES = 4  # one 6-min NASA granule ~= 4.2 NOAA 86-s SDR granules
PAGE_SIZE = 2000


def cmr_granules(short_name: str, date: str) -> list[dict]:
    """CMR granule entries flagged night or both (day/night) for one UTC date."""
    entries: list[dict] = []
    for flag in ("night", "both"):
        page = 1
        while True:
            resp = requests.get(
                CMR_URL,
                params={
                    "short_name": short_name,
                    "temporal": f"{date}T00:00:00Z,{date}T23:59:59Z",
                    "day_night_flag": flag,
                    "page_size": PAGE_SIZE,
                    "page_num": page,
                },
                timeout=60,
            )
            resp.raise_for_status()
            batch = resp.json()["feed"]["entry"]
            entries += batch
            if len(batch) < PAGE_SIZE:
                break
            page += 1
    return entries


def corners(granule: dict) -> tuple[np.ndarray, bool]:
    """Footprint corners as (lat, lon), longitudes shifted to 0-360 if it wraps."""
    c = np.array(granule["polygons"][0][0].split(), dtype=float)[:8].reshape(4, 2)
    wraps = bool(np.ptp(c[:, 1]) > 180)
    if wraps:
        c[:, 1] = np.where(c[:, 1] < 0, c[:, 1] + 360, c[:, 1])
    return c, wraps


def slices(c: np.ndarray, n: int = N_SLICES) -> list[Polygon]:
    """Split a footprint into ``n`` along-track polygons (lon, lat order)."""
    out = []
    for k in range(n):
        a, b = k / n, (k + 1) / n
        quad = np.array(
            [
                c[0] + (c[3] - c[0]) * a,
                c[1] + (c[2] - c[1]) * a,
                c[1] + (c[2] - c[1]) * b,
                c[0] + (c[3] - c[0]) * b,
            ]
        )
        out.append(Polygon(quad[:, ::-1]))
    return out


def site_slices(sites: pd.DataFrame, granules: list[dict]) -> pd.DataFrame:
    """One row per (granule slice, site list) for slices holding >= 1 site."""
    lon = sites["lon"].to_numpy()
    lat = sites["lat"].to_numpy()
    trees = {
        False: STRtree(shapely.points(lon, lat)),
        True: STRtree(shapely.points(np.where(lon < 0, lon + 360, lon), lat)),
    }
    rows = []
    for g in granules:
        if not g.get("polygons"):
            continue
        c, wraps = corners(g)
        t0 = pd.Timestamp(g["time_start"])
        dt = (pd.Timestamp(g["time_end"]) - t0) / N_SLICES
        for k, poly in enumerate(slices(c)):
            hit = trees[wraps].query(poly, predicate="contains")
            if len(hit) == 0:
                continue
            rows.append(
                {
                    "granule": g["producer_granule_id"],
                    "day_night": g.get("day_night_flag"),
                    "slice": k,
                    "slice_start": t0 + k * dt,
                    "slice_end": t0 + (k + 1) * dt,
                    "n_sites": len(hit),
                    "site_ids": " ".join(map(str, sorted(sites["site_id"].iloc[hit]))),
                }
            )
    return pd.DataFrame(rows)


def main(
    date: str = "2026-10-25",
    sites_csv: Path = OUT_DIR / "gfw_oil_sites.csv",
    out_dir: Path = OUT_DIR,
) -> pd.DataFrame:
    """Write ``granule_sites_<date>.csv`` and print per-satellite coverage."""
    sites = pd.read_csv(sites_csv)
    tables = []
    for sat, short_name in PRODUCTS.items():
        granules = cmr_granules(short_name, date)
        table = site_slices(sites, granules).assign(satellite=sat)
        seen = set(" ".join(table["site_ids"]).split()) if len(table) else set()
        print(
            f"{sat}: {len(granules)} night/both 6-min granules, "
            f"{table['granule'].nunique() if len(table) else 0} with sites, "
            f"{len(table)}/{len(granules) * N_SLICES} slices with sites, "
            f"{len(seen)}/{len(sites)} sites covered"
        )
        tables.append(table)
    out = pd.concat(tables, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"granule_sites_{date}.csv", index=False)
    return out


# %%
if __name__ == "__main__":
    main()
