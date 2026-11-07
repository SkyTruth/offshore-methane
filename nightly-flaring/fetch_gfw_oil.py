# %%
"""Fetch GFW offshore fixed-infrastructure points and keep active oil sites.

Same access pattern as cerulean-cloud
(``cerulean_cloud/cloud_function_asa/utils/analyzer.py``): the whole layer comes
back as one z0 Mapbox Vector Tile. Needs ``INFRA_API_TOKEN`` in the environment
or in the repo-root ``.env``.

Writes ``outputs/gfw_oil_points.csv`` (one row per structure) and
``outputs/gfw_oil_sites.csv`` (structures merged within ``radius_m``).
"""

from __future__ import annotations

import os
from pathlib import Path

import mapbox_vector_tile
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
GFW_URL = (
    "https://gateway.api.globalfishingwatch.org/v3/datasets/"
    "public-fixed-infrastructure-filtered:latest/context-layers/0/0/0"
)
EARTH_RADIUS_M = 6_371_000.0


def fetch_infrastructure(token: str) -> pd.DataFrame:
    """Return every GFW fixed-infrastructure point (oil, wind, unknown)."""
    resp = requests.get(
        GFW_URL, headers={"Authorization": f"Bearer {token}"}, timeout=120
    )
    resp.raise_for_status()
    tile = mapbox_vector_tile.decode(resp.content)
    df = pd.DataFrame([f["properties"] for f in tile["main"]["features"]])
    for col in ("structure_start_date", "structure_end_date"):
        ms = pd.to_numeric(df[col].replace("", np.nan))
        df[col] = pd.to_datetime(ms, unit="ms")
    return df


def active_oil(df: pd.DataFrame) -> pd.DataFrame:
    """Oil structures with no end date or an end date in the latest month."""
    oil = df[df["label"] == "oil"]
    latest = oil["structure_end_date"].max()
    keep = oil["structure_end_date"].isna() | (oil["structure_end_date"] >= latest)
    return oil[keep].copy()


def cluster_sites(points: pd.DataFrame, radius_m: float = 750.0) -> pd.DataFrame:
    """Merge structures closer than ``radius_m`` (default: one VIIRS M-band pixel)."""
    xy = np.deg2rad(points[["lat", "lon"]].to_numpy())
    labels = DBSCAN(
        eps=radius_m / EARTH_RADIUS_M, min_samples=1, metric="haversine"
    ).fit_predict(xy)
    return (
        points.assign(site_id=labels)
        .groupby("site_id")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            n_structures=("structure_id", "size"),
            structure_ids=("structure_id", lambda s: " ".join(map(str, sorted(s)))),
        )
        .reset_index()
    )


def main(radius_m: float = 750.0, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Download the GFW layer and write active oil points and merged sites."""
    load_dotenv(HERE.parent / ".env")
    token = os.environ.get("INFRA_API_TOKEN")
    if not token:
        raise RuntimeError("Set INFRA_API_TOKEN (GFW gateway API token).")
    df = fetch_infrastructure(token)
    oil = active_oil(df)
    sites = cluster_sites(oil, radius_m)
    out_dir.mkdir(parents=True, exist_ok=True)
    oil.to_csv(out_dir / "gfw_oil_points.csv", index=False)
    sites.to_csv(out_dir / "gfw_oil_sites.csv", index=False)
    print(
        f"{len(df)} structures, {len(oil)} active oil, "
        f"{len(sites)} sites at {radius_m:.0f} m"
    )
    return sites


# %%
if __name__ == "__main__":
    main()
