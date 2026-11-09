"""Shared loaders for the Opus flare-out scripts. Run scripts from nightly-flaring/."""
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("outputs/flareout_research/scripts_opus/cache")
FIRST, LAST = "2026-08-09", "2026-11-04"


def zone(col: pd.Series) -> pd.Series:
    """VIIRS M-band aggregation zone: 0 nadir (3-agg), 1 mid (2-agg), 2 edge."""
    c = col.to_numpy()
    z = np.full(len(c), 2)
    z[((c >= 640) & (c < 1008)) | ((c >= 2192) & (c < 2560))] = 1
    z[(c >= 1008) & (c < 2192)] = 0
    return pd.Series(z, index=col.index)


def passes() -> pd.DataFrame:
    """Night passes (SZA >= 97) with outcome code.

    D clear detection, Z clear non-detection, C cloudy detection (lit), U cloudy blank.
    """
    p = CACHE / "passes.pkl"
    if p.exists():
        return pd.read_pickle(p)
    o = pd.read_pickle(CACHE / "over.pkl")
    o = o[(o.sza >= 97) & (o.night >= FIRST) & (o.night <= LAST)].copy()
    o["t"] = pd.to_datetime(o.time_utc)
    o["zone"] = zone(o.col)
    o["out"] = np.select(
        [o.detected & o.clear, ~o.detected & o.clear, o.detected & ~o.clear],
        ["D", "Z", "C"],
        "U",
    )
    o = o.sort_values(["site_id", "t"]).reset_index(drop=True)
    o.to_pickle(p)
    return o


def sites() -> pd.DataFrame:
    return pd.read_csv("outputs/gfw_oil_sites.csv")


def region(lat, lon):
    import sys

    sys.path.insert(0, ".")
    from regions import region_of

    return [region_of(a, b) for a, b in zip(lat, lon)]
