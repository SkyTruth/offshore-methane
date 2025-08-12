# %%
# global_SGA_csv_exporter.py
"""
Fast Sentinel-2 sunglint-alpha export (~3 minutes per *year*, ocean-only).

Creates two Google-Drive files per year:

    1.  alpha_stats_<YYYY>.geojson   -  one polygon per MGRS tile +
        alpha_min / alpha_max / alpha_mean

    2.  alpha_timeseries_<YYYY>_<MM>.csv   -  one line per granule
        (tile_id , datetime , alpha)   with *no* geometry
        → One CSV per month so EE can export 12 tasks in parallel.

Everything else (formula, ocean mask) is identical to the previous script.
"""

import math
from typing import List

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

import offshore_methane.config as cfg

# ----------------------------------------------------------------------
# Static assets & constants
# ----------------------------------------------------------------------
OCEAN = ee.FeatureCollection("users/christian/General_Data/Marine/simplifiedOceans")
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
DRIVE_DIR = cfg.EXPORT_PARAMS.get("drive_folder", "S2_Sunglint")


# ----------------------------------------------------------------------
# Angle helpers
# ----------------------------------------------------------------------
def _rad(img: ee.Image, prop: str) -> ee.Number:
    return ee.Number(img.get(prop)).multiply(DEG2RAD)


def sunglint_alpha(img: ee.Image) -> ee.Image:
    """Attach constant alpha (deg) to a Sentinel-2 granule (image property)."""
    # — Solar
    th0 = _rad(img, "MEAN_SOLAR_ZENITH_ANGLE")
    ph0 = _rad(img, "MEAN_SOLAR_AZIMUTH_ANGLE")

    # — Sensor (band-averaged)
    th = (
        _rad(img, "MEAN_INCIDENCE_ZENITH_ANGLE_B11")
        .add(_rad(img, "MEAN_INCIDENCE_ZENITH_ANGLE_B12"))
        .multiply(0.5)
    )
    ph = (
        _rad(img, "MEAN_INCIDENCE_AZIMUTH_ANGLE_B11")
        .add(_rad(img, "MEAN_INCIDENCE_AZIMUTH_ANGLE_B12"))
        .multiply(0.5)
    )

    A = th0.add(th).cos().add(th0.subtract(th).cos())
    B = th0.add(th).cos().subtract(th0.subtract(th).cos())
    C = ph0.subtract(ph).cos()
    arg = A.add(B.multiply(C)).multiply(0.5).max(-1).min(1)

    return ee.Image(img.set("alpha_deg", arg.acos().multiply(RAD2DEG)))


# ----------------------------------------------------------------------
# Collection builders
# ----------------------------------------------------------------------
_REQUIRED_PROPS = [
    "MEAN_INCIDENCE_ZENITH_ANGLE_B11",
    "MEAN_INCIDENCE_AZIMUTH_ANGLE_B11",
    "MEAN_INCIDENCE_ZENITH_ANGLE_B12",
    "MEAN_INCIDENCE_AZIMUTH_ANGLE_B12",
    "MEAN_SOLAR_ZENITH_ANGLE",
    "MEAN_SOLAR_AZIMUTH_ANGLE",
]


def s2_collection(start: str, end: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(OCEAN)
        .filter(ee.Filter.notNull(_REQUIRED_PROPS))
        .map(sunglint_alpha)
        .select([])  # keep *no* bands → tiny Features
    )


# ----------------------------------------------------------------------
# Time-series record builder
# ----------------------------------------------------------------------
def timeseries_records(col: ee.ImageCollection) -> ee.FeatureCollection:
    """
    One Feature (no geometry) per granule:   {tile_id , datetime , alpha}
    """

    def _to_row(img: ee.Image) -> ee.Feature:
        return ee.Feature(
            None,  # ← NO geometry → slender CSV
            {
                "tile_id": img.get("MGRS_TILE"),
                "datetime": img.date(),
                "alpha": img.get("alpha_deg"),
            },
        )

    return ee.FeatureCollection(col.map(_to_row))


# ----------------------------------------------------------------------
# Export wrappers
# ----------------------------------------------------------------------
def _drive_export(
    fc: ee.FeatureCollection, descr: str, fmt: str, selectors: List[str] | None = None
) -> ee.batch.Task:
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=descr,
        folder=DRIVE_DIR,
        fileFormat=fmt,
        selectors=selectors,
    )
    task.start()
    print(f"Started → {descr}   ({task.id})")
    return task


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def export_year(year: int) -> None:
    start, end = f"{year}-01-01", f"{year + 1}-01-01"

    # 1 · All S-2 granules (ocean-only)
    col = s2_collection(start, end)

    # 2 · Long table, split by month  (12 parallel Drive tasks)
    for m in range(1, 13):
        m_start = ee.Date.fromYMD(year, m, 1)
        m_end = m_start.advance(1, "month")
        month = col.filterDate(m_start, m_end)

        _drive_export(
            timeseries_records(month),
            descr=f"S2_tiles_alpha_timeseries_{year}_{m:02d}",
            fmt="CSV",
            selectors=["tile_id", "datetime", "alpha"],
        )


# %%
def main() -> None:
    export_year(2023)  # change / loop as needed


if __name__ == "__main__":
    main()

# %%
year = 2023

df_whole = pd.read_csv(f"../data/{year} SGA raw.csv", parse_dates=["datetime"])
df = df_whole
print(df.head(1))
print(f"Number of captures: {len(df)}")
print(f"Number of footprints: {len(df['tile_id'].unique())}")

# %%
df.hist(["alpha"], bins=1000)

# %%
df.hist(["datetime"], bins=1000)

# %%
# Histogram of unique values of tile_id
df["tile_id"].value_counts().plot(kind="line")

# %%
# plot alpha vs datetime scatter plot
df.plot(x="datetime", y="alpha", kind="scatter", alpha=0.01, s=0.1)

# %%
# For all tile_ids on a single plot, plot the datetime of the higest alpha in one color, and the datetime of the lowest in another color
g = df.groupby("tile_id")["alpha"]
idx_max = g.idxmax()  # row-indices of the highest alpha per tile
idx_min = g.idxmin()  # row-indices of the lowest  alpha per tile
high_pts = df.loc[idx_max]
low_pts = df.loc[idx_min]

fig, ax = plt.subplots(figsize=(9, 5))
# ax.scatter(high_pts['datetime'], high_pts['alpha'],
#            s=6,  c='red',   label='Highest alpha per tile')
ax.scatter(
    low_pts["datetime"],
    low_pts["alpha"],
    label="Lowest alpha per tile",
    alpha=0.5,
    s=0.1,
)
ax.set_xlabel("datetime")
ax.set_ylabel("alpha  (deg)")
ax.legend()
fig.tight_layout()
plt.show()

# %%

# 1 · pick 10 % of tile_ids at random
rng = np.random.default_rng(seed=6)
all_tiles = np.array(df["tile_id"].unique())
sample_size = max(1, int(0.001 * len(all_tiles)))
sample_tiles = rng.choice(all_tiles, size=sample_size, replace=False).tolist()

sample_df = df[df["tile_id"].isin(sample_tiles)]

# ------------------------------------------------------------------
# *** keep prefix length consistent everywhere ***
# ------------------------------------------------------------------
PREFIX_LEN = 2  # ← choose 1, 2, … but ONE value

prefix = sample_df["tile_id"].str[:PREFIX_LEN]
uniq_pref = np.sort(prefix.unique())

# 2 · make a colour for each prefix   (new API)
cmap = colormaps.get_cmap("tab20")
colour_map = {p: cmap(i) for i, p in enumerate(uniq_pref)}

# 3 · overlay scatter
fig, ax = plt.subplots(figsize=(9, 5))

for p, grp in sample_df.groupby(prefix):  # uses SAME prefix variable
    ax.scatter(
        grp["datetime"], grp["alpha"], alpha=0.5, s=10, color=colour_map[p], label=p
    )

# ax.legend(title="prefix",
#           bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
fig.tight_layout()
plt.show()

# %%

# generate a new df with rows for unique tile_id, and the highest and lowest and mean alpha for that tile_id
# For each unique tile_id, compute the min, max, and mean alpha
tile_stats = (
    df.groupby("tile_id")["alpha"]
    .agg(alpha_min="min", alpha_max="max", alpha_mean="mean", count="count")
    .reset_index()
)
tile_stats.to_csv(f"../data/{year} SGA stats.csv", index=False)

# %%
# cumulative distribution function of alpha_min, alpha_max, alpha_mean
tile_stats.hist(["alpha_min", "alpha_max", "alpha_mean"], bins=100)
tile_stats.hist(["alpha_min", "alpha_max", "alpha_mean"], bins=100, cumulative=True)


# %%
sub = tile_stats[tile_stats["alpha_max"] < 40]
sub.hist(
    ["alpha_min", "alpha_max", "alpha_mean", "count"],
    bins=1000,
    cumulative=True,
)

# %%
