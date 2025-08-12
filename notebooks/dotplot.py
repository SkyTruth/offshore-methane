# %%
#!/usr/bin/env python3
"""
dotplot_bulk.py - build the full dot-plot table on the EE server and export once.

Workflow
========
1. **Pre-flight check** - for every Sentinel-2 `system:index` in `SCENES`,
   ensure that the corresponding SGA asset already exists (and create it
   if missing).  This guarantees that subsequent SGI computation will not
   block on SGA generation tasks.
2. Build one SGI + SGA image per scene (20 m scale).
3. Random-sample `PER_SCENE_PIXELS` points (or within an ROI).
4. Flatten all per-scene FeatureCollections into a single one.
5. Export the whole table as CSV to Cloud Storage (EE automatically splits
   into part-files).

CSV columns: ``sid, B8A, B12, SGI, SGA``
"""

from __future__ import annotations

from typing import List, Optional

import ee
from tqdm import tqdm

import offshore_methane.config as cfg
from offshore_methane.sga import ensure_sga_asset

ee.Initialize(opt_url=getattr(cfg, "EE_ENDPOINT", None))

# %%
OCEAN = ee.FeatureCollection("users/christian/General_Data/Marine/simplifiedOceans")

scenes = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterDate("2025-01-01", "2025-01-02")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .filterBounds(OCEAN)
    .randomColumn("rand", 0)
    .sort("rand")
    .limit(1000)
    .aggregate_array("system:index")
    .getInfo()
)
print(len(scenes))
# %%
import csv  # noqa

import offshore_methane.config as cfg  # noqa

with open(cfg.SITES_CSV, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)[:47]

scenes = [row["system_index"].split(";") for row in rows]
scenes = [item for sublist in scenes for item in sublist]
print(len(scenes))

# %%

# ────────────────────────────────────────────────────────────────
#  Parameters
# ────────────────────────────────────────────────────────────────
SCENES: List[str] = scenes  # list of Sentinel-2 system:index strings
ROI_RADIUS_M: Optional[int] = 10_000  # None → whole footprint
SCALE = 20  # metres
TILESCALE = 8  # sampling inner parallelism
EXPORT_DESC = "dotplot_bulk47"
BUCKET = cfg.EXPORT_PARAMS["bucket"]
PREFIX = "dotplot_bulk/table"  # gs://<bucket>/<PREFIX>-*.csv
PER_SCENE_PIXELS = 20_000
SAMPLE_SEED = 42
SELECTORS = ["sid", "B8A", "B12", "SGI", "SGA"]

# ────────────────────────────────────────────────────────────────
#  Ensure all SGAs first  ← NEW
# ────────────────────────────────────────────────────────────────
print("⧗ Ensuring SGA assets exist for all scenes…")
for sid in tqdm(SCENES, desc="Checking/creating SGA"):
    # This will no-op if the asset already exists
    ensure_sga_asset(sid, **cfg.EXPORT_PARAMS)
print("✓ All SGA assets verified.\n")


# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────
def make_img(sid: str) -> ee.Image:
    """Return an ee.Image with bands B8A, B12, SGI, SGA."""
    s2 = ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")
    b_vis = (
        s2.select("B2")
        .add(s2.select("B3"))
        .add(s2.select("B4"))
        .divide(3)
        .rename("B_vis")
    )
    sgi = s2.addBands(b_vis).normalizedDifference(["B8A", "B_vis"]).rename("SGI")

    # Re-invoke to fetch the asset path (cheap now that it already exists)
    sga_src, _ = ensure_sga_asset(sid, **cfg.EXPORT_PARAMS)
    sga = (
        ee.Image.loadGeoTIFF(sga_src)
        if isinstance(sga_src, str) and sga_src.startswith("gs://")
        else ee.Image(sga_src)
    ).rename("SGA")

    return s2.select(["B8A", "B12"]).addBands([sgi, sga]).toFloat()


def sample_img(img: ee.Image, sid: str) -> ee.FeatureCollection:
    """Return a random sample of PER_SCENE_PIXELS features with desired props only."""
    geom = img.geometry()
    if ROI_RADIUS_M:
        geom = geom.centroid().buffer(ROI_RADIUS_M)

    fc = img.sample(
        region=geom,
        scale=SCALE,
        tileScale=TILESCALE,
        numPixels=PER_SCENE_PIXELS,
        seed=SAMPLE_SEED,
        geometries=False,
    ).map(lambda ft: ft.set("sid", sid))

    return fc.select(SELECTORS)


# ────────────────────────────────────────────────────────────────
#  Build the mega-collection entirely server-side
# ────────────────────────────────────────────────────────────────
per_scene_fcs = []
for sid in tqdm(SCENES, desc="Sampling"):
    per_scene_fcs.append(sample_img(make_img(sid), sid))

bulk_fc = ee.FeatureCollection(per_scene_fcs).flatten()

# ────────────────────────────────────────────────────────────────
#  Single export task
# ────────────────────────────────────────────────────────────────
task = ee.batch.Export.table.toCloudStorage(
    collection=bulk_fc,
    description=EXPORT_DESC,
    bucket=BUCKET,
    fileNamePrefix=PREFIX,
    fileFormat="CSV",
    selectors=SELECTORS,  # strips system:index & .geo
)

task.start()
print(f"Started EE export task {task.id} ({len(SCENES)} scenes).")
print("⌚  Monitor progress in the EE Tasks tab or with:")
print(f"    earthengine task info {task.id}")


# %%

import pandas as pd  # noqa

df = pd.read_csv("../data/18M_dotplot.csv")
df["sid_hash"] = df["sid"].apply(lambda x: hash(x))
print(len(df))
# %%
small_df = df
small_df = small_df.sample(frac=0.1, random_state=42)
small_df = small_df[small_df["b_vis"] < 6000]
# small_df = small_df[small_df["B8A"] < 6000]
# small_df = small_df[small_df["SGA"] < 40]
small_df.plot.scatter(
    x="SGA",
    y="SGI",
    alpha=0.01,
    s=10,
    c=small_df["sid_hash"],
    cmap="rainbow",
    figsize=(20, 20),
    colorbar=False,
)
plt.title("Removed b_vis > 6000")  # noqa
# random sample of df
random_df = df.sample(frac=0.01)
# %%
# add chart title
# AttributeError: Rectangle.set() got an unexpected keyword argument 'title'
import matplotlib.pyplot as plt  # noqa

df["b_vis"].hist(bins=1000, range=(0, 1000))
plt.title("B_vis Histogram")


# %%
df["b_vis"] = -df["B8A"] * (df["SGI"] - 1) / (1 + df["SGI"])

# %%
