# %%
import ee
import time

# Authenticate and initialize
ee.Authenticate()
ee.Initialize()


# USER SETTINGS
NIR_THRESHOLD_DN = 1500  # Threshold for NIR band (B12) to detect flaring, adjust to higher threshold if needed (2500)
PIXEL_CLOUD_PROB_MAX = (
    25  # Maximum cloud probability for a pixel to be considered clear
)
ROI_CLOUD_FRACTION_MAX = 25  # Maximum cloud fraction for the ROI to be considered clear
ROI_RADIUS_M = 200  # Radius in meters for the region of interest around each centroid
start_date = "2015-06-27"  # Start date for Sentinel-2 data
end_date = "2025-08-01"

# ----------------------------------------------------------------------------
# tip and queue to more likely scenes. Prioritzation of site opportunities.
# INPUT: FeatureCollection asset from SkyTruth annual flaring CSV data https://skytruth.org/flaring/, go to the website and use drawing tool to download csv of flaring oil and gas infrastructure
# Save csv as an Earth Engine asset in your own account, e.g. "projects/benshostak-skytruth/assets/custom_flaring_data-3"

# ADD input for running directly from a csv file locally


table = ee.FeatureCollection(
    "projects/benshostak-skytruth/assets/custom_flaring_data-4"
)
points = table
print("points:", points.size().getInfo())
# option to process a local CSV file instead
# option to process individual lat/lon points
# point = ee.Geometry.Point(-91.4727, 28.97905)
# points = ee.FeatureCollection([ee.Feature(point)])
# ----------------------------------------------------------------------------


# # function to filter out rows where flaring only occured prior to s2 imagery
# def filter_recent_flaring(fc):
#     bcm_fields = [
#         "bcm_2016",
#         "bcm_2017",
#         "bcm_2018",
#         "bcm_2019",
#         "bcm_2020",
#         "bcm_2021",
#         "bcm_2022",
#         "bcm_2023",
#     ]

#     # Start with first field
#     combined_filter = ee.Filter.gt(bcm_fields[0], 0)

#     # Combine the rest using ee.Filter.or
#     for field in bcm_fields[1:]:
#         combined_filter = ee.Filter.Or(combined_filter, ee.Filter.gt(field, 0))

#     # Apply filter to FeatureCollection
#     return fc.filter(combined_filter)


# points_filtered = filter_recent_flaring(table)
# print("points 2016 - 2023:", points_filtered.size().getInfo())


# PROCESS CENTROIDS: remove duplicate infrastructure from flaring data
buffered100 = points.map(lambda f: f.buffer(100))
dissolved = buffered100.union()
dissolvedPolys = ee.FeatureCollection(
    ee.Geometry(dissolved.geometry())
    .geometries()
    .map(lambda g: ee.Feature(ee.Geometry(g)))
)
print("dissolvedPolys:", dissolvedPolys.size().getInfo())

centroidsAll = dissolvedPolys.map(lambda f: ee.Feature(f.geometry().centroid(1)))
print("centroidsAll:", centroidsAll.size().getInfo())

# ----------------------------------------------------------------------------

# SENTINEL‑2 IMAGE COLLECTION + CLOUD‑PROBABILITY JOIN
s2Sr = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterDate(start_date, end_date)
    .filterBounds(centroidsAll)
)

s2Cloud = (
    ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    .filterDate(start_date, end_date)
    .filterBounds(centroidsAll)
)

joined = s2Sr.linkCollection(s2Cloud, ["probability"])

# ----------------------------------------------------------------------------


# SCENE EVALUATION FUNCTION
# def evaluateScene(img, geom):
#     geom = ee.Geometry(geom)

#     maxB12 = (
#         img.select("B12")
#         .reduceRegion(reducer=ee.Reducer.max(), geometry=geom, scale=20, maxPixels=1e9)
#         .get("B12")
#     )
#     maxB12 = ee.Number(ee.Algorithms.If(maxB12, maxB12, 0))  # default to 0 if no data
#     isFlare = maxB12.gte(NIR_THRESHOLD_DN)

#     cloudFrac = (
#         img.select("probability")
#         .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=20, maxPixels=1e9)
#         .get("probability")
#     )
#     cloudFrac = ee.Number(ee.Algorithms.If(cloudFrac, cloudFrac, 100))
#     isClear = cloudFrac.lte(ROI_CLOUD_FRACTION_MAX)
#     isVenting = isClear.And(
#         isFlare.Not()
#     )  # venting = no flaring and low cloud fraction

#     ## EVENT_TYPE LOGIC
#     event_type = ee.Algorithms.If(
#         isFlare,
#         "flaring",
#         ee.Algorithms.If(
#             isClear.Not(),
#             "cloudy",
#             ee.Algorithms.If(isVenting, "possible_venting", "none"),
#         ),
#     )  # helps deal with over saturated pixels in B12 that are not flaring, otherwise would result in multiple classifications
#     # now all mutually exclusive
#     ##---------------------------------------------------------##

#     return ee.Feature(geom.centroid()).set(
#         {
#             "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
#             "system_index": img.get("system:index"),
#             "event_type": event_type,
#             # "is_flaring": isFlare,
#             # "is_cloudy": isClear.Not(),
#             # "is_venting": isVenting,
#             # "maxB12": maxB12,
#             # "cloudFrac": cloudFrac,
#         }
#     )


def evaluateScene(image, region):
    # Define point and buffer
    point = region.centroid()
    # region = point.buffer(buffer_m)

    # Select relevant bands from S2
    b3 = image.select("B3")  # Green
    b8 = image.select("B8")  # NIR
    b11 = image.select("B11")  # SWIR1
    b12 = image.select("B12")  # SWIR2
    b8a = image.select("B8A")  # Narrow NIR

    # --- Cloudiness (using S2 Cloud Probability dataset) ---
    system_index = image.get("system:index")

    cloud_img = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.equals("system:index", system_index))
        .first()
    )

    # Cloud probability band is "probability" (0–100)
    cloud_prob = cloud_img.select("probability")
    cloud_fraction = cloud_prob.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e8
    ).get("probability")

    # --- Structure Presence (via NDWI min) ---
    ndwi = b3.subtract(b8).divide(b3.add(b8))
    ndwi_stats = ndwi.reduceRegion(
        reducer=ee.Reducer.min(), geometry=region, scale=20, maxPixels=1e8
    )
    min_ndwi = ndwi_stats.get("B3")

    structure_present = ee.Algorithms.If(
        ee.Number(min_ndwi).lt(0),
        1,  # structure
        0,  # only ocean
    )

    # --- Flaring Detection (max difference B12 - B11) ---
    # flare_diff = b12.subtract(b11).rename("flare_diff")
    delta_sw = b12.subtract(b11)
    tai = delta_sw.divide(b8a).rename("TAI")

    # Get max value and pixel location
    flare_reduce = tai.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
        bestEffort=True,
    )
    max_flare_diff = flare_reduce.get("TAI")

    # Get coordinates of max flare pixel
    flare_mask = tai.eq(ee.Number(max_flare_diff))
    flare_coords = (
        flare_mask.reduceToVectors(
            scale=20,
            geometryType="centroid",
            maxPixels=1e8,
            bestEffort=True,
        )
        .filterBounds(region)
        .geometry()
        .coordinates()
    )
    flare_coords = ee.List(flare_coords)

    flare_present = ee.Algorithms.If(
        ee.Number(max_flare_diff).gt(0.15),  # threshold lowered
        1,
        0,
    )

    # Return as dictionary
    result = ee.Feature(point).set(
        {
            "system_index": system_index,
            "cloud_fraction": cloud_fraction,
            "min_ndwi": min_ndwi,
            "structure_present": structure_present,
            "max_TAI": max_flare_diff,
            "flare_present": flare_present,
            "flare_latlon": flare_coords,
        }
    )

    return result


# ----------------------------------------------------------------------------


# MAIN LOOP OVER CENTROIDS
# def process_point(pt):
#     roi = pt.geometry().buffer(ROI_RADIUS_M)
#     coord = pt.geometry().coordinates()

#     scene_eval = joined.filterBounds(roi).map(lambda img: evaluateScene(img, roi))

#     scene_eval = scene_eval.filter(ee.Filter.neq("event_type", "none"))

#     def make_feature(f):
#         return ee.Feature(
#             None,
#             {
#                 "lat": coord.get(1),
#                 "lon": coord.get(0),
#                 "start_date": f.get("date"),
#                 "end_date": f.get("date"),
#                 "event_type": f.get("event_type"),
#                 "system_index": f.get("system_index"),
#                 "note": ee.String(f.get("event_type")).cat(" event"),
#             },
#         )

#     return scene_eval.map(make_feature)


# Process point
def process_point(pt):
    roi = pt.geometry().buffer(ROI_RADIUS_M)
    # coord = pt.geometry().coordinates()

    scene_eval = joined.filterBounds(roi).map(lambda img: evaluateScene(img, roi))

    return scene_eval

    # scene_eval = scene_eval.filter(ee.Filter.neq("event_type", "none"))

    # def make_feature(f):
    #     return ee.Feature(
    #         None,
    #         {
    #             "system_index": system_index,
    #             "cloud_fraction": cloud_fraction,
    #             "min_ndwi": min_ndwi,
    #             "structure_present": structure_present,
    #             "max_TAI": max_flare_diff,
    #             "flare_present": flare_present,
    #             "flare_latlon": flare_coords,
    #         },
    #     )

    # return scene_eval.map(make_feature)


# Apply to all centroids and flatten result
results_list = centroidsAll.map(process_point).flatten()

# Final results FeatureCollection
results_fc = ee.FeatureCollection(results_list)
# print(results_fc.limit(10))

# ----------------------------------------------------------------------------

# Export (optional)
task = ee.batch.Export.table.toDrive(
    collection=results_fc,
    description="GFW_Flaring_Venting_Brazil_2015_2025_filtered",  # can change name here
    fileFormat="CSV",
)

task.start()

while task.active():
    print("Waiting for export... (status: {})".format(task.status()["state"]))
    time.sleep(30)

print("Done. Final status:", task.status())

# add downloaded CSV from Google Drive to data folder

# %%
## USE THIS CODE TO PROCESS THE EXPORTED CSV FILE FROM EARTH ENGINE (S2_Flaring_Venting_Export_test_2.csv)
# This code processes the exported CSV file to create venting ranges based on flaring data around venting or cloudy dates.
import pandas as pd  # noqa
from datetime import timedelta  # noqa

# Load and prepare data
df = pd.read_csv(
    "/Users/bshostak/Documents/GitHub/offshore-methane/data/S2_Flaring_Venting_Brazil_2015_2025_filtered.csv",
    parse_dates=["start_date", "end_date"],
)

# Drop duplicates
df = df.drop_duplicates(
    subset=["start_date", "end_date", "lat", "lon"]
)  # drops duplicate rows from overlapping granules and repeated rows from the same lat/lon/date
df = df.sort_values(by=["lat", "lon", "start_date"]).reset_index(drop=True)
df["latlon"] = list(zip(df["lat"], df["lon"]))

all_ranges = []

# Process each lat/lon group
for latlon, group in df.groupby("latlon"):
    lat, lon = latlon
    group = group.sort_values("start_date").reset_index(drop=True)
    used_indices = set()

    for idx, row in group.iterrows():
        if idx in used_indices or row["event_type"] != "possible_venting":
            continue

        # Expand backward: include previous cloudy days
        start_idx = idx
        for i in range(idx - 1, -1, -1):
            evt = group.loc[i, "event_type"]
            if evt == "cloudy":
                start_idx = i
                used_indices.add(i)
            elif evt == "flaring":
                break
            else:
                break  # stop if we see another venting or something else

        # Expand forward: include all cloudy and possible_venting days
        end_idx = idx
        flare_found = False
        for i in range(idx + 1, len(group)):
            evt = group.loc[i, "event_type"]
            if evt == "flaring":
                flare_found = True
                break
            elif evt in ["cloudy", "possible_venting"]:
                end_idx = i
                used_indices.add(i)
            else:
                break

        # Final range
        start_date = group.loc[start_idx, "start_date"]
        end_date = (
            group.loc[i, "start_date"] - timedelta(days=1)
            if flare_found
            else group.loc[end_idx, "start_date"]
        )  # removes one day from the end date if flaring is found

        used_indices.update(range(start_idx, end_idx + 1))

        all_ranges.append(
            {
                "lat": lat,
                "lon": lon,
                "start": start_date,  # change name to "start" to match orchestrator.py
                "end": end_date,  # change name to "end " to match orchestrator.py
                "note": "possible venting from flaring dataset",
            }
        )

# Final dataframe
venting_ranges_df = pd.DataFrame(all_ranges).sort_values(by=["lat", "lon", "start"])
venting_ranges_df.to_csv(
    "/Users/bshostak/Documents/GitHub/offshore-methane/data/Brazil_Venting_Ranges_20150627_20250801_filtered.csv",
    index=False,
)  # replace with your desired path

# print(venting_ranges_df)


# %%

# count number of rows in venting_ranges_df
print("Number of venting ranges:", len(venting_ranges_df))
# print last 10 rows of venting_ranges_df
print(venting_ranges_df.tail(10))

# %%
import pandas as pd  # noqa

# Load existing sites
sites_df = pd.read_csv(
    "/Users/bshostak/Documents/GitHub/offshore-methane/data/sites.csv"
)

# Strip time, keep only date and convert to string format "YYYY-MM-DD" for orchestrator.py
venting_ranges_df["start"] = pd.to_datetime(venting_ranges_df["start"]).dt.strftime(
    "%Y-%m-%d"
)
venting_ranges_df["end"] = pd.to_datetime(venting_ranges_df["end"]).dt.strftime(
    "%Y-%m-%d"
)


# Append rows, keeping all columns
combined_df = pd.concat([sites_df, venting_ranges_df], ignore_index=True)

# Save to CSV
combined_df.to_csv(
    "/Users/bshostak/Documents/GitHub/offshore-methane/data/sites_gulf_of_thailand.csv",
    index=False,
)

# %%
