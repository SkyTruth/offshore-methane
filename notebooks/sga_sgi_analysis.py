# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sgi_sga_pd = pd.read_csv(r"..\SkyTruth\methane\sgi_sga_mask_sampled.csv")


# %%
def plot_sgi_sga(df, x_field="SGA", y_field="SGI", color_field="system_index"):
    unique_vals = df[color_field].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
    val_to_color = {val: colors[i] for i, val in enumerate(unique_vals)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for val, color in val_to_color.items():
        subset = df[df[color_field] == val]
        ax.scatter(subset[x_field], subset[y_field], label=val, color=color, alpha=0.5)

    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(f"{x_field} vs {y_field} by {color_field}")
    ax.legend(title=color_field, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid()
    plt.show()


# %%""
sgi_sga_pd = sgi_sga_pd[sgi_sga_pd["SGA"] > 0.0]
plot_sgi_sga(sgi_sga_pd, x_field="SGA", y_field="SGI", color_field="system_index")
masks = [
    "saturation_mask",
    "cloud_mask",
    "outlier_mask",
    "ndwi_mask",
    "sga_mask",
    "sgi_mask",
]
for field in masks:
    plot_sgi_sga(sgi_sga_pd, x_field="SGA", y_field="SGI", color_field=field)
sgi_sga_all_mask = sgi_sga_pd[
    (sgi_sga_pd["saturation_mask"] == 1)
    & (sgi_sga_pd["cloud_mask"] == 1)
    & (sgi_sga_pd["outlier_mask"] == 1)
    & (sgi_sga_pd["ndwi_mask"] == 1)
]

plot_sgi_sga(sgi_sga_all_mask, x_field="SGA", y_field="SGI", color_field="system_index")

# %%
sga_pixels_arr = sgi_sga_all_mask["SGA"].values
sgi_pixels_arr = sgi_sga_all_mask["SGI"].values

poly_coeffs = np.polyfit(sga_pixels_arr, sgi_pixels_arr, deg=3)

# # Turn into function
p = np.poly1d(poly_coeffs)
sgi_pixels_fit = p(sga_pixels_arr)

plt.scatter(sga_pixels_arr, sgi_pixels_arr, alpha=0.005)
# plt.scatter(np.linspace(10, 45), m * np.linspace(10, 45) + b)
plt.scatter(np.linspace(5, 45), p(np.linspace(5, 45)))

# %%
plt.scatter(sga_pixels_arr, sgi_pixels_arr - p(sga_pixels_arr), alpha=0.1)
