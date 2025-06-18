"""
Local Three-P plume-polygon extraction (raster → GeoJSON → EE FC).
Byte-identical port of your original implementation.
"""

from __future__ import annotations

import ee
import geojson
import numpy as np
from affine import Affine
from rasterio.features import geometry_mask, shapes
from rasterio.transform import from_origin
from scipy.ndimage import label
from shapely.geometry import MultiPolygon, shape


# ------------------------------------------------------------------
def instances_from_probs(
    raster,
    p1,
    p2,
    p3,
    transform=Affine.identity(),
    addl_props={},
    discard_edge_polygons_buffer=0.005,
):
    """Return GeoJSON features for independent plumes."""

    def overlap_percent(a, b):
        if not a.intersects(b):
            return 0.0
        elif a.within(b):
            return 1.0
        else:
            return a.intersection(b).area / a.area

    nodata_mask = raster == 0

    # choose operator depending on sign
    if p1 < 0 and p2 < 0 and p3 < 0:
        p1_islands, _ = label(raster <= p1)
        p2_mask = raster <= p2
        p3_mask = raster <= p3
    else:
        p1_islands, _ = label(raster >= p1)
        p2_mask = raster >= p2
        p3_mask = raster >= p3

    p1_p3_labels = np.unique(p1_islands[p3_mask])
    p1_p3_mask = np.isin(p1_islands, p1_p3_labels)
    p1_p2_p3_mask = p1_p3_mask & p2_mask
    combined_raster = p1_p2_p3_mask * p1_islands

    scene_edge = MultiPolygon(
        shape(geom)
        for geom, value in shapes(
            nodata_mask.astype(np.uint8), mask=nodata_mask, transform=transform
        )
        if value == 1
    ).buffer(discard_edge_polygons_buffer)

    label_geometries = {}
    for geom, lbl in shapes(
        combined_raster, mask=combined_raster > 0, transform=transform
    ):
        poly = shape(geom)
        if overlap_percent(poly, scene_edge) <= 0.5:
            label_geometries.setdefault(lbl, []).append(poly)

    features = []
    for polys in label_geometries.values():
        mp = MultiPolygon(polys)
        geom_mask = geometry_mask(
            [mp], out_shape=raster.shape, transform=transform, invert=True
        )
        masked = raster[geom_mask]
        features.append(
            geojson.Feature(
                geometry=mp,
                properties={
                    "mean_conf": float(np.mean(masked)),
                    "median_conf": float(np.median(masked)),
                    "max_conf": float(np.max(masked)),
                    "machine_confidence": float(np.median(masked)),
                    **addl_props,
                },
            )
        )
    return features


# ------------------------------------------------------------------
def plume_polygons_three_p(
    R_img: ee.Image, region: ee.Geometry, p1: float, p2: float, p3: float
) -> ee.FeatureCollection:
    """Run Three-P locally on 5 km AOI and push polygons back to EE."""
    info = R_img.sampleRectangle(region=region, defaultValue=0).getInfo()
    try:
        grid = np.array(info["properties"]["MBSP"], dtype=np.float32)
    except KeyError as e:
        raise KeyError(
            f"MBSP band missing in sampleRectangle result; "
            f"available keys={list(info.get('properties', {}).keys())}"
        ) from e
    grid[np.isnan(grid)] = 0.0

    coords = region.bounds().coordinates().getInfo()[0]
    minx, maxx = min(c[0] for c in coords), max(c[0] for c in coords)
    miny, maxy = min(c[1] for c in coords), max(c[1] for c in coords)
    h, w = grid.shape
    transform = from_origin(minx, maxy, (maxx - minx) / w, (maxy - miny) / h)

    feats = instances_from_probs(
        grid,
        p1,
        p2,
        p3,
        transform=transform,
        addl_props={"scale_m": 20, "threeP": True},
        discard_edge_polygons_buffer=0.0005,
    )
    return ee.FeatureCollection(
        [ee.Feature(ee.Geometry(f.geometry), f.properties) for f in feats]
    )


def logistic_speckle(
    img: ee.Image,
    radius_px: int = 4,
    sigma0: float = 0.02,
    k: float = 50,
) -> ee.Image:
    """
    Blend each pixel with its neighbourhood mean using a logistic weight:
        w(sigma) = 1 / (1 + exp( k · (sigma - sigma₀) ))
    Low sigma  ⇒  w→1 (heavy smoothing),  High sigma ⇒ w→0 (retain detail).
    """
    img = ee.Image(img)  # ← ensure Image API is available

    kernel = ee.Kernel.square(radius_px, "pixels", False)

    # nbr_mean = img.reduceNeighborhood(ee.Reducer.mean(), kernel)
    nbr_var = img.reduceNeighborhood(ee.Reducer.variance(), kernel)
    sigma = nbr_var.sqrt()

    w = (sigma.subtract(sigma0).multiply(k).multiply(-1)).exp().add(1).pow(-1)
    return img.multiply(w)
    # return w.multiply(nbr_mean).add(ee.Image(1).subtract(w).multiply(img))
