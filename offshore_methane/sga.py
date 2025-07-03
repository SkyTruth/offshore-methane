# sga.py
"""
Write the 23x23 *coarse* Sun-Glint-Angle grid to GeoTIFF and stage it to
either GCS or EE assets.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio
from google.cloud import storage
from lxml import etree
from rasterio.transform import from_origin
from rio_cogeo import cog_profiles
from rio_cogeo.cogeo import cog_translate

from offshore_methane.gcs_utils import download_xml, get_grid_values_from_xml


# ---------------------------------------------------------------------
#  GeoTIFF writer
# ---------------------------------------------------------------------
def compute_sga_coarse(
    sid: str,
    local_tif: Path,
    bucket_obj: storage.Bucket,
    remote_path: str,
) -> None:
    """
    Save the un-interpolated 23 x 23 SGA grid (5 km px) as a Cloud-Optimised
    GeoTIFF, upload it to ``bucket_obj`` at ``remote_path`` and remove the
    on-disk temporary.
    """
    glint_bytes = download_xml(sid)

    parser = etree.XMLParser(no_network=True, remove_blank_text=True)
    root = etree.fromstring(glint_bytes, parser=parser)

    # ----- raw 23x23 values -----------------------------------------
    sza = get_grid_values_from_xml(root, ".//Sun_Angles_Grid/Zenith")
    saa = get_grid_values_from_xml(root, ".//Sun_Angles_Grid/Azimuth")
    vza = get_grid_values_from_xml(root, ".//Viewing_Incidence_Angles_Grids/Zenith")
    vaa = get_grid_values_from_xml(root, ".//Viewing_Incidence_Angles_Grids/Azimuth")

    # sun-glint angle
    delta_phi = np.deg2rad(np.abs(saa - vaa))
    sga_grid = np.rad2deg(
        np.arccos(
            np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza))
            - np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(delta_phi)
        )
    ).astype(np.float32)

    # geotransform: 5 km pixels
    ulx = float(root.findtext(".//Geoposition/ULX"))
    uly = float(root.findtext(".//Geoposition/ULY"))
    tr = from_origin(ulx, uly, 5000.0, 5000.0)

    profile = {
        "driver": "GTiff",
        "width": 23,
        "height": 23,
        "count": 1,
        "dtype": "float32",
        "crs": root.findtext(".//Tile_Geocoding/HORIZONTAL_CS_CODE"),
        "transform": tr,
    }
    local_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(local_tif, "w", **profile) as dst:
        dst.write(sga_grid, 1)
        cog_translate(dst, local_tif, cog_profiles.get("deflate"), in_memory=False)

    # ─── Upload to GCS ────────────────────────────────────────────────
    blob = bucket_obj.blob(remote_path)
    blob.upload_from_filename(str(local_tif))

    # Clean up local artefact
    os.remove(local_tif)


# ---------------------------------------------------------------------
#  GCS + EE staging helpers
# ---------------------------------------------------------------------
def ensure_sga_asset(
    sid: str,
    bucket: str,
    local_dir: Path = Path("../data/sga"),
    **kwargs,
) -> str:
    """
    Guarantee that a Cloud-Optimised GeoTIFF for the Sentinel-2 ``sid`` exists
    in the requested destination and return its URL.

    Parameters
    ----------
    sid
        Sentinel-2 image identifier (same string you’d pass to
        ``COPERNICUS/S2_HARMONIZED``).
    bucket
        Name of the GCS bucket where the file should live.
    local_dir
        Where to create the temporary GeoTIFF.

    Returns
    -------
    str
        ``gs://`` URL.
    """
    # File naming convention
    filename = f"{sid}_sga_coarse.tif"
    remote_path = f"sga/{filename}"
    local_tif = local_dir / filename

    # Set up GCS client/bucket
    client = storage.Client()  # Uses default credentials
    bucket_obj = client.bucket(bucket)
    #  Set up a blob object for GCS
    blob = bucket_obj.blob(remote_path)

    if not blob.exists(client):  # Pass client for a single round-trip
        print(f"  ↻ computing *coarse* SGA grid for {sid}")
        compute_sga_coarse(sid, local_tif, bucket_obj, remote_path)

    else:
        print(f"Grid for {sid} already exists.")

    return f"gs://{bucket}/{remote_path}"
