# sga.py
"""
Write the 23x23 *coarse* Sun-Glint-Angle grid to GeoTIFF and stage it to
either GCS or EE assets.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import os
import numpy as np
import rasterio
from lxml import etree
from rio_cogeo import cog_profiles
from rio_cogeo.cogeo import cog_translate
from rasterio.transform import from_origin

from offshore_methane.gcs_utils import get_grid_values_from_xml, download_xml
from google.cloud import storage

# ---------------------------------------------------------------------
#  GeoTIFF writer
# ---------------------------------------------------------------------
def compute_sga_coarse(sid: str, tif_path: Path, image_bucket = 'offshore-methane') -> None:
    """
    Save the un-interpolated 23x23 SGA grid (5 km px) as GeoTIFF
    oriented for Earth-Engine ingestion.
    """
    glint_bytes = download_xml(sid)

    parser = etree.XMLParser(no_network=True, remove_blank_text=True)
    root = etree.fromstring(glint_bytes, parser=parser)

    # ----- raw 23x23 values -----------------------------------------
    sza = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Zenith')
    saa = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Azimuth')
    vza = get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Zenith')
    vaa = get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Azimuth')

    # sun-glint angle
    delta_phi = np.deg2rad(np.abs(saa - vaa))
    sga_grid = np.rad2deg(
        np.arccos(
            np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza))
            - np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(delta_phi)
        )
    ).astype(np.float32)

    # geotransform: 5 km pixels
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
    tif_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(sga_grid, 1)
        cog_translate(
                dst,
                tif_path,
                cog_profiles.get('deflate'),
                in_memory=False
            )
    
    blob = image_bucket.blob(tif_path)
    blob.upload_from_filename(tif_path)

    # Removes the file on local.
    os.remove(tif_path)

# ---------------------------------------------------------------------
#  GCS + EE staging helpers
# ---------------------------------------------------------------------

def _is_cog(tif: Path) -> bool:
    try:
        import rasterio

        with rasterio.open(tif) as ds:
            return ds.is_tiled
    except Exception:
        return False


def gcs_stage(local_path: Path, bucket: str) -> str:
    """
    Upload *local_path* to GCS bucket (if not present) and return gs:// URL.
    """
    dst = f"gs://{bucket}/sga/{local_path.name}"
    subprocess.run(
        ["gsutil", "cp", str(local_path.resolve()), dst],  # drop the “-n” (no‑clobber)
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return dst


def ensure_sga_asset(
    sid: str,
    bucket,
    tif_path,
    local_path: Path = Path("../data"),
    **kwargs,
) -> str:
    client = storage.Client()                # Uses default credentials
    bucket = client.bucket(bucket)
    blob = bucket.blob(f"{bucket}/{tif_path}")

    if not blob.exists(client):                  # Pass client for a single round-trip
        print(f"  ↻ computing *coarse* SGA grid for {sid}")
        compute_sga_coarse(sid,tif_path,bucket)

    else:
        print(f'Grid for {sid} already exists.')

    return f"gs://{bucket}/{tif_path}"