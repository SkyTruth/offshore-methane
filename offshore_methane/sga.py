# sga.py
"""
Write the 23x23 *coarse* Sun-Glint-Angle grid to GeoTIFF and stage it to
either GCS or EE assets.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import rasterio
from lxml import etree
from rasterio.transform import from_origin

from offshore_methane.cdse import download_xml
from offshore_methane.ee_utils import ee_asset_ready


# ---------------------------------------------------------------------
#  GeoTIFF writer
# ---------------------------------------------------------------------
def compute_sga_coarse(sid: str, tif_path: Path) -> None:
    """
    Save the un-interpolated 23x23 SGA grid (5 km px) as GeoTIFF
    oriented for Earth-Engine ingestion.
    """
    xml_path = tif_path.with_suffix(".xml")
    download_xml(sid, xml_path)  # @Brendan which is cheaper and faster? GCS vs CDSE
    root = etree.parse(str(xml_path))

    # ----- helpers ---------------------------------------------------
    def _grid2array(values_list):
        rows = [np.fromstring(row, sep=" ") for row in values_list]
        return np.vstack(rows).astype(np.float32)

    def _mean_detector_grid(root_, band_id: str, xpath_suffix: str):
        expr = (
            f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']/"
            f"{xpath_suffix}/Values_List/VALUES/text()"
        )
        blocks = root_.xpath(expr)
        arrays = [
            _grid2array(blocks[i * 23 : (i + 1) * 23]) for i in range(len(blocks) // 23)
        ]
        return np.nanmean(arrays, axis=0)

    # ----- raw 23x23 values -----------------------------------------
    sza = _grid2array(root.xpath(".//Sun_Angles_Grid/Zenith/Values_List/VALUES/text()"))
    saa = _grid2array(
        root.xpath(".//Sun_Angles_Grid/Azimuth/Values_List/VALUES/text()")
    )
    vza = _mean_detector_grid(root, "12", "Zenith")
    vaa = _mean_detector_grid(root, "12", "Azimuth")

    # sun-glint angle
    delta_phi = np.deg2rad(np.abs(saa - vaa))
    sga_grid = np.rad2deg(
        np.arccos(
            np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza))
            - np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(delta_phi)
        )
    ).astype(np.float32)

    # EE expects origin top-left with row-major order
    sga_grid = np.rot90(
        np.flipud(sga_grid.T), k=-1
    )  # @Brendan WTF is jona doing here? Should I really be rotating and flipping arbitrarily to make it "look right"?

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
        # --- COG‑compliant settings ----------------------------------
        "tiled": True,  # GeoTIFF is now tiled
        "blockxsize": 16,  # must be multiple of 16
        "blockysize": 16,
        "compress": "deflate",
        "predictor": 2,
    }
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(sga_grid, 1)


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


def gcs_stage(local_path: Path, sid: str, datatype: str, bucket: str) -> str:
    """
    Upload *local_path* to GCS bucket (if not present) and return gs:// URL.
    """
    dst = f"gs://{bucket}/{sid}/{sid}_{datatype}.tif"
    subprocess.run(
        ["gsutil", "cp", str(local_path.resolve()), dst],  # drop the “-n” (no‑clobber)
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return dst


def ensure_sga_asset(
    sid: str,
    ee_asset_folder: str,
    bucket: str,
    local_path: Path = Path("../data"),
    preferred_location: str | None = None,
    **kwargs,
) -> tuple[str, bool]:
    """
    Return (location, exported)

    exported == True  ⇢ we created / overwrote something this call
    exported == False ⇢ artefact was already present & ready
    """
    datatype = "SGA"
    asset_id = f"{ee_asset_folder}/{sid}_{datatype}"
    overwrite: bool = kwargs.get("overwrite", False)
    timeout: int = kwargs.get("timeout", 300)
    exported = False

    tif_path = local_path / sid / f"{sid}_{datatype}.tif"
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not tif_path.is_file():
        print(f"  ↻ computing *coarse* SGA grid for {sid}")
        compute_sga_coarse(sid, tif_path)
        exported = True

    gcs_url = f"gs://{bucket}/{sid}/{tif_path.name}"
    gcs_exists = (
        subprocess.run(
            ["gsutil", "ls", gcs_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if overwrite or not gcs_exists:
        gcs_url = gcs_stage(tif_path, sid, datatype, bucket)
        exported = True

    if preferred_location == "ee_asset_folder":
        if overwrite or not ee_asset_ready(asset_id):
            print(f"  ↑ ingesting as EE asset {asset_id}")
            subprocess.run(
                ["earthengine", "upload", "image", f"--asset_id={asset_id}", gcs_url],
                check=True,
            )
            start = time.time()
            while not ee_asset_ready(asset_id):
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for EE asset {asset_id}")
                time.sleep(5)
            exported = True

        if overwrite or not gcs_exists:
            subprocess.run(["gsutil", "rm", gcs_url], stdout=subprocess.DEVNULL)

        return asset_id, exported

    return gcs_url, exported
