"""Helper utilities for loading Sentinel-2 images."""
from __future__ import annotations

import re

import ee


_SCENE_RE = re.compile(r"^\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3}$")
_GRANULE_RE = re.compile(
    r"^S2[AB]_MSI(?:L1C|L2A)_\d{8}T\d{6}_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_\d{8}T\d{6}$"
)


def load_s2_image(identifier: str, sr: bool = True) -> ee.Image:
    """Load a Sentinel-2 image by scene or granule identifier.

    Parameters
    ----------
    identifier : str
        Either a scene ID (e.g. ``20210101T123456_20210101T123459_T31TCJ``) or a
        product/granule ID (e.g. ``S2A_MSIL2A_20210101T123456_N0214_R137_T31TCJ_20210101T145857``).
    sr : bool, optional
        If ``True`` the image is loaded from the surface reflectance collection
        ``COPERNICUS/S2_SR_HARMONIZED``. Otherwise the top-of-atmosphere
        collection ``COPERNICUS/S2_HARMONIZED`` is used.

    Returns
    -------
    ee.Image
        Earth Engine image referenced by ``identifier``.
    """

    dataset = "COPERNICUS/S2_SR_HARMONIZED" if sr else "COPERNICUS/S2_HARMONIZED"

    # If a full Earth Engine path is supplied, just load it directly.
    if identifier.startswith("COPERNICUS/"):
        return ee.Image(identifier)

    if _SCENE_RE.match(identifier):
        return ee.Image(f"{dataset}/{identifier}")

    if _GRANULE_RE.match(identifier):
        parts = identifier.split("_")
        scene_id = f"{parts[2]}_{parts[-1]}_{parts[5]}"
        return ee.Image(f"{dataset}/{scene_id}")

    raise ValueError(f"Unrecognized Sentinel-2 identifier: {identifier}")
