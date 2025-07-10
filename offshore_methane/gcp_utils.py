# gcp_utils.py
"""
Utility functions for interacting with Google Cloud Platform.
"""

import shutil

from google.cloud import storage


# ------------------------------------------------------------------
#  GSUtil helper
# ------------------------------------------------------------------
def gsutil_cmd() -> str:
    cmd = shutil.which("gsutil") or shutil.which("gsutil.cmd")
    if not cmd:
        raise RuntimeError("gsutil not found on system PATH.")
    return cmd


def download_xml_gcs(gcs_path, xml_path):
    client = storage.Client()
    s2_bucket = client.get_bucket("gcp-public-data-sentinel-2")
    blob = s2_bucket.blob(gcs_path)
    blob.download_to_filename(xml_path)
