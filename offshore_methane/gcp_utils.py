# gcp_utils.py
"""
Utility functions for interacting with Google Cloud Platform.
"""

import shutil
from typing import Optional

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


# ------------------------------------------------------------------
#  Small text helpers (upload/read) for GCS
# ------------------------------------------------------------------
def upload_text_gcs(bucket: str, blob_path: str, content: str) -> None:
    """Upload a small text content to ``gs://{bucket}/{blob_path}``.

    Creates intermediate paths if needed. Overwrites existing by default.
    """
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(blob_path)
    blob.upload_from_string(content, content_type="text/plain")


def read_text_gcs(bucket: str, blob_path: str) -> Optional[str]:
    """Read small text from ``gs://{bucket}/{blob_path}``.

    Returns None if the object does not exist or cannot be read.
    """
    try:
        client = storage.Client()
        b = client.bucket(bucket)
        blob = b.blob(blob_path)
        if not blob.exists():
            return None
        return blob.download_as_text()
    except Exception:
        return None
