# gcp_utils.py
"""
Utility functions for interacting with Google Cloud Platform.
"""

import shutil


# ------------------------------------------------------------------
#  GSUtil helper
# ------------------------------------------------------------------
def gsutil_cmd() -> str:
    cmd = shutil.which("gsutil") or shutil.which("gsutil.cmd")
    if not cmd:
        raise RuntimeError("gsutil not found on system PATH.")
    return cmd
