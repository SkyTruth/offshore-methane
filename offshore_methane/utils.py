"""
Lightweight utilities shared across modules.

Notably, refresh_config() reloads offshore_methane.config so that
subsequent accesses reflect on-disk edits without restarting the kernel.
"""

from __future__ import annotations

import importlib


def refresh_config():
    """Reload the config module in-place and return it.

    Any existing references to the module object (e.g., `cfg` imported at
    module scope) will see updated attributes after this call.
    """
    from . import config as cfg

    return importlib.reload(cfg)
