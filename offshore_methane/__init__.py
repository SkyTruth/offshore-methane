"""
Lazy module accessors so callers can do:

    from offshore_methane import mbsp, sga, ee_utils

without importing heavy dependencies at package import time.
"""

from importlib import import_module as _imp

__all__ = [
    "orchestrator",
    "cdse",
    "sga",
    "ee_utils",
    "mbsp",
    "algos",
    "csv_utils",
    "masking",
]


def __getattr__(name):
    if name in __all__:
        mod = _imp(f".{name}", package=__name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
