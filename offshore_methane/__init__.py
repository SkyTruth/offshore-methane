"""
Lightweight convenience re-exports so callers can do:

    from offshore_methane import mbsp, three_p, sga
"""

from importlib import import_module as _imp

__all__ = [
    "orchestrator",
    "cdse",
    "sga",
    "ee_utils",
    "mbsp",
    "algos",
]

for _m in __all__:
    globals()[_m] = _imp(f".{_m}", package=__name__)
