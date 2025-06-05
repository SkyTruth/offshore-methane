import sys
from pathlib import Path

# ruff: noqa: E402

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))  # noqa: E402

import numpy as np
from offshore_methane import mbsp


def test_invert_mbsp_identity():
    const = mbsp.S2_CONSTANTS["A"]
    delta_true = 0.65
    r = np.exp(-const.k12 * delta_true) - np.exp(-const.k11 * delta_true)
    delta = mbsp.invert_mbsp(np.array([r]), const)[0]
    assert abs(delta - delta_true) < 1e-3
