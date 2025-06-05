import numpy as np
from offshore_methane import mbsp


def test_invert_mbsp_identity():
    const = mbsp.S2_CONSTANTS["A"]
    delta_true = 0.65
    r = np.exp(-const.k12 * delta_true) - np.exp(-const.k11 * delta_true)
    delta = mbsp.invert_mbsp(np.array([r]), const)[0]
    assert abs(delta - delta_true) < 1e-3


def test_mbsp_fractional_zero_for_uniform_scene():
    b11 = np.ones((5, 5))
    b12 = 2 * b11
    c, r = mbsp.mbsp_fractional_absorption(b11, b12)
    assert np.allclose(r, 0.0)
    assert abs(c - 0.5) < 1e-6
