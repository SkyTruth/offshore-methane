import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import mbsp


def test_mbsp_arrays():
    b11 = np.full((3, 3), 1000, dtype=np.int16)
    b12 = np.full((3, 3), 1100, dtype=np.int16)
    mbsp_arr, c = mbsp.mbsp_arrays(b11, b12)
    assert np.isclose(c, 1.1, atol=1e-3)
    assert np.allclose(mbsp_arr, 0.0, atol=1e-6)


def test_mbsp_to_column_monotonic():
    mbsp_vals = np.array([-0.01, 0.0, 0.05])
    col = mbsp.mbsp_to_column(mbsp_vals)
    assert np.all(np.diff(col) >= 0)

