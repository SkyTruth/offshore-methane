"""Pre-baked SGI≈f(SGA) model (degree=4) - generated automatically."""

import numpy as np

_C0 = -0.3048542086504608
_C1 = 0.05348627759480248
_C2 = -0.003638766938640032
_C3 = 7.606464506980443e-05
_C4 = -5.244303829367348e-07

_S0 = 0.004000184504171324
_S1 = 0.00014620621473049927
_S2 = 6.977547846386697e-05
_S3 = -1.7049612764821823e-06
_S4 = 1.3259552900809324e-08


def _poly(arr, coeffs):
    return sum(c * arr**i for i, c in enumerate(coeffs))


def sgi_hat(alpha):
    a = np.asarray(alpha, float)
    return _poly(
        a,
        [
            -0.3048542086504608,
            0.05348627759480248,
            -0.003638766938640032,
            7.606464506980443e-05,
            -5.244303829367348e-07,
        ],
    )


def sgi_std(alpha):
    a = np.asarray(alpha, float)
    return np.clip(
        _poly(
            a,
            [
                0.004000184504171324,
                0.00014620621473049927,
                6.977547846386697e-05,
                -1.7049612764821823e-06,
                1.3259552900809324e-08,
            ],
        ),
        1e-4,
        None,
    )


# -------- Earth-Engine helpers -----------
def sgi_hat_img(alpha_img):
    return (
        alpha_img.pow(4)
        .multiply(_C4)
        .add(alpha_img.pow(3).multiply(_C3))
        .add(alpha_img.pow(2).multiply(_C2))
        .add(alpha_img.pow(1).multiply(_C1))
        .add(_C0)
    )


def sgi_std_img(alpha_img):
    std = (
        alpha_img.pow(4)
        .multiply(_S4)
        .add(alpha_img.pow(3).multiply(_S3))
        .add(alpha_img.pow(2).multiply(_S2))
        .add(alpha_img.pow(1).multiply(_S1))
        .add(_S0)
    )
    return std.max(1e-4)
