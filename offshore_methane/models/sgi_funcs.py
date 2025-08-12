"""Pre-baked SGI≈f(SGA) model (degree=4) - generated automatically."""

import numpy as np

_C0 = -0.21836443382073098
_C1 = 0.04022614760278719
_C2 = -0.002929080320565447
_C3 = 6.0272862796648723e-05
_C4 = -3.984986063009622e-07

_S0 = -0.025113424633160627
_S1 = 0.003558565213275315
_S2 = -0.00010591658198753598
_S3 = 8.938828979178076e-07
_S4 = 6.249164415977171e-09


def _poly(arr, coeffs):
    return sum(c * arr**i for i, c in enumerate(coeffs))


def sgi_hat(alpha):
    a = np.asarray(alpha, float)
    return _poly(
        a,
        [
            -0.21836443382073098,
            0.04022614760278719,
            -0.002929080320565447,
            6.0272862796648723e-05,
            -3.984986063009622e-07,
        ],
    )


def sgi_std(alpha):
    a = np.asarray(alpha, float)
    return np.clip(
        _poly(
            a,
            [
                -0.025113424633160627,
                0.003558565213275315,
                -0.00010591658198753598,
                8.938828979178076e-07,
                6.249164415977171e-09,
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
