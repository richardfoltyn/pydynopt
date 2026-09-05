"""Exercise interpolation annotations as a package consumer."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from pydynopt.interpolate import (
    interp1d,
    interp1d_eval,
    interp1d_locate,
    interp2d,
    interp2d_eval,
    interp2d_locate,
)


def test_consumer_return_types() -> None:
    xp = np.array([0.0, 1.0, 2.0])
    fp = np.array([0.0, 2.0, 4.0])
    x = np.array([0.25, 0.75])
    index = np.array([0, 0], dtype=np.int64)
    weight = np.array([0.75, 0.25])

    assert_type(interp1d_locate(0.5, xp), tuple[int, float])
    assert_type(interp1d_locate(np.float64(0.5), xp), tuple[int, float])
    assert_type(
        interp1d_locate(x, xp),
        tuple[NDArray[np.int64], NDArray[np.float64]],
    )
    assert_type(
        interp1d_locate([0.25, 0.75], xp),
        tuple[NDArray[np.int64], NDArray[np.float64]],
    )
    assert_type(interp1d_eval(0, 0.5, fp), float)
    assert_type(interp1d_eval(np.int64(0), np.float64(0.5), fp), float)
    assert_type(interp1d_eval(index, weight, fp), NDArray[np.float64])
    assert_type(interp1d(0.5, xp, fp), float)
    assert_type(interp1d(np.float32(0.5), xp, fp), float)
    assert_type(interp1d(x, xp, fp), NDArray[np.float64])
    assert_type(interp1d([0.25, 0.75], xp, fp), NDArray[np.float64])

    assert_type(
        interp2d_locate(0.5, np.float64(0.5), xp, xp),
        tuple[NDArray[np.int64], NDArray[np.float64]],
    )
    assert_type(
        interp2d_locate(x, 0.5, xp, xp),
        tuple[NDArray[np.int64], NDArray[np.float64]],
    )
    index2, weight2 = interp2d_locate(x, x, xp, xp)
    assert_type(
        interp2d_eval(index2[0], weight2[0], fp[:, None] + fp),
        float | NDArray[np.float64],
    )
    assert_type(
        interp2d_eval(index2, weight2, fp[:, None] + fp), float | NDArray[np.float64]
    )
    assert_type(interp2d(0.5, 0.5, xp, xp, fp[:, None] + fp), float)
    assert_type(interp2d(x, 0.5, xp, xp, fp[:, None] + fp), NDArray[np.float64])
