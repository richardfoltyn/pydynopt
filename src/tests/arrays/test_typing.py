"""Exercise public array annotations as a package consumer."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from pydynopt.arrays import clip_prob, ind2sub, logspace, powerspace, sub2ind


def test_consumer_return_types() -> None:
    indices = np.array([0, 4, 5], dtype=np.int64)
    coords = np.array([[0, 1, 1], [0, 1, 2]], dtype=np.int64)
    index_out = np.empty((2, 3), dtype=np.int64)
    axis_out = np.empty(3, dtype=np.int64)
    prob = np.array([0.05, 0.5, 0.95])
    prob_out = np.empty(3)

    assert_type(ind2sub(4, (2, 3)), NDArray[np.int64])
    assert_type(ind2sub(np.int64(4), (2, 3)), NDArray[np.int64])
    assert_type(ind2sub(4, (2, 3), axis=1), int)
    assert_type(ind2sub(np.int32(4), (2, 3), axis=-1), int)
    assert_type(ind2sub([0, 4, 5], (2, 3)), NDArray[np.int64])
    assert_type(ind2sub((0, 4, 5), (2, 3), axis=1), NDArray[np.int64])
    assert_type(ind2sub(indices, (2, 3)), NDArray[np.int64])
    assert_type(ind2sub(indices, (2, 3), out=index_out), NDArray[np.int64])
    assert_type(ind2sub(indices, (2, 3), axis=1, out=axis_out), NDArray[np.int64])

    assert_type(sub2ind((1, 2), (2, 3)), int)
    assert_type(sub2ind([np.int64(1), np.int32(2)], (2, 3)), int)
    assert_type(sub2ind(coords, (2, 3)), int | NDArray[np.int64])
    assert_type(
        sub2ind(coords, (2, 3), out=np.empty(3, dtype=np.int64)),
        int | NDArray[np.int64],
    )

    assert_type(clip_prob(0.5, 0.1), float)
    assert_type(clip_prob(np.float32(0.5), np.float64(0.1)), float)
    assert_type(clip_prob([0.05, 0.5], 0.1), NDArray[np.float64])
    assert_type(clip_prob(prob, 0.1), NDArray[np.float64])
    assert_type(clip_prob(prob, 0.1, out=prob_out), NDArray[np.float64])
    assert_type(powerspace(0.0, 1.0, 3, 2.0), NDArray[np.float64])
    assert_type(logspace(1.0, 10.0, 3), NDArray[np.float64])


def test_ebl_scalar_patterns() -> None:
    ih = 1
    it = np.int64(0)
    iyp = 2
    shape3 = (2, 2, 3)
    assert_type(sub2ind((ih, it, iyp), shape3), int)

    ixo = np.int64(4)
    shape2 = (2, 3)
    coord_result = assert_type(ind2sub(ixo, shape2), NDArray[np.int64])
    it_result, iyp_result = coord_result
    assert int(it_result) == 1
    assert int(iyp_result) == 1
