"""Test checked flat-index and coordinate conversion."""

from typing import Any

import numpy as np
import pytest

from pydynopt.arrays import ind2sub, sub2ind

_ind2sub_any: Any = ind2sub
_sub2ind_any: Any = sub2ind


@pytest.mark.parametrize('index', [0, 4, 5, np.int32(4), np.uint64(5)])
def test_ind2sub_integer_scalars_match_numpy(index: int | np.integer) -> None:
    expected = np.asarray(np.unravel_index(index, (2, 3)), dtype=np.int64)
    result = ind2sub(index, (2, 3))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)
    assert ind2sub(index, (2, 3), axis=0) == int(expected[0])
    assert ind2sub(index, (2, 3), axis=-1) == int(expected[-1])


@pytest.mark.parametrize(
    'indices',
    [
        [0, 2, 5],
        (0, 2, 5),
        np.array([0, 2, 5], dtype=np.int32),
        np.arange(6, dtype=np.uint64).reshape(2, 3),
    ],
)
def test_ind2sub_arrays_match_numpy(indices: Any) -> None:
    index_array = np.asarray(indices)
    expected = np.stack(np.unravel_index(index_array, (2, 3)))

    result = ind2sub(indices, (2, 3))
    assert result.shape == (2, *index_array.shape)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(ind2sub(indices, (2, 3), 0), expected[0])
    np.testing.assert_array_equal(ind2sub(indices, (2, 3), -1), expected[-1])


@pytest.mark.parametrize(
    ('coords', 'shape'),
    [
        ([1, 2], (2, 3)),
        ((np.int32(1), np.int64(2)), (2, 3)),
        (np.array([1, 2]), (2, 3)),
    ],
)
def test_sub2ind_single_point_matches_numpy(
    coords: Any, shape: tuple[int, ...]
) -> None:
    result = sub2ind(coords, shape)
    assert type(result) is int
    assert result == np.ravel_multi_index(tuple(coords), shape)


def test_sub2ind_dimension_first_batches_match_numpy() -> None:
    coords = np.array([[0, 1, 1], [0, 1, 2]])
    expected = np.array([0, 4, 5])
    np.testing.assert_array_equal(sub2ind(coords, (2, 3)), expected)

    coords3 = np.array(
        [
            [[0, 0], [1, 1]],
            [[0, 2], [1, 2]],
        ]
    )
    expected3 = np.ravel_multi_index((coords3[0], coords3[1]), (2, 3))
    result3 = sub2ind(coords3, (2, 3))
    assert isinstance(result3, np.ndarray)
    assert result3.shape == (2, 2)
    assert result3.dtype == np.int64
    np.testing.assert_array_equal(result3, expected3)


def test_index_conversion_round_trip_preserves_sample_shape() -> None:
    indices = np.arange(24).reshape(2, 3, 4)
    coords = ind2sub(indices, (2, 3, 4))
    assert coords.shape == (3, 2, 3, 4)
    np.testing.assert_array_equal(sub2ind(coords, (2, 3, 4)), indices)


def test_index_output_buffers_are_fully_written_and_returned() -> None:
    indices = np.array([[0, 1, 5], [3, 4, 2]])
    full_out = np.full((2, 2, 3), -1, dtype=np.int64)
    axis_out = np.full(indices.shape, -1, dtype=np.int64)
    coords = ind2sub(indices, (2, 3), out=full_out)
    axis = ind2sub(indices, (2, 3), axis=1, out=axis_out)

    assert coords is full_out
    assert axis is axis_out
    np.testing.assert_array_equal(coords, np.stack(np.unravel_index(indices, (2, 3))))
    np.testing.assert_array_equal(axis, coords[1])

    scalar_out = np.full(2, -1, dtype=np.int64)
    assert ind2sub(4, (2, 3), out=scalar_out) is scalar_out
    np.testing.assert_array_equal(scalar_out, [1, 1])

    flat_out = np.full(indices.shape, -1, dtype=np.int64)
    assert sub2ind(coords, (2, 3), out=flat_out) is flat_out
    np.testing.assert_array_equal(flat_out, indices)


@pytest.mark.parametrize('index', [-1, 6, 11, 12])
def test_ind2sub_rejects_scalar_indices_outside_bounds(index: int) -> None:
    with pytest.raises(ValueError, match='0 <= indices'):
        ind2sub(index, (2, 3))


@pytest.mark.parametrize('indices', [[-1, 0], [0, 6], np.array([[12]])])
def test_ind2sub_rejects_array_indices_outside_bounds(indices: Any) -> None:
    with pytest.raises(ValueError, match='0 <= indices'):
        _ind2sub_any(indices, (2, 3))


@pytest.mark.parametrize('coords', [[-1, 0], [2, 0], [0, -1], [0, 3]])
def test_sub2ind_rejects_scalar_coordinates_outside_bounds(coords: list[int]) -> None:
    with pytest.raises(ValueError, match='coordinates for axis'):
        sub2ind(coords, (2, 3))


@pytest.mark.parametrize(
    'coords',
    [
        np.array([[0, 2], [0, 1]]),
        np.array([[0, 1], [0, 3]]),
        np.zeros((3, 2), dtype=int),
        np.array(1),
    ],
)
def test_sub2ind_rejects_invalid_batches(coords: np.ndarray) -> None:
    with pytest.raises(ValueError):
        sub2ind(coords, (2, 3))


@pytest.mark.parametrize(
    'shape',
    [(), [], (0, 2), (-1, 2), (2.0, 3.0), (True, 3), ((2, 3),)],
)
def test_indexing_rejects_invalid_shapes(shape: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        _ind2sub_any(0, shape)
    with pytest.raises((TypeError, ValueError)):
        _sub2ind_any([0], shape)


@pytest.mark.parametrize(
    'indices',
    [1.0, np.float64(1), True, [0.0, 1.0], np.array([True, False])],
)
def test_ind2sub_rejects_non_integer_indices(indices: Any) -> None:
    with pytest.raises(TypeError):
        _ind2sub_any(indices, (2, 3))


@pytest.mark.parametrize(
    'coords',
    [[0.0, 1.0], np.array([0.0, 1.0]), [True, False], [[0.0], [1.0]]],
)
def test_sub2ind_rejects_non_integer_coordinates(coords: Any) -> None:
    with pytest.raises(TypeError):
        _sub2ind_any(coords, (2, 3))


@pytest.mark.parametrize('axis', [-3, 2, 3])
def test_ind2sub_rejects_axes_outside_shape(axis: int) -> None:
    with pytest.raises(ValueError, match='outside'):
        ind2sub(0, (2, 3), axis)


def test_ind2sub_rejects_non_integer_axis() -> None:
    with pytest.raises(TypeError):
        _ind2sub_any(0, (2, 3), 1.0)


@pytest.mark.parametrize(
    'out',
    [
        np.empty((2, 2), dtype=np.int64),
        np.empty(2, dtype=np.int32),
        np.empty(2, dtype=np.float64),
    ],
)
def test_ind2sub_rejects_invalid_output_buffers(out: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        _ind2sub_any(0, (2, 3), out=out)


def test_indexing_rejects_read_only_outputs() -> None:
    ind_out = np.empty(2, dtype=np.int64)
    ind_out.flags.writeable = False
    with pytest.raises(ValueError, match='writable'):
        ind2sub(0, (2, 3), out=ind_out)

    sub_out = np.empty(2, dtype=np.int64)
    sub_out.flags.writeable = False
    with pytest.raises(ValueError, match='writable'):
        sub2ind([[0, 1], [0, 1]], (2, 3), out=sub_out)


def test_scalar_results_reject_output_buffers() -> None:
    out = np.empty(1, dtype=np.int64)
    with pytest.raises(TypeError, match='do not accept out'):
        _ind2sub_any(4, (2, 3), axis=1, out=out)
    with pytest.raises(TypeError, match='do not accept out'):
        _sub2ind_any([1, 1], (2, 3), out=out)
