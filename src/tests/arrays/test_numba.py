"""Test public array utilities from actual Numba-compiled callers."""

from typing import Any

from numba import njit
from numba.core.errors import TypingError
import numpy as np
import pytest

from pydynopt.arrays import clip_prob, ind2sub, logspace, powerspace, sub2ind
import pydynopt.arrays.numba.arrays as array_kernels
import pydynopt.arrays.numba.indexing as index_kernels

_clip_prob_any: Any = clip_prob
_ind2sub_any: Any = ind2sub
_logspace_any: Any = logspace
_sub2ind_any: Any = sub2ind


@njit
def _clip_scalar(value: float) -> float:
    return clip_prob(value, tol=0.1)


@njit
def _clip_array(value: np.ndarray) -> np.ndarray:
    return clip_prob(value, 0.1)


@njit
def _clip_array_out(value: np.ndarray, out: np.ndarray) -> tuple[bool, np.ndarray]:
    result = clip_prob(value=value, tol=0.1, out=out)
    return result is out, result


@njit
def _power_grid(xmin: float, xmax: float, n: int, exponent: float) -> np.ndarray:
    return powerspace(xmin, xmax, n, exponent)


@njit
def _ind_scalar_full(index: int) -> np.ndarray:
    return ind2sub(index, (2, 3))


@njit
def _ind_scalar_axis(index: int) -> int:
    return ind2sub(indices=index, shape=(2, 3), axis=-1)


@njit
def _ind_scalar_out(index: int, out: np.ndarray) -> tuple[bool, np.ndarray]:
    result = ind2sub(index, (2, 3), out=out)
    return result is out, result


@njit
def _ind_array_full(indices: np.ndarray) -> np.ndarray:
    return ind2sub(indices, (2, 3))


@njit
def _ind_array_axis(indices: np.ndarray) -> np.ndarray:
    return ind2sub(indices, (2, 3), axis=1)


@njit
def _ind_array_out(indices: np.ndarray, out: np.ndarray) -> tuple[bool, np.ndarray]:
    result = ind2sub(indices, (2, 3), out=out)
    return result is out, result


@njit
def _sub_tuple(coord0: int, coord1: int) -> int:
    return sub2ind((coord0, coord1), (2, 3))


@njit
def _sub_array(coords: np.ndarray) -> int | np.ndarray:
    return sub2ind(coords=coords, shape=(2, 3))


@njit
def _sub_array_out(
    coords: np.ndarray, out: np.ndarray
) -> tuple[bool, int | np.ndarray]:
    result = sub2ind(coords, (2, 3), out=out)
    return result is out, result


@njit
def _bad_logspace() -> np.ndarray:
    return _logspace_any(1.0, 10.0, 3)


@njit
def _bad_clip_scalar_out(value: float, out: np.ndarray) -> float:
    return _clip_prob_any(value, 0.1, out=out)


@njit
def _bad_ind_scalar_axis_out(index: int, out: np.ndarray) -> int:
    return _ind2sub_any(index, (2, 3), axis=1, out=out)


@njit
def _bad_sub_scalar_out(coords: np.ndarray, out: np.ndarray) -> int:
    return _sub2ind_any(coords, (2, 3), out=out)


def test_numba_clip_prob_scalar_array_and_output_paths() -> None:
    value = np.array([[0.05, 0.1, 0.5], [0.9, 0.95, 0.75]])
    expected = clip_prob(value, 0.1)

    assert _clip_scalar(0.05) == 0.0
    np.testing.assert_allclose(_clip_array(value), expected)
    out = np.full(value.shape, -99.0)
    identical, result = _clip_array_out(value, out)
    assert identical
    assert result is out
    np.testing.assert_allclose(out, expected)

    for function in (_clip_scalar, _clip_array, _clip_array_out):
        assert function.nopython_signatures


def test_numba_powerspace_matches_python() -> None:
    for args in ((0.0, 1.0, 5, 2.0), (1.0, 0.0, 5, 0.5)):
        np.testing.assert_allclose(_power_grid(*args), powerspace(*args))
    assert _power_grid.nopython_signatures


def test_numba_ind2sub_scalar_array_axis_and_output_paths() -> None:
    assert _ind_scalar_axis(4) == ind2sub(4, (2, 3), axis=-1)
    np.testing.assert_array_equal(_ind_scalar_full(4), ind2sub(4, (2, 3)))
    scalar_out = np.full(2, -1, dtype=np.int64)
    scalar_identical, scalar_result = _ind_scalar_out(4, scalar_out)
    assert scalar_identical
    assert scalar_result is scalar_out
    np.testing.assert_array_equal(scalar_result, [1, 1])

    indices = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(_ind_array_full(indices), ind2sub(indices, (2, 3)))
    np.testing.assert_array_equal(
        _ind_array_axis(indices), ind2sub(indices, (2, 3), axis=1)
    )

    out = np.full((2, 2, 3), -1, dtype=np.int64)
    identical, result = _ind_array_out(indices, out)
    assert identical
    assert result is out
    np.testing.assert_array_equal(result, ind2sub(indices, (2, 3)))

    for function in (
        _ind_scalar_full,
        _ind_scalar_axis,
        _ind_scalar_out,
        _ind_array_full,
        _ind_array_axis,
        _ind_array_out,
    ):
        assert function.nopython_signatures


def test_numba_sub2ind_scalar_batch_and_output_paths() -> None:
    assert _sub_tuple(1, 2) == 5
    assert _sub_array(np.array([1, 2])) == 5

    coords = np.array(
        [
            [[0, 0], [1, 1]],
            [[0, 2], [1, 2]],
        ]
    )
    expected = sub2ind(coords, (2, 3))
    np.testing.assert_array_equal(_sub_array(coords), expected)

    out = np.full((2, 2), -1, dtype=np.int64)
    identical, result = _sub_array_out(coords, out)
    assert identical
    assert isinstance(result, np.ndarray)
    assert result is out
    np.testing.assert_array_equal(result, expected)

    for function in (_sub_tuple, _sub_array, _sub_array_out):
        assert function.nopython_signatures


@pytest.mark.parametrize(
    ('function', 'args'),
    [
        (_bad_clip_scalar_out, (0.5, np.empty(1))),
        (_bad_ind_scalar_axis_out, (4, np.empty(1, dtype=np.int64))),
        (
            _bad_sub_scalar_out,
            (np.array([1, 1]), np.empty(1, dtype=np.int64)),
        ),
    ],
)
def test_numba_rejects_scalar_output_buffers(
    function: Any, args: tuple[object, ...]
) -> None:
    with pytest.raises(TypingError):
        function(*args)


def test_logspace_remains_python_only() -> None:
    with pytest.raises(TypingError):
        _bad_logspace()


@pytest.mark.parametrize(
    ('function', 'args'),
    [
        (_ind_scalar_full, (-1,)),
        (_ind_scalar_full, (6,)),
        (_ind_array_full, (np.array([0, 6]),)),
        (_sub_tuple, (2, 0)),
        (_sub_tuple, (0, 3)),
        (_sub_array, (np.array([[0, 2], [0, 1]]),)),
    ],
)
def test_numba_indexing_retains_bounds_checks(
    function: Any, args: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize('tol', [-0.1, 0.6, np.nan])
def test_numba_clip_prob_retains_tolerance_checks(tol: float) -> None:
    @njit
    def call(value: float, tolerance: float) -> float:
        return clip_prob(value, tolerance)

    with pytest.raises(ValueError):
        call(0.5, tol)


@pytest.mark.parametrize(('n', 'exponent'), [(0, 1.0), (2, 0.0), (2, np.inf)])
def test_numba_powerspace_retains_argument_checks(n: int, exponent: float) -> None:
    with pytest.raises(ValueError):
        _power_grid(0.0, 1.0, n, exponent)


def test_all_retained_array_kernels() -> None:
    value = np.array([0.05, 0.5, 0.95])
    out = np.full(3, -99.0)
    assert array_kernels.clip_prob_scalar(0.05, 0.1) == 0.0
    array_kernels.clip_prob_array_impl(value, 0.1, out)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        array_kernels.clip_prob_array(value, 0.1), [0.0, 0.5, 1.0]
    )
    np.testing.assert_allclose(
        array_kernels.powerspace_impl(0.0, 1.0, 3, 2.0), [0.0, 0.25, 1.0]
    )


def test_all_retained_indexing_kernels() -> None:
    shape = (2, 3)
    indices = np.arange(6).reshape(2, 3)
    expected = np.stack(np.unravel_index(indices, shape))

    scalar_out = np.empty(2, dtype=np.int64)
    index_kernels.ind2sub_scalar_impl(4, shape, scalar_out)
    np.testing.assert_array_equal(scalar_out, [1, 1])
    np.testing.assert_array_equal(index_kernels.ind2sub_scalar(4, shape), [1, 1])
    assert index_kernels.ind2sub_axis_scalar(4, shape, 1) == 1

    full_out = np.empty((2, 2, 3), dtype=np.int64)
    index_kernels.ind2sub_array_impl(indices, shape, full_out)
    np.testing.assert_array_equal(full_out, expected)
    np.testing.assert_array_equal(index_kernels.ind2sub_array(indices, shape), expected)

    axis_out = np.empty(indices.shape, dtype=np.int64)
    index_kernels.ind2sub_axis_array_impl(indices, shape, -1, axis_out)
    np.testing.assert_array_equal(axis_out, expected[-1])
    np.testing.assert_array_equal(
        index_kernels.ind2sub_axis_array(indices, shape, -1), expected[-1]
    )

    assert index_kernels.sub2ind_scalar((1, 2), shape) == 5
    flat_out = np.empty(indices.shape, dtype=np.int64)
    index_kernels.sub2ind_array_impl(expected, shape, flat_out)
    np.testing.assert_array_equal(flat_out, indices)
    np.testing.assert_array_equal(index_kernels.sub2ind_array(expected, shape), indices)
