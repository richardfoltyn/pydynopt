"""Test checked two-dimensional interpolation functions."""

from typing import Any

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pydynopt.interpolate import interp2d, interp2d_eval, interp2d_locate

_interp2d_any: Any = interp2d


@pytest.fixture
def grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xp0 = np.array([-2.0, 0.0, 1.0, 4.0])
    xp1 = np.array([-1.0, 2.0, 5.0])
    fp = xp0[:, None] + 2.0 * xp1[None, :] + xp0[:, None] * xp1[None, :]
    return xp0, xp1, fp


def test_locate_scalar_shape_and_buffers(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, _ = grid
    index_out = np.full(2, -1, dtype=np.int64)
    weight_out = np.full(2, np.nan)
    index, weight = interp2d_locate(
        np.float64(0.5),
        1.0,
        xp0,
        xp1,
        ilb=np.array([99, -1]),
        index_out=index_out,
        weight_out=weight_out,
    )
    assert index is index_out
    assert weight is weight_out
    np.testing.assert_array_equal(index, [1, 0])
    np.testing.assert_allclose(weight, [0.5, 1.0 / 3.0])


def test_locate_broadcasting_and_partial_buffers(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, _ = grid
    x0 = np.array([[-2.0], [0.5]])
    x1 = np.array([-2, 2, 8])
    shape = (2, 3, 2)
    weight_out = np.empty(shape)
    index, weight = interp2d_locate(x0, x1, xp0, xp1, weight_out=weight_out)
    assert index.shape == shape
    assert index.dtype == np.int64
    assert weight is weight_out
    assert weight.dtype == np.float64
    np.testing.assert_array_equal(index[..., 0], [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(index[..., 1], [[0, 1, 1], [0, 1, 1]])


def test_eval_single_point_returns_float(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, _, fp = grid
    index = np.array([1, 0], dtype=np.int32)
    weight = np.array([0.5, 1.0 / 3.0])
    value = interp2d_eval(index, weight, fp)
    assert isinstance(value, float)
    assert value == pytest.approx(3.0)


def test_eval_array_and_output_identity(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, fp = grid
    x0 = np.array([[-3.0, 0.5], [1.0, 6.0]])
    x1 = np.array([[-2.0, 1.0], [5.0, 7.0]])
    index, weight = interp2d_locate(x0, x1, xp0, xp1)
    out = np.full(x0.shape, np.nan)
    result = interp2d_eval(index, weight, fp, out=out)
    assert result is out
    np.testing.assert_allclose(result, interp2d(x0, x1, xp0, xp1, fp))


def test_eval_disables_extrapolation(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, fp = grid
    x0 = np.array([-3.0, 0.5, 1.0])
    x1 = np.array([1.0, 7.0, 2.0])
    index, weight = interp2d_locate(x0, x1, xp0, xp1)
    result = interp2d_eval(index, weight, fp, extrapolate=False)
    assert isinstance(result, np.ndarray)
    assert np.isnan(result[:2]).all()
    assert result[2] == pytest.approx(7.0)


def test_combined_broadcasting_matches_scipy(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, fp = grid
    x0 = np.array([[-3.0], [0.5], [5.0]])
    x1 = np.array([-2.0, 1.0, 7.0])
    result = interp2d(x0, x1, xp0, xp1, fp)

    xx0, xx1 = np.broadcast_arrays(x0, x1)
    points = np.column_stack((xx0.ravel(), xx1.ravel()))
    reference = RegularGridInterpolator(
        (xp0, xp1), fp, bounds_error=False, fill_value=None
    )
    expected = reference(points).reshape(xx0.shape)
    np.testing.assert_allclose(result, expected)


def test_combined_scalar_and_zero_dimensional_array(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, fp = grid
    scalar = interp2d(0.5, 1.0, xp0, xp1, fp)
    array = interp2d(np.array(0.5), np.array(1.0), xp0, xp1, fp)
    assert isinstance(scalar, float)
    assert isinstance(array, np.ndarray)
    assert array.shape == ()
    assert scalar == pytest.approx(array.item())


def test_combined_integer_inputs_do_not_truncate() -> None:
    xp = np.array([0, 2])
    fp = np.array([[0, 2], [2, 4]])
    result = interp2d(np.array([1]), np.array([1]), xp, xp, fp)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [2.0])


def test_combined_output_identity(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, fp = grid
    x0 = np.array([0.5, 1.0])
    x1 = np.array([1.0, 2.0])
    out = np.empty(4)[::2]
    result = interp2d(x0, x1, xp0, xp1, fp, out=out)
    assert result is out
    np.testing.assert_allclose(result, [3.0, 7.0])


@pytest.mark.parametrize(
    'xp',
    [
        np.array([]),
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, np.inf]),
    ],
)
def test_invalid_grids(xp: np.ndarray) -> None:
    valid = np.array([0.0, 1.0])
    fp = np.zeros((xp.size, valid.size))
    with pytest.raises(ValueError):
        interp2d(0.0, 0.0, xp, valid, fp)


def test_invalid_shapes_and_indices(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, fp = grid
    with pytest.raises(ValueError):
        interp2d(np.ones(2), np.ones(3), xp0, xp1, fp)
    with pytest.raises(ValueError):
        interp2d(0.0, 0.0, xp0, xp1, fp[:-1])
    with pytest.raises(ValueError):
        interp2d_eval(np.zeros((2, 3), dtype=np.int64), np.zeros((2, 3)), fp)
    with pytest.raises(ValueError):
        interp2d_eval(np.zeros((2, 2), dtype=np.int64), np.zeros((1, 2)), fp)
    with pytest.raises(IndexError):
        interp2d_eval(np.array([fp.shape[0] - 1, 0]), np.array([0.5, 0.5]), fp)


def test_invalid_output_buffers(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, fp = grid
    x = np.array([0.5, 1.0])
    with pytest.raises(TypeError):
        _interp2d_any(0.5, 1.0, xp0, xp1, fp, out=np.empty(1))
    with pytest.raises(TypeError):
        interp2d_eval(np.array([1, 0]), np.array([0.5, 0.5]), fp, out=np.empty(1))
    with pytest.raises(ValueError):
        interp2d(x, x, xp0, xp1, fp, out=np.empty(3))
    with pytest.raises(TypeError):
        _interp2d_any(x, x, xp0, xp1, fp, out=np.empty(2, dtype=np.float32))
    with pytest.raises(ValueError):
        interp2d_locate(x, x, xp0, xp1, index_out=np.empty((2, 3), dtype=np.int64))
