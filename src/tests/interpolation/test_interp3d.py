"""Test checked three-dimensional interpolation functions."""

from typing import Any

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pydynopt.interpolate import (
    interp1d_locate,
    interp3d,
    interp3d_eval,
    interp3d_locate,
)

_interp3d_any: Any = interp3d
_interp3d_eval_any: Any = interp3d_eval


@pytest.fixture
def grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xp0 = np.array([-2.0, 0.0, 1.0, 4.0])
    xp1 = np.array([-1.0, 2.0, 5.0])
    xp2 = np.array([0.0, 4.0, 6.0, 9.0])
    x0 = xp0[:, None, None]
    x1 = xp1[None, :, None]
    x2 = xp2[None, None, :]
    fp = x0 + 2.0 * x1 + 3.0 * x2 + 0.25 * x0 * x1 * x2
    return xp0, xp1, xp2, fp


def _reference(
    index: tuple[int, int, int],
    weight: tuple[float, float, float],
    fp: np.ndarray,
) -> float:
    """Evaluate the explicit trilinear arithmetic tree."""
    i0, i1, i2 = index
    w0, w1, w2 = weight
    value00 = w0 * fp[i0, i1, i2] + (1.0 - w0) * fp[i0 + 1, i1, i2]
    value01 = w0 * fp[i0, i1, i2 + 1] + (1.0 - w0) * fp[i0 + 1, i1, i2 + 1]
    value10 = w0 * fp[i0, i1 + 1, i2] + (1.0 - w0) * fp[i0 + 1, i1 + 1, i2]
    value11 = w0 * fp[i0, i1 + 1, i2 + 1] + (1.0 - w0) * fp[i0 + 1, i1 + 1, i2 + 1]
    value0 = w1 * value00 + (1.0 - w1) * value10
    value1 = w1 * value01 + (1.0 - w1) * value11
    value = w2 * value0 + (1.0 - w2) * value1
    return float(value)


def test_locate_point_shape_and_buffers(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, xp2, _ = grid
    index_out = np.full(3, -1, dtype=np.int64)
    weight_out = np.full(3, np.nan)
    index, weight = interp3d_locate(
        np.float64(0.5),
        1.0,
        5.0,
        xp0,
        xp1,
        xp2,
        ilb=np.array([99, -1, 99]),
        index_out=index_out,
        weight_out=weight_out,
    )
    assert index is index_out
    assert weight is weight_out
    np.testing.assert_array_equal(index, [1, 0, 1])
    np.testing.assert_allclose(weight, [0.5, 1.0 / 3.0, 0.5])


def test_locate_broadcasting_matches_independent_1d_calls(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, xp2, _ = grid
    x0 = np.array([[-3.0], [0.5]])
    x1 = np.array(1.0)
    x2 = np.array([-1.0, 5.0, 10.0])
    shape = (2, 3, 3)
    index_out = np.empty(shape, dtype=np.int64)
    index, weight = interp3d_locate(x0, x1, x2, xp0, xp1, xp2, index_out=index_out)
    assert index is index_out
    assert weight.shape == shape

    xx0, xx1, xx2 = np.broadcast_arrays(x0, x1, x2)
    for axis, (xx, xp) in enumerate(((xx0, xp0), (xx1, xp1), (xx2, xp2))):
        expected_index, expected_weight = interp1d_locate(xx, xp)
        np.testing.assert_array_equal(index[..., axis], expected_index)
        np.testing.assert_allclose(weight[..., axis], expected_weight)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize(
    ('index', 'weight', 'extrapolate'),
    [
        ((1, 0, 1), (0.5, 1.0 / 3.0, 0.5), True),
        ((0, 0, 0), (1.0, 1.0, 1.0), True),
        ((2, 1, 2), (0.0, 0.0, 0.0), True),
        ((0, 1, 1), (1.5, -0.5, 0.25), True),
        ((0, 1, 1), (1.5, -0.5, 0.25), False),
        ((1, 0, 1), (np.nan, 0.5, 0.5), True),
    ],
)
def test_eval_point_matches_explicit_reference(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dtype: type[np.float32] | type[np.float64],
    strided: bool,
    index: tuple[int, int, int],
    weight: tuple[float, float, float],
    extrapolate: bool,
) -> None:
    _, _, _, fp = grid
    values = fp.astype(dtype)
    if strided:
        backing = np.empty(
            (values.shape[0], values.shape[1], 2 * values.shape[2]), dtype=dtype
        )
        backing[:, :, ::2] = values
        values = backing[:, :, ::2]
    assert values.flags.c_contiguous is not strided

    if not extrapolate and any(value < 0.0 or value > 1.0 for value in weight):
        expected = np.nan
    else:
        expected = _reference(index, weight, values)
    result_tuple = interp3d_eval(index, weight, values, extrapolate)
    result_array = interp3d_eval(
        np.asarray(index, dtype=np.int32),
        np.asarray(weight, dtype=dtype),
        values,
        extrapolate,
    )
    assert isinstance(result_tuple, float)
    assert isinstance(result_array, float)
    np.testing.assert_allclose(result_tuple, expected, equal_nan=True)
    np.testing.assert_allclose(result_array, expected, equal_nan=True)


def test_eval_preserves_nonfinite_corner_values(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, _, _, fp = grid
    values = fp.copy()
    values[1, 1, 2] = np.inf
    index = (1, 0, 1)
    weight = (0.5, 0.0, 0.0)
    expected = _reference(index, weight, values)
    result = interp3d_eval(index, weight, values)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_eval_array_output_and_extrapolation(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, xp2, fp = grid
    x0 = np.array([[-3.0, 0.5], [1.0, 6.0]])
    x1 = np.array([[-2.0, 1.0], [5.0, 7.0]])
    x2 = np.array([[1.0, 5.0], [9.0, 4.0]])
    index, weight = interp3d_locate(x0, x1, x2, xp0, xp1, xp2)
    out = np.empty((2, 4))[:, ::2]
    result = interp3d_eval(index, weight, fp, out=out)
    assert result is out
    np.testing.assert_allclose(result, interp3d(x0, x1, x2, xp0, xp1, xp2, fp))

    bounded = interp3d_eval(index, weight, fp, extrapolate=False)
    assert isinstance(bounded, np.ndarray)
    np.testing.assert_array_equal(np.isnan(bounded), [[True, False], [False, True]])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('strided', [False, True])
def test_combined_broadcasting_matches_scipy(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dtype: type[np.float32] | type[np.float64],
    strided: bool,
) -> None:
    xp0, xp1, xp2, fp = grid
    values = fp.astype(dtype)
    if strided:
        backing = np.empty(
            (values.shape[0], values.shape[1], 2 * values.shape[2]), dtype=dtype
        )
        backing[:, :, ::2] = values
        values = backing[:, :, ::2]

    x0 = np.array([[[-3.0]], [[0.5]], [[5.0]]])
    x1 = np.array([[[-2.0], [1.0], [7.0]]])
    x2 = np.array([-1.0, 5.0, 10.0])
    result = interp3d(x0, x1, x2, xp0, xp1, xp2, values)

    xx0, xx1, xx2 = np.broadcast_arrays(x0, x1, x2)
    points = np.column_stack((xx0.ravel(), xx1.ravel(), xx2.ravel()))
    reference = RegularGridInterpolator(
        (xp0, xp1, xp2), values, bounds_error=False, fill_value=None
    )
    expected = reference(points).reshape(xx0.shape)
    np.testing.assert_allclose(result, expected, rtol=2.0e-6)


def test_combined_point_zero_dimensional_and_integer_inputs(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, xp2, fp = grid
    point = interp3d(0.5, 1.0, 5.0, xp0, xp1, xp2, fp)
    array = interp3d(np.array(0.5), np.array(1.0), np.array(5.0), xp0, xp1, xp2, fp)
    assert isinstance(point, float)
    assert isinstance(array, np.ndarray)
    assert array.shape == ()
    assert point == pytest.approx(array.item())

    xp = np.array([0, 2])
    values = np.indices((2, 2, 2)).sum(axis=0)
    result = interp3d(np.array([1]), np.array([1]), np.array([1]), xp, xp, xp, values)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [1.5])


def test_invalid_inputs_and_output_buffers(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xp0, xp1, xp2, fp = grid
    with pytest.raises(ValueError):
        interp3d(np.ones(2), np.ones(3), np.ones(2), xp0, xp1, xp2, fp)
    with pytest.raises(ValueError):
        interp3d(0.0, 0.0, 0.0, xp0, xp1, xp2, fp[:-1])
    with pytest.raises(ValueError):
        interp3d_locate(0.0, 0.0, 0.0, xp0, xp1, xp2, ilb=(0, 0))
    with pytest.raises(ValueError):
        interp3d_eval(np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2)), fp)
    with pytest.raises(ValueError):
        _interp3d_eval_any((0, 0), (0.5, 0.5), fp)
    with pytest.raises(TypeError):
        _interp3d_eval_any((0, 0, 0), np.array([0.5, 0.5, 0.5]), fp)
    with pytest.raises(TypeError):
        _interp3d_eval_any((0.0, 0, 0), (0.5, 0.5, 0.5), fp)
    with pytest.raises(IndexError):
        interp3d_eval((fp.shape[0] - 1, 0, 0), (0.5, 0.5, 0.5), fp)
    with pytest.raises(TypeError):
        _interp3d_any(0.5, 1.0, 5.0, xp0, xp1, xp2, fp, out=np.empty(1))
    with pytest.raises(TypeError):
        interp3d_eval(
            np.array([1, 0, 1]), np.array([0.5, 0.5, 0.5]), fp, out=np.empty(1)
        )
    with pytest.raises(ValueError):
        interp3d_locate(
            np.ones(2),
            np.ones(2),
            np.ones(2),
            xp0,
            xp1,
            xp2,
            index_out=np.empty((2, 2), dtype=np.int64),
        )
