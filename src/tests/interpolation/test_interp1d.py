"""Test checked one-dimensional interpolation functions."""

from typing import Any

import numpy as np
import pytest

from pydynopt.interpolate import interp1d, interp1d_eval, interp1d_locate

_interp1d_any: Any = interp1d
_interp1d_eval_any: Any = interp1d_eval
_interp1d_locate_any: Any = interp1d_locate


@pytest.fixture
def grid() -> tuple[np.ndarray, np.ndarray]:
    xp = np.array([-2.0, 0.0, 1.0, 4.0])
    fp = 3.0 * xp - 1.0
    return xp, fp


@pytest.mark.parametrize('x', [0.5, np.float32(0.5), np.float64(0.5)])
def test_locate_scalar_types(x: float, grid: tuple[np.ndarray, np.ndarray]) -> None:
    xp, _ = grid
    index, weight = interp1d_locate(x, xp, ilb=np.int64(2))
    assert isinstance(index, int)
    assert isinstance(weight, float)
    assert index == 1
    assert weight == pytest.approx(0.5)


def test_locate_array_shapes_dtype_and_partial_buffers(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    xp, _ = grid
    x = np.array([[-3, 0], [1, 7]])
    index_out = np.full(x.shape, -1, dtype=np.int64)

    index, weight = interp1d_locate(x, xp, ilb=99, index_out=index_out)

    assert index is index_out
    assert weight.shape == x.shape
    assert weight.dtype == np.float64
    np.testing.assert_array_equal(index, [[0, 1], [2, 2]])
    np.testing.assert_allclose(weight, [[1.5, 1.0], [1.0, -1.0]])


def test_locate_sequence_and_weight_buffer_identity(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    xp, _ = grid
    weight_out = np.empty(3, dtype=np.float64)
    index, weight = interp1d_locate([-2.0, 0.5, 4.0], xp, weight_out=weight_out)
    assert weight is weight_out
    np.testing.assert_array_equal(index, [0, 1, 2])
    np.testing.assert_allclose(weight, [1.0, 0.5, 0.0])


@pytest.mark.parametrize('index', [1, np.int32(1), np.int64(1)])
def test_eval_scalar_integer_types(
    index: int,
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    _, fp = grid
    value = interp1d_eval(index, np.float32(0.25), fp)
    assert isinstance(value, float)
    assert value == pytest.approx(1.25)


def test_eval_arrays_and_extrapolation_controls(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    _, fp = grid
    index = np.array([[0, 1], [2, 2]], dtype=np.int32)
    weight = np.array([[1.5, 0.5], [1.0, -1.0]])
    expected = np.array([[-9.0, 0.5], [2.0, 99.0]])
    result = interp1d_eval(
        index,
        weight,
        fp,
        extrapolate=False,
        left=-9.0,
        right=99.0,
    )
    np.testing.assert_allclose(result, expected)


def test_eval_output_identity_and_integer_values() -> None:
    fp = np.array([0, 1, 4])
    index = np.array([0, 1])
    weight = np.array([0.4, 0.5])
    out = np.full(2, np.nan)
    result = interp1d_eval(index, weight, fp, out=out)
    assert result is out
    np.testing.assert_allclose(result, [0.6, 2.5])


@pytest.mark.parametrize(
    'x',
    [0.5, [0.5, 2.5], np.array(0.5), np.array([[0.5], [2.5]])],
)
def test_combined_matches_affine_function(
    x: Any,
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    xp, fp = grid
    result = interp1d(x, xp, fp)
    expected = 3.0 * np.asarray(x) - 1.0
    np.testing.assert_allclose(result, expected)
    if isinstance(x, np.ndarray):
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
    elif np.isscalar(x):
        assert isinstance(result, float)


def test_combined_matches_numpy_interp(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    xp, fp = grid
    x = np.array([-5.0, -2.0, -0.5, 0.0, 0.25, 1.0, 2.5, 4.0, 7.0])
    expected = np.interp(x, xp, fp)
    result = interp1d(
        x,
        xp,
        fp,
        extrapolate=False,
        left=float(fp[0]),
        right=float(fp[-1]),
    )
    np.testing.assert_allclose(result, expected)


def test_linear_extrapolation_matches_affine_reference(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    xp, fp = grid
    x = np.array([[-5.0, -3.0], [5.0, 7.0]])
    expected = 3.0 * x - 1.0

    index, weight = interp1d_locate(x, xp)
    np.testing.assert_allclose(interp1d_eval(index, weight, fp), expected)
    np.testing.assert_allclose(interp1d(x, xp, fp), expected)


def test_combined_matches_locate_then_eval(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    xp, fp = grid
    x = np.array([[-3.0, -2.0, 0.5], [1.0, 4.0, 7.0]])
    index, weight = interp1d_locate(x, xp)
    split = interp1d_eval(index, weight, fp)
    np.testing.assert_allclose(interp1d(x, xp, fp), split)


def test_combined_output_identity(grid: tuple[np.ndarray, np.ndarray]) -> None:
    xp, fp = grid
    x = np.array([0.5, 2.5])
    backing = np.empty(4)
    out = backing[::2]
    result = interp1d(x, xp, fp, out=out)
    assert result is out
    np.testing.assert_allclose(result, 3.0 * x - 1.0)


@pytest.mark.parametrize(
    'xp',
    [
        np.array([]),
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, np.nan]),
    ],
)
def test_invalid_grids(xp: np.ndarray) -> None:
    with pytest.raises(ValueError):
        interp1d_locate(0.0, xp)


def test_invalid_function_values_and_indices(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    xp, fp = grid
    with pytest.raises(ValueError):
        interp1d(0.0, xp, fp[:, None])
    with pytest.raises(ValueError):
        interp1d(0.0, xp, fp[:-1])
    with pytest.raises(IndexError):
        interp1d_eval(np.array([-1]), np.array([0.5]), fp)
    with pytest.raises(IndexError):
        interp1d_eval(np.array([fp.size - 1]), np.array([0.5]), fp)
    with pytest.raises(ValueError):
        interp1d_eval(np.array([0, 1]), np.array([0.5]), fp)
    with pytest.raises(TypeError):
        _interp1d_eval_any(0, np.array(0.5), fp)


def test_invalid_output_buffers(grid: tuple[np.ndarray, np.ndarray]) -> None:
    xp, fp = grid
    x = np.array([0.5, 2.0])
    readonly = np.empty(x.shape)
    readonly.flags.writeable = False

    with pytest.raises(TypeError):
        _interp1d_any(0.5, xp, fp, out=np.empty(1))
    with pytest.raises(TypeError):
        _interp1d_locate_any(0.5, xp, index_out=np.empty(1, dtype=np.int64))
    with pytest.raises(ValueError):
        interp1d(x, xp, fp, out=np.empty(1))
    with pytest.raises(TypeError):
        interp1d(x, xp, fp, out=np.empty(x.shape, dtype=np.float32))
    with pytest.raises(TypeError):
        interp1d(x, xp, fp, out=np.empty(x.shape, dtype=np.int64))
    with pytest.raises(ValueError):
        interp1d(x, xp, fp, out=readonly)
    with pytest.raises(TypeError):
        interp1d_locate(x, xp, index_out=np.empty(x.shape, dtype=np.int32))


def test_removed_axis_argument(grid: tuple[np.ndarray, np.ndarray]) -> None:
    xp, fp = grid
    with pytest.raises(TypeError):
        interp1d(0.5, xp, fp, axis=0)  # type: ignore
