"""Test public interpolation calls made from Numba-compiled functions."""

from typing import Any

from numba import njit
from numba.core.errors import TypingError
import numpy as np
import pytest

from pydynopt.arrays import clip_prob, ind2sub
from pydynopt.interpolate import (
    interp1d,
    interp1d_eval,
    interp1d_locate,
    interp2d,
    interp2d_eval,
    interp2d_locate,
)
import pydynopt.interpolate.numba.linear as kernels

_clip_prob_any: Any = clip_prob
_ind2sub_any: Any = ind2sub
_interp1d_any: Any = interp1d
_interp2d_eval_any: Any = interp2d_eval


@njit
def _locate1_scalar(x: float, xp: np.ndarray) -> tuple[int, float]:
    return interp1d_locate(x, xp)


@njit
def _locate1_array(x: np.ndarray, xp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return interp1d_locate(x, xp)


@njit
def _eval1_scalar(index: int, weight: float, fp: np.ndarray) -> float:
    return interp1d_eval(index, weight, fp, False, -10.0, 10.0)


@njit
def _eval1_array(index: np.ndarray, weight: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return interp1d_eval(index, weight, fp)


@njit
def _interp1_scalar(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    return interp1d(x, xp, fp)


@njit
def _interp1_array(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return interp1d(x, xp, fp)


@njit
def _locate2_scalar(
    x0: float,
    x1: float,
    xp0: np.ndarray,
    xp1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return interp2d_locate(x0, x1, xp0, xp1)


@njit
def _locate2_array(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return interp2d_locate(x0, x1, xp0, xp1)


@njit
def _eval2_scalar(index: np.ndarray, weight: np.ndarray, fp: np.ndarray) -> float:
    return _interp2d_eval_any(index, weight, fp)


@njit
def _eval2_array(index: np.ndarray, weight: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return _interp2d_eval_any(index, weight, fp)


@njit
def _interp2_scalar(
    x0: float,
    x1: float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
) -> float:
    return interp2d(x0, x1, xp0, xp1, fp)


@njit
def _interp2_array(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    return interp2d(x0, x1, xp0, xp1, fp)


@njit
def _all_output_buffers(
    x: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp1: np.ndarray,
    fp2: np.ndarray,
) -> tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
    index1 = np.empty(x.shape, dtype=np.int64)
    weight1 = np.empty(x.shape, dtype=np.float64)
    index1_ret, weight1_ret = interp1d_locate(x, xp0, 0, index1, weight1)
    out1_eval = np.empty(x.shape, dtype=np.float64)
    out1_eval_ret = interp1d_eval(index1, weight1, fp1, out=out1_eval)
    out1 = np.empty(x.shape, dtype=np.float64)
    out1_ret = interp1d(x, xp0, fp1, out=out1)

    shape2 = (x.size, 2)
    index2 = np.empty(shape2, dtype=np.int64)
    weight2 = np.empty(shape2, dtype=np.float64)
    index2_ret, weight2_ret = interp2d_locate(x, x, xp0, xp1, None, index2, weight2)
    out2_eval = np.empty(x.shape, dtype=np.float64)
    out2_eval_ret = interp2d_eval(index2, weight2, fp2, out=out2_eval)
    out2 = np.empty(x.shape, dtype=np.float64)
    out2_ret = interp2d(x, x, xp0, xp1, fp2, out=out2)
    return (
        index1_ret is index1,
        weight1_ret is weight1,
        out1_eval_ret is out1_eval,
        out1_ret is out1,
        index2_ret is index2,
        weight2_ret is weight2,
        out2_eval_ret is out2_eval,
        out2_ret is out2,
    )


@njit
def _bad_scalar_out(x: float, xp: np.ndarray, fp: np.ndarray, out: np.ndarray) -> float:
    return _interp1d_any(x, xp, fp, out=out)


@njit
def _jitted_clip_scalar(value: float) -> float:
    return _clip_prob_any(value, 0.1)


@njit
def _jitted_clip_array(value: np.ndarray) -> np.ndarray:
    return _clip_prob_any(value, 0.1)


@njit
def _jitted_ind2sub(index: int) -> np.ndarray:
    return _ind2sub_any(index, (2, 3))


def test_public_1d_numba_scalar_and_array_paths() -> None:
    xp = np.array([0.0, 1.0, 3.0])
    fp = 2.0 * xp
    x = np.array([[-1.0, 0.5], [2.0, 4.0]])

    assert _locate1_scalar(0.5, xp) == pytest.approx((0, 0.5))
    index, weight = _locate1_array(x, xp)
    np.testing.assert_allclose(_eval1_array(index, weight, fp), 2.0 * x)
    assert _eval1_scalar(0, 1.5, fp) == -10.0
    assert _interp1_scalar(0.5, xp, fp) == pytest.approx(1.0)
    np.testing.assert_allclose(_interp1_array(x, xp, fp), 2.0 * x)

    for function in (
        _locate1_scalar,
        _locate1_array,
        _eval1_scalar,
        _eval1_array,
        _interp1_scalar,
        _interp1_array,
    ):
        assert function.nopython_signatures


def test_public_2d_numba_scalar_and_array_paths() -> None:
    xp0 = np.array([0.0, 1.0, 3.0])
    xp1 = np.array([0.0, 2.0, 5.0])
    fp = xp0[:, None] + 2.0 * xp1[None, :]
    x0 = np.array([0.5, 2.0])
    x1 = np.array([1.0, 4.0])

    index, weight = _locate2_scalar(0.5, 1.0, xp0, xp1)
    assert index.shape == weight.shape == (2,)
    assert _eval2_scalar(index, weight, fp) == pytest.approx(2.5)
    assert _interp2_scalar(0.5, 1.0, xp0, xp1, fp) == pytest.approx(2.5)

    index, weight = _locate2_array(x0, x1, xp0, xp1)
    expected = x0 + 2.0 * x1
    np.testing.assert_allclose(_eval2_array(index, weight, fp), expected)
    np.testing.assert_allclose(_interp2_array(x0, x1, xp0, xp1, fp), expected)

    for function in (
        _locate2_scalar,
        _locate2_array,
        _eval2_scalar,
        _eval2_array,
        _interp2_scalar,
        _interp2_array,
    ):
        assert function.nopython_signatures


def test_numba_output_buffer_identity() -> None:
    xp0 = np.array([0.0, 1.0, 3.0])
    xp1 = np.array([0.0, 2.0, 5.0])
    fp1 = 2.0 * xp0
    fp2 = xp0[:, None] + xp1[None, :]
    x = np.array([0.5, 2.0])
    assert all(_all_output_buffers(x, xp0, xp1, fp1, fp2))
    assert _all_output_buffers.nopython_signatures


def test_numba_rejects_scalar_output_buffer() -> None:
    xp = np.array([0.0, 1.0])
    fp = np.array([0.0, 2.0])
    with pytest.raises(TypingError):
        _bad_scalar_out(0.5, xp, fp, np.empty(1))


def test_integer_inputs_allocate_floating_results_in_numba() -> None:
    xp = np.array([0, 2])
    fp = np.array([0, 1])
    result = _interp1_array(np.array([1]), xp, fp)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [0.5])


def test_representative_shared_overloads_compile() -> None:
    assert _jitted_clip_scalar(0.05) == 0.0
    np.testing.assert_allclose(
        _jitted_clip_array(np.array([0.05, 0.5, 0.95])),
        [0.0, 0.5, 1.0],
    )
    np.testing.assert_array_equal(_jitted_ind2sub(4), [1, 1])
    assert _jitted_clip_scalar.nopython_signatures
    assert _jitted_clip_array.nopython_signatures
    assert _jitted_ind2sub.nopython_signatures


def test_all_retained_kernels() -> None:
    xp0 = np.array([0.0, 1.0, 3.0])
    xp1 = np.array([0.0, 2.0, 5.0])
    fp1 = 2.0 * xp0
    fp2 = xp0[:, None] + 2.0 * xp1[None, :]
    x0 = np.array([0.5, 2.0])
    x1 = np.array([1.0, 4.0])

    index, weight = kernels.interp1d_locate_array(x0, xp0)
    index_out = np.empty_like(index)
    weight_out = np.empty_like(weight)
    kernels.interp1d_locate_array_impl(x0, xp0, 0, index_out, weight_out)
    assert kernels.interp1d_locate_scalar(0.5, xp0) == pytest.approx((0, 0.5))
    np.testing.assert_allclose(index, index_out)
    np.testing.assert_allclose(weight, weight_out)

    expected1 = 2.0 * x0
    assert kernels.interp1d_eval_scalar(0, 0.5, fp1) == pytest.approx(1.0)
    np.testing.assert_allclose(
        kernels.interp1d_eval_array(index, weight, fp1), expected1
    )
    out1 = np.empty_like(expected1)
    kernels.interp1d_eval_array_impl(index, weight, fp1, True, np.nan, np.nan, out1)
    np.testing.assert_allclose(out1, expected1)
    assert kernels.interp1d_scalar(0.5, xp0, fp1) == pytest.approx(1.0)
    np.testing.assert_allclose(kernels.interp1d_array(x0, xp0, fp1), expected1)
    kernels.interp1d_array_impl(x0, xp0, fp1, 0, True, np.nan, np.nan, out1)
    np.testing.assert_allclose(out1, expected1)

    index2, weight2 = kernels.interp2d_locate_array(x0, x1, xp0, xp1)
    index2_out = np.empty_like(index2)
    weight2_out = np.empty_like(weight2)
    kernels.interp2d_locate_array_impl(x0, x1, xp0, xp1, None, index2_out, weight2_out)
    index_scalar, weight_scalar = kernels.interp2d_locate_scalar(0.5, 1.0, xp0, xp1)
    kernels.interp2d_locate_scalar_impl(
        0.5, 1.0, xp0, xp1, None, index_scalar, weight_scalar
    )
    np.testing.assert_allclose(index2, index2_out)
    np.testing.assert_allclose(weight2, weight2_out)

    expected2 = x0 + 2.0 * x1
    assert kernels.interp2d_eval_scalar(
        index_scalar, weight_scalar, fp2
    ) == pytest.approx(2.5)
    np.testing.assert_allclose(
        kernels.interp2d_eval_array(index2, weight2, fp2), expected2
    )
    out2 = np.empty_like(expected2)
    kernels.interp2d_eval_array_impl(index2, weight2, fp2, True, out2)
    np.testing.assert_allclose(out2, expected2)
    assert kernels.interp2d_scalar(0.5, 1.0, xp0, xp1, fp2) == pytest.approx(2.5)
    np.testing.assert_allclose(kernels.interp2d_array(x0, x1, xp0, xp1, fp2), expected2)
    kernels.interp2d_array_impl(x0, x1, xp0, xp1, fp2, None, True, out2)
    np.testing.assert_allclose(out2, expected2)
