"""Test public three-dimensional interpolation from Numba callers."""

from typing import Any

from numba import njit
import numpy as np
import pytest

from pydynopt.interpolate import interp3d, interp3d_eval, interp3d_locate
import pydynopt.interpolate.numba.linear as kernels

_interp3d_eval_any: Any = interp3d_eval


@njit
def _locate3_point(
    x0: float,
    x1: float,
    x2: float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return interp3d_locate(x0, x1, x2, xp0, xp1, xp2)


@njit
def _locate3_array(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return interp3d_locate(x0, x1, x2, xp0, xp1, xp2)


@njit
def _eval3_point(index: np.ndarray, weight: np.ndarray, fp: np.ndarray) -> float:
    return _interp3d_eval_any(index, weight, fp)


@njit
def _eval3_array(index: np.ndarray, weight: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return _interp3d_eval_any(index, weight, fp)


@njit
def _eval3_tuple(
    index0: int,
    index1: int,
    index2: int,
    weight0: float,
    weight1: float,
    weight2: float,
    fp: np.ndarray,
    extrapolate: bool = True,
) -> float:
    return interp3d_eval(
        (index0, index1, index2),
        (weight0, weight1, weight2),
        fp,
        extrapolate,
    )


@njit
def _eval3_tuples_repeated(
    index0: int,
    index1: int,
    index2: int,
    weight0: float,
    weight1: float,
    weight2: float,
    fp0: np.ndarray,
    fp1: np.ndarray,
    fp2: np.ndarray,
) -> tuple[float, float, float]:
    index = (index0, index1, index2)
    weight = (weight0, weight1, weight2)
    return (
        interp3d_eval(index, weight, fp0),
        interp3d_eval(index, weight, fp1),
        interp3d_eval(index, weight, fp2),
    )


@njit
def _interp3_point(
    x0: float,
    x1: float,
    x2: float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
) -> float:
    return interp3d(x0, x1, x2, xp0, xp1, xp2, fp)


@njit
def _interp3_array(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    return interp3d(x0, x1, x2, xp0, xp1, xp2, fp)


@njit
def _output_buffers(
    x: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
) -> tuple[bool, bool, bool, bool]:
    shape = (x.size, 3)
    index = np.empty(shape, dtype=np.int64)
    weight = np.empty(shape, dtype=np.float64)
    index_ret, weight_ret = interp3d_locate(x, x, x, xp0, xp1, xp2, None, index, weight)
    out_eval = np.empty(x.shape, dtype=np.float64)
    out_eval_ret = interp3d_eval(index, weight, fp, out=out_eval)
    out = np.empty(x.shape, dtype=np.float64)
    out_ret = interp3d(x, x, x, xp0, xp1, xp2, fp, out=out)
    return (
        index_ret is index,
        weight_ret is weight,
        out_eval_ret is out_eval,
        out_ret is out,
    )


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create an affine three-dimensional test field."""
    xp0 = np.array([0.0, 1.0, 3.0])
    xp1 = np.array([0.0, 2.0, 5.0])
    xp2 = np.array([-1.0, 3.0, 6.0])
    fp = xp0[:, None, None] + 2.0 * xp1[None, :, None] + 3.0 * xp2[None, None, :]
    return xp0, xp1, xp2, fp


def test_public_3d_numba_point_and_array_paths() -> None:
    xp0, xp1, xp2, fp = _data()
    x0 = np.array([0.5, 2.0])
    x1 = np.array([1.0, 4.0])
    x2 = np.array([1.0, 5.0])

    index, weight = _locate3_point(0.5, 1.0, 1.0, xp0, xp1, xp2)
    assert index.shape == weight.shape == (3,)
    assert _eval3_point(index, weight, fp) == pytest.approx(5.5)
    assert _interp3_point(0.5, 1.0, 1.0, xp0, xp1, xp2, fp) == pytest.approx(5.5)

    index, weight = _locate3_array(x0, x1, x2, xp0, xp1, xp2)
    expected = x0 + 2.0 * x1 + 3.0 * x2
    np.testing.assert_allclose(_eval3_array(index, weight, fp), expected)
    np.testing.assert_allclose(_interp3_array(x0, x1, x2, xp0, xp1, xp2, fp), expected)

    for function in (
        _locate3_point,
        _locate3_array,
        _eval3_point,
        _eval3_array,
        _interp3_point,
        _interp3_array,
    ):
        assert function.nopython_signatures


def test_numba_point_tuples_are_allocation_free_for_all_layouts() -> None:
    _, _, _, values = _data()
    fields_c = (values, 2.0 * values, -values)
    backing = tuple(np.repeat(fp, 2, axis=2) for fp in fields_c)
    fields_a = tuple(fp[:, :, ::2] for fp in backing)
    assert all(fp.flags.c_contiguous for fp in fields_c)
    assert all(
        not fp.flags.c_contiguous and not fp.flags.f_contiguous for fp in fields_a
    )

    for fields in (fields_c, fields_a):
        expected = tuple(
            interp3d_eval((1, 1, 0), (0.25, 0.75, 0.5), fp) for fp in fields
        )
        result = _eval3_tuples_repeated(1, 1, 0, 0.25, 0.75, 0.5, *fields)
        np.testing.assert_allclose(result, expected)

    index = np.array([1, 1, 0], dtype=np.int64)
    weight = np.array([0.25, 0.75, 0.5])
    expected = interp3d_eval((1, 1, 0), (0.25, 0.75, 0.5), fields_a[0])
    assert _eval3_point(index, weight, fields_a[0]) == pytest.approx(expected)
    assert np.isnan(_eval3_tuple(0, 0, 0, 1.5, 0.5, 0.5, values, False))

    layouts = {
        getattr(sig[6], 'layout', None) for sig in _eval3_tuples_repeated.signatures
    }
    assert layouts == {'A', 'C'}
    for llvm in _eval3_tuples_repeated.inspect_llvm().values():
        allocation_calls = [
            line
            for line in llvm.splitlines()
            if 'call' in line and 'NRT_MemInfo_alloc' in line
        ]
        assert not allocation_calls


def test_numba_output_buffers_and_integer_results() -> None:
    xp0, xp1, xp2, fp = _data()
    x = np.array([0.5, 2.0])
    assert all(_output_buffers(x, xp0, xp1, xp2, fp))

    xp = np.array([0, 2])
    values = np.indices((2, 2, 2)).sum(axis=0)
    result = _interp3_array(
        np.array([1]), np.array([1]), np.array([1]), xp, xp, xp, values
    )
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [1.5])


def test_retained_3d_kernels() -> None:
    xp0, xp1, xp2, fp = _data()
    x0 = np.array([0.5, 2.0])
    x1 = np.array([1.0, 4.0])
    x2 = np.array([1.0, 5.0])
    expected = x0 + 2.0 * x1 + 3.0 * x2

    index, weight = kernels.interp3d_locate_array(x0, x1, x2, xp0, xp1, xp2)
    index_out = np.empty_like(index)
    weight_out = np.empty_like(weight)
    kernels.interp3d_locate_array_impl(
        x0, x1, x2, xp0, xp1, xp2, None, index_out, weight_out
    )
    index_point, weight_point = kernels.interp3d_locate_point(
        0.5, 1.0, 1.0, xp0, xp1, xp2
    )
    kernels.interp3d_locate_point_impl(
        0.5, 1.0, 1.0, xp0, xp1, xp2, None, index_point, weight_point
    )
    np.testing.assert_array_equal(index, index_out)
    np.testing.assert_allclose(weight, weight_out)

    assert kernels.interp3d_eval_point(index_point, weight_point, fp) == pytest.approx(
        5.5
    )
    np.testing.assert_allclose(kernels.interp3d_eval_array(index, weight, fp), expected)
    out = np.empty_like(expected)
    kernels.interp3d_eval_array_impl(index, weight, fp, True, out)
    np.testing.assert_allclose(out, expected)
    assert kernels.interp3d_point(0.5, 1.0, 1.0, xp0, xp1, xp2, fp) == pytest.approx(
        5.5
    )
    np.testing.assert_allclose(
        kernels.interp3d_array(x0, x1, x2, xp0, xp1, xp2, fp), expected
    )
    kernels.interp3d_array_impl(x0, x1, x2, xp0, xp1, xp2, fp, None, True, out)
    np.testing.assert_allclose(out, expected)
