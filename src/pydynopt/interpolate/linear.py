"""Provide checked one-, two-, and three-dimensional linear interpolation.

- Normalize and validate Python inputs before entering numerical kernels.
- Register Numba overloads that dispatch public calls to the same kernels.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence
from operator import index as operator_index
from typing import Any, overload as typing_overload

import numpy as np
from numpy.typing import NDArray

from pydynopt.numba import JIT_OPTIONS, jit, overload as numba_overload

from .numba.linear import (
    _interp2d_eval_point_c,
    _interp2d_scalar_c,
    _interp3d_eval_point_c,
    _interp3d_point_c,
    interp1d_array,
    interp1d_array_impl,
    interp1d_eval_array,
    interp1d_eval_array_impl,
    interp1d_eval_point,
    interp1d_locate_array,
    interp1d_locate_array_impl,
    interp1d_locate_scalar,
    interp1d_scalar,
    interp2d_array,
    interp2d_array_impl,
    interp2d_eval_array,
    interp2d_eval_array_impl,
    interp2d_eval_point as _interp2d_eval_point,
    interp2d_locate_array,
    interp2d_locate_array_impl,
    interp2d_locate_scalar,
    interp2d_locate_scalar_impl,
    interp2d_scalar,
    interp3d_array,
    interp3d_array_impl,
    interp3d_eval_array,
    interp3d_eval_array_impl,
    interp3d_eval_point as _interp3d_eval_point,
    interp3d_locate_array,
    interp3d_locate_array_impl,
    interp3d_locate_point,
    interp3d_locate_point_impl,
    interp3d_point,
)

__all__ = [
    'interp1d',
    'interp1d_eval',
    'interp1d_locate',
    'interp2d',
    'interp2d_eval',
    'interp2d_locate',
    'interp3d',
    'interp3d_eval',
    'interp3d_locate',
]

type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type IntegerScalar = int | np.integer[Any]
type ArrayQuery = Sequence[RealScalar] | np.ndarray
type FloatArray = NDArray[np.float64]
type IndexArray = NDArray[np.int64]
type InitialIndex2D = Sequence[IntegerScalar] | np.ndarray | None
type InitialIndex3D = Sequence[IntegerScalar] | np.ndarray | None
type ScalarIndex2D = tuple[IntegerScalar, IntegerScalar]
type ScalarIndex3D = tuple[IntegerScalar, IntegerScalar, IntegerScalar]
type ScalarWeight2D = tuple[RealScalar, RealScalar]
type ScalarWeight3D = tuple[RealScalar, RealScalar, RealScalar]

_interp1d_locate_scalar_jit = jit(interp1d_locate_scalar, **JIT_OPTIONS)
_interp1d_locate_array_jit = jit(interp1d_locate_array_impl, **JIT_OPTIONS)
_interp1d_eval_point_jit = jit(interp1d_eval_point, **JIT_OPTIONS)
_interp1d_eval_array_jit = jit(interp1d_eval_array_impl, **JIT_OPTIONS)
_interp1d_scalar_jit = jit(interp1d_scalar, **JIT_OPTIONS)
_interp1d_array_jit = jit(interp1d_array_impl, **JIT_OPTIONS)
_interp2d_locate_scalar_jit = jit(interp2d_locate_scalar_impl, **JIT_OPTIONS)
_interp2d_locate_array_jit = jit(interp2d_locate_array_impl, **JIT_OPTIONS)
_interp2d_eval_point_jit = jit(_interp2d_eval_point, **JIT_OPTIONS)
_interp2d_eval_point_c_jit = jit(_interp2d_eval_point_c, **JIT_OPTIONS)
_interp2d_eval_array_jit = jit(interp2d_eval_array_impl, **JIT_OPTIONS)
_interp2d_scalar_jit = jit(interp2d_scalar, **JIT_OPTIONS)
_interp2d_array_jit = jit(interp2d_array_impl, **JIT_OPTIONS)
_interp3d_locate_point_jit = jit(interp3d_locate_point_impl, **JIT_OPTIONS)
_interp3d_locate_array_jit = jit(interp3d_locate_array_impl, **JIT_OPTIONS)
_interp3d_eval_point_jit = jit(_interp3d_eval_point, **JIT_OPTIONS)
_interp3d_eval_point_c_jit = jit(_interp3d_eval_point_c, **JIT_OPTIONS)
_interp3d_eval_array_jit = jit(interp3d_eval_array_impl, **JIT_OPTIONS)
_interp3d_point_jit = jit(interp3d_point, **JIT_OPTIONS)
_interp3d_point_c_jit = jit(_interp3d_point_c, **JIT_OPTIONS)
_interp3d_array_jit = jit(interp3d_array_impl, **JIT_OPTIONS)


def _is_supported_real_dtype(dtype: np.dtype[Any]) -> bool:
    """Return whether a dtype satisfies the interpolation numeric contract."""
    if dtype.kind in 'iu':
        return dtype.itemsize <= 8
    return dtype == np.dtype(np.float32) or dtype == np.dtype(np.float64)


def _validate_real_array(array: np.ndarray, name: str, ndim: int | None = None) -> None:
    """Validate a NumPy array containing supported real values."""
    if not isinstance(array, np.ndarray):
        msg = f'{name} must be a NumPy array'
        raise TypeError(msg)
    if ndim is not None and array.ndim != ndim:
        msg = f'{name} must be {ndim}-dimensional'
        raise ValueError(msg)
    if not _is_supported_real_dtype(array.dtype):
        msg = f'{name} must have an integer, float32, or float64 dtype'
        raise TypeError(msg)


def _validate_grid(xp: np.ndarray, name: str) -> None:
    """Validate a one-dimensional interpolation grid."""
    _validate_real_array(xp, name, ndim=1)
    if xp.size < 2:
        msg = f'{name} must contain at least two points'
        raise ValueError(msg)
    if not np.all(np.isfinite(xp)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    if not np.all(xp[1:] > xp[:-1]):
        msg = f'{name} must be strictly increasing'
        raise ValueError(msg)


def _validate_real_scalar(value: RealScalar, name: str) -> None:
    """Validate a supported real scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or not _is_supported_real_dtype(array.dtype):
        msg = f'{name} must be a real scalar with a supported dtype'
        raise TypeError(msg)


def _normalize_query(
    value: RealScalar | ArrayQuery,
    name: str,
) -> tuple[bool, np.ndarray]:
    """Convert a query to a contiguous array and identify scalar inputs."""
    scalar = np.isscalar(value) and not isinstance(value, np.ndarray)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must contain real numeric values'
        raise TypeError(msg) from exc
    if not _is_supported_real_dtype(array.dtype):
        msg = f'{name} must contain integer, float32, or float64 values'
        raise TypeError(msg)
    if array.ndim == 0:
        return scalar, array
    return scalar, np.ascontiguousarray(array)


def _as_contiguous(array: np.ndarray) -> np.ndarray:
    """Make an array contiguous without changing zero-dimensional shape."""
    if array.ndim == 0:
        return array
    return np.ascontiguousarray(array)


def _normalize_index(ilb: IntegerScalar, size: int, name: str = 'ilb') -> int:
    """Convert and clamp an initial lower-bound index."""
    try:
        index = operator_index(ilb)
    except TypeError as exc:
        msg = f'{name} must be an integer'
        raise TypeError(msg) from exc
    return max(0, min(index, size - 2))


def _normalize_indices_2d(
    ilb: InitialIndex2D,
    size0: int,
    size1: int,
) -> IndexArray:
    """Normalize and clamp two initial lower-bound indices."""
    if ilb is None:
        return np.zeros(2, dtype=np.int64)

    array = np.asarray(ilb)
    if array.shape != (2,) or array.dtype.kind not in 'iu':
        msg = 'ilb must contain exactly two integer indices'
        raise ValueError(msg)
    index = np.empty(2, dtype=np.int64)
    index[0] = max(0, min(operator_index(array[0]), size0 - 2))
    index[1] = max(0, min(operator_index(array[1]), size1 - 2))
    return index


def _normalize_indices_3d(
    ilb: InitialIndex3D,
    size0: int,
    size1: int,
    size2: int,
) -> IndexArray:
    """Prepare three search hints for the checked Python entry points.

    Missing hints start at the first interval. Supplied hints must be an integer
    triple and are clamped independently because they guide the search rather than
    identify corners that will be evaluated directly.
    """
    if ilb is None:
        return np.zeros(3, dtype=np.int64)

    array = np.asarray(ilb)
    if array.shape != (3,) or array.dtype.kind not in 'iu':
        msg = 'ilb must contain exactly three integer indices'
        raise ValueError(msg)
    index = np.empty(3, dtype=np.int64)
    index[0] = max(0, min(operator_index(array[0]), size0 - 2))
    index[1] = max(0, min(operator_index(array[1]), size1 - 2))
    index[2] = max(0, min(operator_index(array[2]), size2 - 2))
    return index


def _result_dtype(*values: Any) -> np.dtype[np.float64]:
    """Return the common floating dtype used for allocated results."""
    dtype = np.result_type(*values, np.float64)
    if dtype != np.dtype(np.float64):
        msg = 'interpolation inputs cannot be promoted safely to float64'
        raise TypeError(msg)
    return np.dtype(np.float64)


def _validate_output(
    out: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    name: str,
) -> None:
    """Validate a floating-point output buffer."""
    if not isinstance(out, np.ndarray):
        msg = f'{name} must be a NumPy array'
        raise TypeError(msg)
    if out.shape != shape:
        msg = f'{name} must have shape {shape}, got {out.shape}'
        raise ValueError(msg)
    if not _is_supported_real_dtype(out.dtype) or out.dtype.kind != 'f':
        msg = f'{name} must have a supported floating dtype'
        raise TypeError(msg)
    if not np.can_cast(dtype, out.dtype, casting='safe'):
        msg = f'{name} dtype {out.dtype} cannot safely represent {dtype}'
        raise TypeError(msg)
    if not out.flags.writeable:
        msg = f'{name} must be writable'
        raise ValueError(msg)


def _prepare_float_output(
    out: np.ndarray | None,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    name: str = 'out',
) -> np.ndarray:
    """Allocate or validate a floating-point output buffer."""
    if out is None:
        return np.empty(shape, dtype=dtype)
    _validate_output(out, shape, dtype, name)
    return out


def _prepare_index_output(
    out: np.ndarray | None,
    shape: tuple[int, ...],
    name: str = 'index_out',
) -> np.ndarray:
    """Allocate or validate an int64 output buffer."""
    if out is None:
        return np.empty(shape, dtype=np.int64)
    if not isinstance(out, np.ndarray):
        msg = f'{name} must be a NumPy array'
        raise TypeError(msg)
    if out.shape != shape:
        msg = f'{name} must have shape {shape}, got {out.shape}'
        raise ValueError(msg)
    if out.dtype != np.dtype(np.int64):
        msg = f'{name} must have dtype int64'
        raise TypeError(msg)
    if not out.flags.writeable:
        msg = f'{name} must be writable'
        raise ValueError(msg)
    return out


@typing_overload
def interp1d_locate(
    x: RealScalar,
    xp: np.ndarray,
    ilb: IntegerScalar = 0,
    index_out: None = None,
    weight_out: None = None,
) -> tuple[int, float]: ...


@typing_overload
def interp1d_locate(
    x: ArrayQuery,
    xp: np.ndarray,
    ilb: IntegerScalar = 0,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[IndexArray, FloatArray]: ...


def interp1d_locate(
    x: RealScalar | ArrayQuery,
    xp: np.ndarray,
    ilb: IntegerScalar = 0,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[int, float] | tuple[IndexArray, FloatArray]:
    """Locate one-dimensional interpolation brackets and lower-point weights.

    Parameters
    ----------
    x
        Scalar, sequence, or array of query coordinates. A NumPy array, including
        a zero-dimensional array, follows the array return path.
    xp
        Strictly increasing one-dimensional NumPy grid with at least two points.
    ilb
        Initial lower-bound guess, clamped to the valid range.
    index_out
        Optional writable int64 buffer with the query shape. Only supported for
        array-valued queries.
    weight_out
        Optional writable float64 buffer with the query shape. Only supported for
        array-valued queries.

    Returns
    -------
    index
        Lower-bound index for a scalar query or an int64 array for an array query.
    weight
        Lower-grid-point weight for a scalar query or a float64 array for an array
        query. Supplied output buffers are returned by identity.

    Raises
    ------
    TypeError
        If an input has an unsupported type or dtype, or output buffers are supplied
        for a scalar query.
    ValueError
        If the grid is invalid or an output buffer has an invalid shape or is not
        writable.

    Notes
    -----
    A weight of one selects the lower grid point; a weight of zero selects the
    upper grid point. Weights outside ``[0, 1]`` identify extrapolated points.
    """
    _validate_grid(xp, 'xp')
    scalar, xx = _normalize_query(x, 'x')
    index0 = _normalize_index(ilb, xp.size)

    if scalar:
        if index_out is not None or weight_out is not None:
            msg = 'scalar interp1d_locate calls do not accept output buffers'
            raise TypeError(msg)
        index, weight = _interp1d_locate_scalar_jit(xx.item(), xp, index0)
        return int(index), float(weight)

    dtype = _result_dtype(xx, xp)
    index = _prepare_index_output(index_out, xx.shape)
    weight = _prepare_float_output(weight_out, xx.shape, dtype, 'weight_out')
    _interp1d_locate_array_jit(xx, xp, index0, index, weight)
    return index, weight


@typing_overload
def interp1d_eval(
    index: IntegerScalar,
    weight: RealScalar,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: RealScalar = np.nan,
    right: RealScalar = np.nan,
    out: None = None,
) -> float: ...


@typing_overload
def interp1d_eval(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: RealScalar = np.nan,
    right: RealScalar = np.nan,
    out: FloatArray | None = None,
) -> FloatArray: ...


def interp1d_eval(
    index: IntegerScalar | np.ndarray,
    weight: RealScalar | np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: RealScalar = np.nan,
    right: RealScalar = np.nan,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Evaluate a one-dimensional interpolant from indices and weights.

    Parameters
    ----------
    index
        Scalar lower-bound index or an integer NumPy array. Every index must satisfy
        ``0 <= index < len(fp) - 1``.
    weight
        Lower-grid-point weight with the same scalar or array category and shape as
        ``index``.
    fp
        One-dimensional NumPy array containing at least two function values.
    extrapolate
        Whether to evaluate the linear extrapolant when a weight lies outside
        ``[0, 1]``.
    left
        Value used below the grid when ``extrapolate`` is false.
    right
        Value used above the grid when ``extrapolate`` is false.
    out
        Optional writable float64 buffer with the index shape. Only supported for
        array-valued inputs.

    Returns
    -------
    A Python float for scalar inputs, a newly allocated float64 array for array
    inputs, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If inputs have incompatible scalar and array categories or unsupported
        types or dtypes, or ``out`` is supplied for scalar inputs.
    ValueError
        If ``fp`` or the input and output shapes are invalid, or ``out`` is not
        writable.
    IndexError
        If any lower-bound index is outside the valid range.
    """
    _validate_real_array(fp, 'fp', ndim=1)
    if fp.size < 2:
        msg = 'fp must contain at least two values'
        raise ValueError(msg)
    _validate_real_scalar(left, 'left')
    _validate_real_scalar(right, 'right')
    left_value = float(left)
    right_value = float(right)

    scalar_index = np.isscalar(index) and not isinstance(index, np.ndarray)
    scalar_weight = np.isscalar(weight) and not isinstance(weight, np.ndarray)
    if scalar_index != scalar_weight:
        msg = 'index and weight must both be scalars or both be arrays'
        raise TypeError(msg)

    index_array = np.asarray(index)
    weight_array = np.asarray(weight)
    if index_array.dtype.kind not in 'iu':
        msg = 'index must contain integers'
        raise TypeError(msg)
    if not _is_supported_real_dtype(weight_array.dtype):
        msg = 'weight must contain integer, float32, or float64 values'
        raise TypeError(msg)
    if index_array.shape != weight_array.shape:
        msg = 'index and weight must have equal shapes'
        raise ValueError(msg)
    if np.any(index_array < 0) or np.any(index_array >= fp.size - 1):
        msg = 'index values must satisfy 0 <= index < len(fp) - 1'
        raise IndexError(msg)

    if scalar_index:
        if out is not None:
            msg = 'scalar interp1d_eval calls do not accept an output buffer'
            raise TypeError(msg)
        value = _interp1d_eval_point_jit(
            operator_index(index_array.item()),
            weight_array.item(),
            fp,
            extrapolate,
            left_value,
            right_value,
        )
        return float(value)

    index_work = np.ascontiguousarray(index_array)
    weight_work = np.ascontiguousarray(weight_array)
    dtype = _result_dtype(weight_work, fp, left_value, right_value)
    result = _prepare_float_output(out, index_work.shape, dtype)
    _interp1d_eval_array_jit(
        index_work,
        weight_work,
        fp,
        extrapolate,
        left_value,
        right_value,
        result,
    )
    return result


@typing_overload
def interp1d(
    x: RealScalar,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: IntegerScalar = 0,
    extrapolate: bool = True,
    left: RealScalar = np.nan,
    right: RealScalar = np.nan,
    out: None = None,
) -> float: ...


@typing_overload
def interp1d(
    x: ArrayQuery,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: IntegerScalar = 0,
    extrapolate: bool = True,
    left: RealScalar = np.nan,
    right: RealScalar = np.nan,
    out: FloatArray | None = None,
) -> FloatArray: ...


def interp1d(
    x: RealScalar | ArrayQuery,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: IntegerScalar = 0,
    extrapolate: bool = True,
    left: RealScalar = np.nan,
    right: RealScalar = np.nan,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Perform one-dimensional linear interpolation.

    Parameters
    ----------
    x
        Scalar, sequence, or array of query coordinates. A NumPy array, including
        a zero-dimensional array, follows the array return path.
    xp
        Strictly increasing one-dimensional NumPy grid with at least two points.
    fp
        One-dimensional NumPy array of function values with the same shape as
        ``xp``.
    ilb
        Initial lower-bound guess, clamped to the valid range.
    extrapolate
        Whether to linearly extrapolate outside the grid.
    left
        Value used below the grid when ``extrapolate`` is false.
    right
        Value used above the grid when ``extrapolate`` is false.
    out
        Optional writable float64 buffer with the query shape. Only supported for
        array-valued queries.

    Returns
    -------
    A Python float for a scalar query, a newly allocated float64 array for an array
    query, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If an input has an unsupported type or dtype, or ``out`` is supplied for a
        scalar query.
    ValueError
        If the grid, function values, or output buffer is not conformable.
    """
    _validate_grid(xp, 'xp')
    _validate_real_array(fp, 'fp', ndim=1)
    if xp.shape != fp.shape:
        msg = 'xp and fp must have equal shapes'
        raise ValueError(msg)
    _validate_real_scalar(left, 'left')
    _validate_real_scalar(right, 'right')
    left_value = float(left)
    right_value = float(right)

    scalar, xx = _normalize_query(x, 'x')
    index0 = _normalize_index(ilb, xp.size)
    if scalar:
        if out is not None:
            msg = 'scalar interp1d calls do not accept an output buffer'
            raise TypeError(msg)
        value = _interp1d_scalar_jit(
            xx.item(),
            xp,
            fp,
            index0,
            extrapolate,
            left_value,
            right_value,
        )
        return float(value)

    dtype = _result_dtype(xx, xp, fp, left_value, right_value)
    result = _prepare_float_output(out, xx.shape, dtype)
    _interp1d_array_jit(
        xx,
        xp,
        fp,
        index0,
        extrapolate,
        left_value,
        right_value,
        result,
    )
    return result


@typing_overload
def interp2d_locate(
    x0: RealScalar,
    x1: RealScalar,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: InitialIndex2D = None,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[IndexArray, FloatArray]: ...


@typing_overload
def interp2d_locate(
    x0: RealScalar | ArrayQuery,
    x1: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: InitialIndex2D = None,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[IndexArray, FloatArray]: ...


def interp2d_locate(
    x0: RealScalar | ArrayQuery,
    x1: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: InitialIndex2D = None,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[IndexArray, FloatArray]:
    """Locate two-dimensional samples and lower-point weights.

    Parameters
    ----------
    x0
        Scalar, sequence, or array of coordinates along the first axis.
    x1
        Scalar, sequence, or array of coordinates along the second axis. Python
        calls broadcast ``x0`` and ``x1`` using NumPy broadcasting rules.
    xp0
        Strictly increasing one-dimensional NumPy grid for the first axis.
    xp1
        Strictly increasing one-dimensional NumPy grid for the second axis.
    ilb
        Optional pair of initial lower-bound guesses, clamped independently.
    index_out
        Optional writable int64 buffer with shape ``sample_shape + (2,)``.
    weight_out
        Optional writable float64 buffer with shape ``sample_shape + (2,)``.

    Returns
    -------
    index
        Lower-bound indices with shape ``sample_shape + (2,)``.
    weight
        Lower-grid-point weights with shape ``sample_shape + (2,)``. Supplied
        buffers are returned by identity.

    Raises
    ------
    TypeError
        If an input or output buffer has an unsupported type or dtype.
    ValueError
        If a grid is invalid, coordinates cannot be broadcast, or an output buffer
        has an invalid shape or is not writable.

    Notes
    -----
    For two scalar coordinates, ``sample_shape`` is empty and both returned arrays
    have shape ``(2,)``. Jitted array calls require conformable coordinate shapes;
    the unchecked Numba path does not perform broadcasting.
    """
    _validate_grid(xp0, 'xp0')
    _validate_grid(xp1, 'xp1')
    scalar0, xx0 = _normalize_query(x0, 'x0')
    scalar1, xx1 = _normalize_query(x1, 'x1')
    try:
        xx0, xx1 = np.broadcast_arrays(xx0, xx1)
    except ValueError as exc:
        msg = 'x0 and x1 cannot be broadcast to a common shape'
        raise ValueError(msg) from exc
    xx0 = _as_contiguous(xx0)
    xx1 = _as_contiguous(xx1)

    shape = (*xx0.shape, 2)
    dtype = _result_dtype(xx0, xx1, xp0, xp1)
    index = _prepare_index_output(index_out, shape)
    weight = _prepare_float_output(weight_out, shape, dtype, 'weight_out')
    index0 = _normalize_indices_2d(ilb, xp0.size, xp1.size)

    if scalar0 and scalar1:
        _interp2d_locate_scalar_jit(
            xx0.item(), xx1.item(), xp0, xp1, index0, index, weight
        )
    else:
        _interp2d_locate_array_jit(xx0, xx1, xp0, xp1, index0, index, weight)
    return index, weight


@typing_overload
def interp2d_eval(
    index: ScalarIndex2D,
    weight: ScalarWeight2D,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: None = None,
) -> float: ...


@typing_overload
def interp2d_eval(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> float | FloatArray: ...


def interp2d_eval(
    index: ScalarIndex2D | np.ndarray,
    weight: ScalarWeight2D | np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Evaluate a bilinear interpolant from indices and weights.

    Parameters
    ----------
    index
        A length-two tuple of scalar lower-bound indices for one point, or an integer
        NumPy array with shape ``sample_shape + (2,)``.
    weight
        Lower-grid-point weights with the same category and shape as ``index``. A
        tuple is evaluated without temporary arrays in Numba-compiled callers.
    fp
        Two-dimensional NumPy array with at least two function values on each axis.
    extrapolate
        Whether to evaluate the bilinear extrapolant outside either grid. If false,
        an exterior point receives ``NaN``.
    out
        Optional writable float64 buffer with ``sample_shape``. Not supported for
        tuple inputs or when ``index`` is an array with shape ``(2,)``.

    Returns
    -------
    A Python float for one point, a newly allocated float64 array for multiple
    points, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If inputs have incompatible tuple and array categories or unsupported types
        or dtypes, or ``out`` is supplied for one point.
    ValueError
        If input or output shapes are invalid, ``fp`` is not conformable, or ``out``
        is not writable.
    IndexError
        If any lower-bound index is outside the corresponding valid range.
    """
    _validate_real_array(fp, 'fp', ndim=2)
    if fp.shape[0] < 2 or fp.shape[1] < 2:
        msg = 'fp must have at least two values on each axis'
        raise ValueError(msg)

    if isinstance(index, tuple):
        if not isinstance(weight, tuple):
            msg = 'index and weight must both be arrays or both be length-two tuples'
            raise TypeError(msg)
        if len(index) != 2 or len(weight) != 2:
            msg = 'tuple index and weight inputs must have length two'
            raise ValueError(msg)
        if out is not None:
            msg = 'single-point interp2d_eval calls do not accept an output buffer'
            raise TypeError(msg)

        try:
            index0 = operator_index(index[0])
            index1 = operator_index(index[1])
        except TypeError as exc:
            msg = 'tuple index inputs must contain integer scalars'
            raise TypeError(msg) from exc
        _validate_real_scalar(weight[0], 'weight[0]')
        _validate_real_scalar(weight[1], 'weight[1]')
        if (
            index0 < 0
            or index0 >= fp.shape[0] - 1
            or index1 < 0
            or index1 >= fp.shape[1] - 1
        ):
            msg = 'index values are outside the valid lower-bound ranges'
            raise IndexError(msg)

        if fp.flags.c_contiguous:
            value = _interp2d_eval_point_c_jit(index, weight, fp, extrapolate)
        else:
            value = _interp2d_eval_point_jit(index, weight, fp, extrapolate)
        return float(value)

    if not isinstance(index, np.ndarray) or not isinstance(weight, np.ndarray):
        msg = 'index and weight must both be arrays or both be length-two tuples'
        raise TypeError(msg)
    if index.dtype.kind not in 'iu':
        msg = 'index must have an integer dtype'
        raise TypeError(msg)
    _validate_real_array(weight, 'weight')
    if index.shape != weight.shape:
        msg = 'index and weight must have equal shapes'
        raise ValueError(msg)
    if index.ndim < 1 or index.shape[-1] != 2:
        msg = 'index and weight must end in a coordinate dimension of length two'
        raise ValueError(msg)
    if (
        np.any(index[..., 0] < 0)
        or np.any(index[..., 0] >= fp.shape[0] - 1)
        or np.any(index[..., 1] < 0)
        or np.any(index[..., 1] >= fp.shape[1] - 1)
    ):
        msg = 'index values are outside the valid lower-bound ranges'
        raise IndexError(msg)

    index_work = np.ascontiguousarray(index)
    weight_work = np.ascontiguousarray(weight)
    if index.ndim == 1:
        if out is not None:
            msg = 'single-point interp2d_eval calls do not accept an output buffer'
            raise TypeError(msg)
        value = _interp2d_eval_point_jit(index_work, weight_work, fp, extrapolate)
        return float(value)

    shape = index.shape[:-1]
    dtype = _result_dtype(weight_work, fp)
    result = _prepare_float_output(out, shape, dtype)
    _interp2d_eval_array_jit(index_work, weight_work, fp, extrapolate, result)
    return result


@typing_overload
def interp2d(
    x0: RealScalar,
    x1: RealScalar,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex2D = None,
    extrapolate: bool = True,
    out: None = None,
) -> float: ...


@typing_overload
def interp2d(
    x0: ArrayQuery,
    x1: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex2D = None,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> FloatArray: ...


@typing_overload
def interp2d(
    x0: RealScalar,
    x1: ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex2D = None,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> FloatArray: ...


def interp2d(
    x0: RealScalar | ArrayQuery,
    x1: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex2D = None,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Perform bilinear interpolation at two-dimensional query coordinates.

    Parameters
    ----------
    x0
        Scalar, sequence, or array of coordinates along the first axis.
    x1
        Scalar, sequence, or array of coordinates along the second axis. Python
        calls broadcast ``x0`` and ``x1`` using NumPy broadcasting rules.
    xp0
        Strictly increasing one-dimensional NumPy grid for the first axis.
    xp1
        Strictly increasing one-dimensional NumPy grid for the second axis.
    fp
        Two-dimensional NumPy array with shape ``(len(xp0), len(xp1))``.
    ilb
        Optional pair of initial lower-bound guesses, clamped independently.
    extrapolate
        Whether to evaluate the bilinear extrapolant outside either grid. If false,
        an exterior point receives ``NaN``.
    out
        Optional writable float64 buffer with the broadcast coordinate shape. Only
        supported when at least one coordinate input follows the array path.

    Returns
    -------
    A Python float for two scalar coordinates, a newly allocated float64 array for
    array-valued coordinates, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If an input or output buffer has an unsupported type or dtype, or ``out`` is
        supplied for two scalar coordinates.
    ValueError
        If a grid or ``fp`` is invalid, coordinates cannot be broadcast, or an
        output buffer has an invalid shape or is not writable.

    Notes
    -----
    Jitted array calls require conformable coordinate shapes; the unchecked Numba
    path does not perform broadcasting.
    """
    _validate_grid(xp0, 'xp0')
    _validate_grid(xp1, 'xp1')
    _validate_real_array(fp, 'fp', ndim=2)
    expected = (xp0.size, xp1.size)
    if fp.shape != expected:
        msg = f'fp must have shape {expected}, got {fp.shape}'
        raise ValueError(msg)

    scalar0, xx0 = _normalize_query(x0, 'x0')
    scalar1, xx1 = _normalize_query(x1, 'x1')
    try:
        xx0, xx1 = np.broadcast_arrays(xx0, xx1)
    except ValueError as exc:
        msg = 'x0 and x1 cannot be broadcast to a common shape'
        raise ValueError(msg) from exc
    xx0 = _as_contiguous(xx0)
    xx1 = _as_contiguous(xx1)
    index0 = _normalize_indices_2d(ilb, xp0.size, xp1.size)

    if scalar0 and scalar1:
        if out is not None:
            msg = 'scalar interp2d calls do not accept an output buffer'
            raise TypeError(msg)
        value = _interp2d_scalar_jit(
            xx0.item(),
            xx1.item(),
            xp0,
            xp1,
            fp,
            index0,
            extrapolate,
        )
        return float(value)

    dtype = _result_dtype(xx0, xx1, xp0, xp1, fp)
    result = _prepare_float_output(out, xx0.shape, dtype)
    _interp2d_array_jit(
        xx0,
        xx1,
        xp0,
        xp1,
        fp,
        index0,
        extrapolate,
        result,
    )
    return result


@typing_overload
def interp3d_locate(
    x0: RealScalar,
    x1: RealScalar,
    x2: RealScalar,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    ilb: InitialIndex3D = None,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[IndexArray, FloatArray]: ...


@typing_overload
def interp3d_locate(
    x0: RealScalar | ArrayQuery,
    x1: RealScalar | ArrayQuery,
    x2: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    ilb: InitialIndex3D = None,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[IndexArray, FloatArray]: ...


def interp3d_locate(
    x0: RealScalar | ArrayQuery,
    x1: RealScalar | ArrayQuery,
    x2: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    ilb: InitialIndex3D = None,
    index_out: IndexArray | None = None,
    weight_out: FloatArray | None = None,
) -> tuple[IndexArray, FloatArray]:
    """Locate three-dimensional samples and lower-point weights.

    Each coordinate is located independently with the optimized 1D search. Python
    coordinate inputs are broadcast to a common sample shape before entering the
    point or array kernel. Returned indices and weights have shape
    ``sample_shape + (3,)``; supplied output buffers are returned by identity.

    The weights select the lower grid point: one selects the lower endpoint, zero
    selects the upper endpoint, and values outside ``[0, 1]`` indicate
    extrapolation. ``ilb`` values are search hints, so valid integer triples are
    clamped instead of rejected when they lie outside the grids.

    Parameters
    ----------
    x0
        Coordinates along the first axis.
    x1
        Coordinates along the second axis.
    x2
        Coordinates along the third axis.
    xp0
        Strictly increasing grid for the first axis.
    xp1
        Strictly increasing grid for the second axis.
    xp2
        Strictly increasing grid for the third axis.
    ilb
        Optional three initial lower-bound guesses, clamped independently.
    index_out
        Optional writable int64 buffer with shape ``sample_shape + (3,)``.
    weight_out
        Optional writable float64 buffer with shape ``sample_shape + (3,)``.

    Returns
    -------
    index
        Lower-bound indices for every coordinate.
    weight
        Corresponding lower-grid-point weights.

    Raises
    ------
    TypeError
        If a coordinate, grid, or output buffer has an unsupported dtype.
    ValueError
        If a grid is invalid, coordinates cannot be broadcast, a hint is not an
        integer triple, or an output buffer is not conformable and writable.

    Notes
    -----
    Numba array calls use the unchecked equal-shaped path and do not perform
    broadcasting. Point calls may reuse caller-provided length-three buffers.
    """
    _validate_grid(xp0, 'xp0')
    _validate_grid(xp1, 'xp1')
    _validate_grid(xp2, 'xp2')
    scalar0, xx0 = _normalize_query(x0, 'x0')
    scalar1, xx1 = _normalize_query(x1, 'x1')
    scalar2, xx2 = _normalize_query(x2, 'x2')
    try:
        xx0, xx1, xx2 = np.broadcast_arrays(xx0, xx1, xx2)
    except ValueError as exc:
        msg = 'x0, x1, and x2 cannot be broadcast to a common shape'
        raise ValueError(msg) from exc
    xx0 = _as_contiguous(xx0)
    xx1 = _as_contiguous(xx1)
    xx2 = _as_contiguous(xx2)

    shape = (*xx0.shape, 3)
    dtype = _result_dtype(xx0, xx1, xx2, xp0, xp1, xp2)
    index = _prepare_index_output(index_out, shape)
    weight = _prepare_float_output(weight_out, shape, dtype, 'weight_out')
    index0 = _normalize_indices_3d(ilb, xp0.size, xp1.size, xp2.size)

    if scalar0 and scalar1 and scalar2:
        _interp3d_locate_point_jit(
            xx0.item(),
            xx1.item(),
            xx2.item(),
            xp0,
            xp1,
            xp2,
            index0,
            index,
            weight,
        )
    else:
        _interp3d_locate_array_jit(xx0, xx1, xx2, xp0, xp1, xp2, index0, index, weight)
    return index, weight


@typing_overload
def interp3d_eval(
    index: ScalarIndex3D,
    weight: ScalarWeight3D,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: None = None,
) -> float: ...


@typing_overload
def interp3d_eval(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> float | FloatArray: ...


def interp3d_eval(
    index: ScalarIndex3D | np.ndarray,
    weight: ScalarWeight3D | np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Evaluate a trilinear interpolant from indices and weights.

    Evaluation follows a fixed arithmetic tree: interpolate axis 0 at four corner
    pairs, axis 1 at the resulting two pairs, and axis 2 last. This order matches
    the lower-point weight convention and is retained across C-contiguous and
    arbitrary-strided kernels.

    Length-three tuples provide an allocation-free point representation for Numba
    callers that reuse one location across several fields. C-contiguous point
    values use a single flattened base offset; arbitrary-strided values retain
    explicit plane and row addressing.

    Parameters
    ----------
    index
        A length-three tuple for one point, or an integer array with shape
        ``sample_shape + (3,)``.
    weight
        Lower-grid-point weights with the same category and shape as ``index``.
    fp
        Three-dimensional function values with at least two values per axis.
    extrapolate
        Whether to evaluate the trilinear extrapolant outside any grid. If false,
        an exterior point receives ``NaN``.
    out
        Optional writable float64 buffer with ``sample_shape``. Not supported for
        a single point.

    Returns
    -------
    A Python float for one point, an allocated float64 array for multiple points,
    or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If indices and weights use incompatible representations or unsupported
        dtypes, or a point call receives an output buffer.
    ValueError
        If the arrays have incompatible shapes, ``fp`` is not three-dimensional,
        or an output buffer is not conformable and writable.
    IndexError
        If any lower-bound index is outside its valid grid interval range.

    Notes
    -----
    When ``extrapolate`` is false, a weight outside ``[0, 1]`` produces ``NaN``.
    Non-finite weights otherwise propagate through the interpolation arithmetic.
    """
    _validate_real_array(fp, 'fp', ndim=3)
    if fp.shape[0] < 2 or fp.shape[1] < 2 or fp.shape[2] < 2:
        msg = 'fp must have at least two values on each axis'
        raise ValueError(msg)

    if isinstance(index, tuple):
        if not isinstance(weight, tuple):
            msg = 'index and weight must both be arrays or both be length-three tuples'
            raise TypeError(msg)
        if len(index) != 3 or len(weight) != 3:
            msg = 'tuple index and weight inputs must have length three'
            raise ValueError(msg)
        if out is not None:
            msg = 'single-point interp3d_eval calls do not accept an output buffer'
            raise TypeError(msg)

        try:
            index0 = operator_index(index[0])
            index1 = operator_index(index[1])
            index2 = operator_index(index[2])
        except TypeError as exc:
            msg = 'tuple index inputs must contain integer scalars'
            raise TypeError(msg) from exc
        _validate_real_scalar(weight[0], 'weight[0]')
        _validate_real_scalar(weight[1], 'weight[1]')
        _validate_real_scalar(weight[2], 'weight[2]')
        if (
            index0 < 0
            or index0 >= fp.shape[0] - 1
            or index1 < 0
            or index1 >= fp.shape[1] - 1
            or index2 < 0
            or index2 >= fp.shape[2] - 1
        ):
            msg = 'index values are outside the valid lower-bound ranges'
            raise IndexError(msg)

        if fp.flags.c_contiguous:
            value = _interp3d_eval_point_c_jit(index, weight, fp, extrapolate)
        else:
            value = _interp3d_eval_point_jit(index, weight, fp, extrapolate)
        return float(value)

    if not isinstance(index, np.ndarray) or not isinstance(weight, np.ndarray):
        msg = 'index and weight must both be arrays or both be length-three tuples'
        raise TypeError(msg)
    if index.dtype.kind not in 'iu':
        msg = 'index must have an integer dtype'
        raise TypeError(msg)
    _validate_real_array(weight, 'weight')
    if index.shape != weight.shape:
        msg = 'index and weight must have equal shapes'
        raise ValueError(msg)
    if index.ndim < 1 or index.shape[-1] != 3:
        msg = 'index and weight must end in a coordinate dimension of length three'
        raise ValueError(msg)
    if (
        np.any(index[..., 0] < 0)
        or np.any(index[..., 0] >= fp.shape[0] - 1)
        or np.any(index[..., 1] < 0)
        or np.any(index[..., 1] >= fp.shape[1] - 1)
        or np.any(index[..., 2] < 0)
        or np.any(index[..., 2] >= fp.shape[2] - 1)
    ):
        msg = 'index values are outside the valid lower-bound ranges'
        raise IndexError(msg)

    index_work = np.ascontiguousarray(index)
    weight_work = np.ascontiguousarray(weight)
    if index.ndim == 1:
        if out is not None:
            msg = 'single-point interp3d_eval calls do not accept an output buffer'
            raise TypeError(msg)
        if fp.flags.c_contiguous:
            value = _interp3d_eval_point_c_jit(index_work, weight_work, fp, extrapolate)
        else:
            value = _interp3d_eval_point_jit(index_work, weight_work, fp, extrapolate)
        return float(value)

    shape = index.shape[:-1]
    dtype = _result_dtype(weight_work, fp)
    result = _prepare_float_output(out, shape, dtype)
    _interp3d_eval_array_jit(index_work, weight_work, fp, extrapolate, result)
    return result


@typing_overload
def interp3d(
    x0: RealScalar,
    x1: RealScalar,
    x2: RealScalar,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex3D = None,
    extrapolate: bool = True,
    out: None = None,
) -> float: ...


@typing_overload
def interp3d(
    x0: ArrayQuery,
    x1: RealScalar | ArrayQuery,
    x2: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex3D = None,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> FloatArray: ...


@typing_overload
def interp3d(
    x0: RealScalar,
    x1: ArrayQuery,
    x2: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex3D = None,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> FloatArray: ...


@typing_overload
def interp3d(
    x0: RealScalar,
    x1: RealScalar,
    x2: ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex3D = None,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> FloatArray: ...


def interp3d(
    x0: RealScalar | ArrayQuery,
    x1: RealScalar | ArrayQuery,
    x2: RealScalar | ArrayQuery,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: InitialIndex3D = None,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Perform trilinear interpolation at three-dimensional coordinates.

    This fused entry point locates each coordinate and evaluates the eight
    surrounding values without materializing intermediate index and weight arrays.
    It uses the same axis-0, axis-1, then axis-2 arithmetic tree as
    :func:`interp3d_eval`.

    Python inputs are broadcast before dispatch. Numba point calls select a
    C-contiguous flat-offset kernel or a correct arbitrary-strided kernel at typing
    time; Numba array calls require equal-shaped coordinates and keep the large
    loop out of line.

    Parameters
    ----------
    x0
        Coordinates along the first axis.
    x1
        Coordinates along the second axis.
    x2
        Coordinates along the third axis. Python calls broadcast all coordinates.
    xp0
        Strictly increasing grid for the first axis.
    xp1
        Strictly increasing grid for the second axis.
    xp2
        Strictly increasing grid for the third axis.
    fp
        Function values with shape ``(len(xp0), len(xp1), len(xp2))``.
    ilb
        Optional three initial lower-bound guesses, clamped independently.
    extrapolate
        Whether to evaluate outside the grids. If false, exterior points receive
        ``NaN``.
    out
        Optional writable float64 buffer with the broadcast coordinate shape. Only
        supported when at least one coordinate follows the array path.

    Returns
    -------
    A Python float for scalar coordinates, an allocated float64 array for array
    coordinates, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If an input or output buffer has an unsupported dtype, or a point call
        receives an output buffer.
    ValueError
        If a grid or ``fp`` is invalid, coordinates cannot be broadcast, or an
        output buffer is not conformable and writable.

    Notes
    -----
    Search hints are clamped independently. With extrapolation disabled, any
    coordinate outside its grid produces ``NaN``.
    """
    _validate_grid(xp0, 'xp0')
    _validate_grid(xp1, 'xp1')
    _validate_grid(xp2, 'xp2')
    _validate_real_array(fp, 'fp', ndim=3)
    expected = (xp0.size, xp1.size, xp2.size)
    if fp.shape != expected:
        msg = f'fp must have shape {expected}, got {fp.shape}'
        raise ValueError(msg)

    scalar0, xx0 = _normalize_query(x0, 'x0')
    scalar1, xx1 = _normalize_query(x1, 'x1')
    scalar2, xx2 = _normalize_query(x2, 'x2')
    try:
        xx0, xx1, xx2 = np.broadcast_arrays(xx0, xx1, xx2)
    except ValueError as exc:
        msg = 'x0, x1, and x2 cannot be broadcast to a common shape'
        raise ValueError(msg) from exc
    xx0 = _as_contiguous(xx0)
    xx1 = _as_contiguous(xx1)
    xx2 = _as_contiguous(xx2)
    index0 = _normalize_indices_3d(ilb, xp0.size, xp1.size, xp2.size)

    if scalar0 and scalar1 and scalar2:
        if out is not None:
            msg = 'single-point interp3d calls do not accept an output buffer'
            raise TypeError(msg)
        if fp.flags.c_contiguous:
            value = _interp3d_point_c_jit(
                xx0.item(),
                xx1.item(),
                xx2.item(),
                xp0,
                xp1,
                xp2,
                fp,
                index0,
                extrapolate,
            )
        else:
            value = _interp3d_point_jit(
                xx0.item(),
                xx1.item(),
                xx2.item(),
                xp0,
                xp1,
                xp2,
                fp,
                index0,
                extrapolate,
            )
        return float(value)

    dtype = _result_dtype(xx0, xx1, xx2, xp0, xp1, xp2, fp)
    result = _prepare_float_output(out, xx0.shape, dtype)
    _interp3d_array_jit(
        xx0,
        xx1,
        xx2,
        xp0,
        xp1,
        xp2,
        fp,
        index0,
        extrapolate,
        result,
    )
    return result


def _numba_none(value: Any) -> bool:
    """Return whether a Numba overload argument represents ``None``."""
    from numba import types

    return value is None or isinstance(value, (types.NoneType, types.Omitted))


def _numba_real_scalar(value: Any) -> bool:
    """Return whether a Numba type is a supported real scalar."""
    from numba import types

    return value in types.integer_domain or value in types.real_domain


def _numba_real_array(value: Any) -> bool:
    """Return whether a Numba type is an array with a real numeric dtype."""
    from numba import types

    return isinstance(value, types.Array) and (
        value.dtype in types.integer_domain or value.dtype in types.real_domain
    )


def _numba_integer_pair(value: Any) -> bool:
    """Return whether a Numba type is a length-two integer tuple."""
    from numba import types

    return (
        isinstance(value, types.BaseTuple)
        and len(value) == 2
        and all(item in types.integer_domain for item in value.types)
    )


def _numba_real_pair(value: Any) -> bool:
    """Return whether a Numba type is a length-two real numeric tuple."""
    from numba import types

    return (
        isinstance(value, types.BaseTuple)
        and len(value) == 2
        and all(_numba_real_scalar(item) for item in value.types)
    )


def _numba_integer_triplet(value: Any) -> bool:
    """Identify the allocation-free index representation for one 3D point.

    Both homogeneous and heterogeneous Numba tuples are accepted, but every
    component must belong to Numba's integer domain.
    """
    from numba import types

    return (
        isinstance(value, types.BaseTuple)
        and len(value) == 3
        and all(item in types.integer_domain for item in value.types)
    )


def _numba_real_triplet(value: Any) -> bool:
    """Identify the allocation-free weight representation for one 3D point.

    Fixed-size tuples let outer compiled kernels retain the three weights as scalar
    values instead of constructing a temporary NumPy array.
    """
    from numba import types

    return (
        isinstance(value, types.BaseTuple)
        and len(value) == 3
        and all(_numba_real_scalar(item) for item in value.types)
    )


@numba_overload(interp1d_locate, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp1d_locate_scalar_inline(
    x: Any,
    xp: Any,
    ilb: Any = 0,
    index_out: Any = None,
    weight_out: Any = None,
) -> Any:
    """Claim scalar locations for forced caller inlining.

    Exposing the small scalar kernel lets Numba scalarize its ``(index, weight)``
    result in hot loops. This template must not claim arrays: forcing the allocating
    array wrapper inline makes its output variables optional in the caller and also
    expands the full array loop there.
    """
    if _numba_real_scalar(x):
        if not _numba_none(index_out) or not _numba_none(weight_out):
            return None

        def impl(x, xp, ilb=0, index_out=None, weight_out=None):
            return interp1d_locate_scalar(x, xp, ilb)

        return impl
    return None


@numba_overload(interp1d_locate, jit_options=JIT_OPTIONS)
def _overload_interp1d_locate(
    x: Any,
    xp: Any,
    ilb: Any = 0,
    index_out: Any = None,
    weight_out: Any = None,
) -> Any:
    """Claim array locations without forcing the wrapper inline.

    The scalar template above owns scalar signatures. Keeping
    ``interp1d_locate_array`` out of line resolves its optional output-allocation
    branches inside the wrapper, so callers receive concrete arrays. Otherwise a
    chained ``interp1d_eval(..., out=...)`` can carry optional arrays into Numba's
    unsupported keyword-resolution path and fail with an internal assertion.
    The separate boundary also avoids copying the array loop into every caller.
    """
    if _numba_real_scalar(x):
        return None
    if _numba_real_array(x):
        return interp1d_locate_array
    return None


@numba_overload(interp1d_eval, jit_options=JIT_OPTIONS)
def _overload_interp1d_eval(
    index: Any,
    weight: Any,
    fp: Any,
    extrapolate: Any = True,
    left: Any = np.nan,
    right: Any = np.nan,
    out: Any = None,
) -> Any:
    from numba import types

    if index in types.integer_domain and _numba_real_scalar(weight):
        if not _numba_none(out):
            return None

        def impl(
            index,
            weight,
            fp,
            extrapolate=True,
            left=np.nan,
            right=np.nan,
            out=None,
        ):
            return interp1d_eval_point(
                index,
                weight,
                fp,
                extrapolate,
                left,
                right,
            )

        return impl
    if _numba_real_array(index) and _numba_real_array(weight):
        return interp1d_eval_array
    return None


@numba_overload(interp1d, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp1d_scalar_inline(
    x: Any,
    xp: Any,
    fp: Any,
    ilb: Any = 0,
    extrapolate: Any = True,
    left: Any = np.nan,
    right: Any = np.nan,
    out: Any = None,
) -> Any:
    # Keep scalar inlining separate: forcing the array overload inline expands hot loops.
    if _numba_real_scalar(x) and _numba_none(out):

        def impl(
            x,
            xp,
            fp,
            ilb=0,
            extrapolate=True,
            left=np.nan,
            right=np.nan,
            out=None,
        ):
            return interp1d_scalar(x, xp, fp, ilb, extrapolate, left, right)

        return impl
    return None


@numba_overload(interp1d, jit_options=JIT_OPTIONS)
def _overload_interp1d(
    x: Any,
    xp: Any,
    fp: Any,
    ilb: Any = 0,
    extrapolate: Any = True,
    left: Any = np.nan,
    right: Any = np.nan,
    out: Any = None,
) -> Any:
    # Scalar signatures are claimed above; array wrappers must remain out of line.
    if _numba_real_scalar(x):
        return None
    if _numba_real_array(x):
        return interp1d_array
    return None


@numba_overload(interp2d_locate, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp2d_locate_scalar_inline(
    x0: Any,
    x1: Any,
    xp0: Any,
    xp1: Any,
    ilb: Any = None,
    index_out: Any = None,
    weight_out: Any = None,
) -> Any:
    """Claim scalar locations for forced caller inlining.

    Inlining this point-sized wrapper lets Numba scalarize its two index and weight
    components, including caller-provided length-two buffers used in stateful hot
    loops. Array signatures are deliberately left to the normal overload below so
    their allocation branches and loops are not expanded into scalar callers.
    """
    if _numba_real_scalar(x0) and _numba_real_scalar(x1):
        return interp2d_locate_scalar
    return None


@numba_overload(interp2d_locate, jit_options=JIT_OPTIONS)
def _overload_interp2d_locate(
    x0: Any,
    x1: Any,
    xp0: Any,
    xp1: Any,
    ilb: Any = None,
    index_out: Any = None,
    weight_out: Any = None,
) -> Any:
    """Claim array locations without forcing the wrapper inline.

    The scalar template above owns point signatures. An always-inlined
    ``interp2d_locate_array`` exposes optional output branches to the caller; its
    returned index and weight values can then remain optional while typing a chained
    ``interp2d_eval(..., out=...)`` call. Keeping this template out of line avoids
    Numba's unsupported keyword-resolution path and prevents array-loop expansion.
    """
    if _numba_real_scalar(x0) and _numba_real_scalar(x1):
        return None
    if _numba_real_array(x0) and _numba_real_array(x1):
        return interp2d_locate_array
    return None


@numba_overload(interp2d_eval, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp2d_eval_strided(
    index: Any,
    weight: Any,
    fp: Any,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    from numba import types

    tuple_input = (
        _numba_integer_pair(index)
        and _numba_real_pair(weight)
        and _numba_real_array(fp)
        and fp.ndim == 2
    )
    array_input = isinstance(index, types.Array) and index.ndim == 1
    # A-layout row views need caller inlining; tuples scalarize without NRT allocation.
    if (
        (tuple_input or array_input)
        and getattr(fp, 'layout', None) == 'A'
        and _numba_none(out)
    ):

        def impl(index, weight, fp, extrapolate=True, out=None):
            return _interp2d_eval_point(index, weight, fp, extrapolate)

        return impl
    return None


@numba_overload(interp2d_eval, jit_options=JIT_OPTIONS)
def _overload_interp2d_eval(
    index: Any,
    weight: Any,
    fp: Any,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    from numba import types

    tuple_input = (
        _numba_integer_pair(index)
        and _numba_real_pair(weight)
        and _numba_real_array(fp)
        and fp.ndim == 2
    )
    array_input = isinstance(index, types.Array) and index.ndim == 1
    if tuple_input or array_input:
        # The always-inline template above exclusively owns arbitrary-strided points.
        if getattr(fp, 'layout', None) == 'A' or not _numba_none(out):
            return None

        # Keep this public C-layout call compact; only its flat-address leaf is inline.
        if getattr(fp, 'layout', None) == 'C':

            def impl(index, weight, fp, extrapolate=True, out=None):
                return _interp2d_eval_point_c(index, weight, fp, extrapolate)

        else:

            def impl(index, weight, fp, extrapolate=True, out=None):
                return _interp2d_eval_point(index, weight, fp, extrapolate)

        return impl
    if isinstance(index, types.Array) and index.ndim > 1:
        return interp2d_eval_array
    return None


@numba_overload(interp2d, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp2d_scalar_c(
    x0: Any,
    x1: Any,
    xp0: Any,
    xp1: Any,
    fp: Any,
    ilb: Any = None,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    # Isolate C scalars so forced inlining never expands generic or array implementations.
    if (
        _numba_real_scalar(x0)
        and _numba_real_scalar(x1)
        and getattr(fp, 'layout', None) == 'C'
        and _numba_none(out)
    ):

        def impl(
            x0,
            x1,
            xp0,
            xp1,
            fp,
            ilb=None,
            extrapolate=True,
            out=None,
        ):
            return _interp2d_scalar_c(x0, x1, xp0, xp1, fp, ilb, extrapolate)

        return impl
    return None


@numba_overload(interp2d, jit_options=JIT_OPTIONS)
def _overload_interp2d(
    x0: Any,
    x1: Any,
    xp0: Any,
    xp1: Any,
    fp: Any,
    ilb: Any = None,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    # C scalar signatures are claimed above; keep generic and array paths uninlined.
    if _numba_real_scalar(x0) and _numba_real_scalar(x1):
        if getattr(fp, 'layout', None) == 'C' or not _numba_none(out):
            return None

        def impl(
            x0,
            x1,
            xp0,
            xp1,
            fp,
            ilb=None,
            extrapolate=True,
            out=None,
        ):
            return interp2d_scalar(x0, x1, xp0, xp1, fp, ilb, extrapolate)

        return impl
    if _numba_real_array(x0) and _numba_real_array(x1):
        return interp2d_array
    return None


@numba_overload(interp3d_locate, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp3d_locate_point_inline(
    x0: Any,
    x1: Any,
    x2: Any,
    xp0: Any,
    xp1: Any,
    xp2: Any,
    ilb: Any = None,
    index_out: Any = None,
    weight_out: Any = None,
) -> Any:
    """Claim point locations for forced caller inlining.

    The point wrapper is small enough to inline profitably, and exposing it lets
    Numba scalarize its three index and weight components or caller-provided
    length-three buffers. Restricting this template to scalar coordinates prevents
    the allocating array wrapper and its loop from inheriting the same policy.
    """
    if _numba_real_scalar(x0) and _numba_real_scalar(x1) and _numba_real_scalar(x2):
        return interp3d_locate_point
    return None


@numba_overload(interp3d_locate, jit_options=JIT_OPTIONS)
def _overload_interp3d_locate(
    x0: Any,
    x1: Any,
    x2: Any,
    xp0: Any,
    xp1: Any,
    xp2: Any,
    ilb: Any = None,
    index_out: Any = None,
    weight_out: Any = None,
) -> Any:
    """Claim array locations without forcing the wrapper inline.

    The point template above owns scalar-coordinate signatures. Compiling
    ``interp3d_locate_array`` behind a normal call boundary keeps its optional
    allocation results concrete before a chained ``interp3d_eval(..., out=...)``
    is typed. This avoids Numba's internal keyword assertion and keeps the larger
    three-dimensional array loop out of its callers.
    """
    if _numba_real_scalar(x0) and _numba_real_scalar(x1) and _numba_real_scalar(x2):
        return None
    if _numba_real_array(x0) and _numba_real_array(x1) and _numba_real_array(x2):
        return interp3d_locate_array
    return None


@numba_overload(interp3d_eval, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp3d_eval_strided(
    index: Any,
    weight: Any,
    fp: Any,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    """Claim arbitrary-strided point evaluation for forced caller inlining.

    Inlining this layout-specific boundary lets Numba scalarize point-sized index
    and weight views. C-contiguous calls are deliberately left for the compact,
    normally compiled overload below.
    """
    from numba import types

    tuple_input = (
        _numba_integer_triplet(index)
        and _numba_real_triplet(weight)
        and _numba_real_array(fp)
        and fp.ndim == 3
    )
    array_input = isinstance(index, types.Array) and index.ndim == 1
    if (
        (tuple_input or array_input)
        and getattr(fp, 'layout', None) == 'A'
        and _numba_none(out)
    ):

        def impl(index, weight, fp, extrapolate=True, out=None):
            return _interp3d_eval_point(index, weight, fp, extrapolate)

        return impl
    return None


@numba_overload(interp3d_eval, jit_options=JIT_OPTIONS)
def _overload_interp3d_eval(
    index: Any,
    weight: Any,
    fp: Any,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    """Select compact point leaves or the out-of-line array evaluator.

    Keeping the public C-layout point boundary out of line avoids expanding
    repeated-field callers; only the flattened corner-load leaf is inlined there.
    Arbitrary-strided points were claimed by the preceding overload.
    """
    from numba import types

    tuple_input = (
        _numba_integer_triplet(index)
        and _numba_real_triplet(weight)
        and _numba_real_array(fp)
        and fp.ndim == 3
    )
    array_input = isinstance(index, types.Array) and index.ndim == 1
    if tuple_input or array_input:
        if getattr(fp, 'layout', None) == 'A' or not _numba_none(out):
            return None

        if getattr(fp, 'layout', None) == 'C':

            def impl(index, weight, fp, extrapolate=True, out=None):
                return _interp3d_eval_point_c(index, weight, fp, extrapolate)

        else:

            def impl(index, weight, fp, extrapolate=True, out=None):
                return _interp3d_eval_point(index, weight, fp, extrapolate)

        return impl
    if isinstance(index, types.Array) and index.ndim > 1:
        return interp3d_eval_array
    return None


@numba_overload(interp3d, jit_options=JIT_OPTIONS, inline='always')
def _overload_interp3d_point_c(
    x0: Any,
    x1: Any,
    x2: Any,
    xp0: Any,
    xp1: Any,
    xp2: Any,
    fp: Any,
    ilb: Any = None,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    """Claim fused C-contiguous point interpolation for forced inlining.

    Isolating this signature prevents the large array implementation and generic
    stride handling from being copied into compact point callers.
    """
    if (
        _numba_real_scalar(x0)
        and _numba_real_scalar(x1)
        and _numba_real_scalar(x2)
        and getattr(fp, 'layout', None) == 'C'
        and _numba_none(out)
    ):

        def impl(
            x0,
            x1,
            x2,
            xp0,
            xp1,
            xp2,
            fp,
            ilb=None,
            extrapolate=True,
            out=None,
        ):
            return _interp3d_point_c(x0, x1, x2, xp0, xp1, xp2, fp, ilb, extrapolate)

        return impl
    return None


@numba_overload(interp3d, jit_options=JIT_OPTIONS)
def _overload_interp3d(
    x0: Any,
    x1: Any,
    x2: Any,
    xp0: Any,
    xp1: Any,
    xp2: Any,
    fp: Any,
    ilb: Any = None,
    extrapolate: Any = True,
    out: Any = None,
) -> Any:
    """Handle generic-layout points and equal-shaped coordinate arrays.

    C-contiguous points are owned by the forced-inline overload above. Keeping this
    fallback normally compiled avoids duplicating generic addressing and array
    loops in every caller.
    """
    if _numba_real_scalar(x0) and _numba_real_scalar(x1) and _numba_real_scalar(x2):
        if getattr(fp, 'layout', None) == 'C' or not _numba_none(out):
            return None

        def impl(
            x0,
            x1,
            x2,
            xp0,
            xp1,
            xp2,
            fp,
            ilb=None,
            extrapolate=True,
            out=None,
        ):
            return interp3d_point(x0, x1, x2, xp0, xp1, xp2, fp, ilb, extrapolate)

        return impl
    if _numba_real_array(x0) and _numba_real_array(x1) and _numba_real_array(x2):
        return interp3d_array
    return None
