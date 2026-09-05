"""Provide checked one- and two-dimensional linear interpolation.

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
    interp1d_array,
    interp1d_array_impl,
    interp1d_eval_array,
    interp1d_eval_array_impl,
    interp1d_eval_scalar,
    interp1d_locate_array,
    interp1d_locate_array_impl,
    interp1d_locate_scalar,
    interp1d_scalar,
    interp2d_array,
    interp2d_array_impl,
    interp2d_eval_array,
    interp2d_eval_array_impl,
    interp2d_eval_scalar,
    interp2d_locate_array,
    interp2d_locate_array_impl,
    interp2d_locate_scalar,
    interp2d_locate_scalar_impl,
    interp2d_scalar,
)

__all__ = [
    'interp1d',
    'interp1d_eval',
    'interp1d_locate',
    'interp2d',
    'interp2d_eval',
    'interp2d_locate',
]

type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type IntegerScalar = int | np.integer[Any]
type ArrayQuery = Sequence[RealScalar] | np.ndarray
type FloatArray = NDArray[np.float64]
type IndexArray = NDArray[np.int64]
type InitialIndex2D = Sequence[IntegerScalar] | np.ndarray | None

_interp1d_locate_scalar_jit = jit(interp1d_locate_scalar, **JIT_OPTIONS)
_interp1d_locate_array_jit = jit(interp1d_locate_array_impl, **JIT_OPTIONS)
_interp1d_eval_scalar_jit = jit(interp1d_eval_scalar, **JIT_OPTIONS)
_interp1d_eval_array_jit = jit(interp1d_eval_array_impl, **JIT_OPTIONS)
_interp1d_scalar_jit = jit(interp1d_scalar, **JIT_OPTIONS)
_interp1d_array_jit = jit(interp1d_array_impl, **JIT_OPTIONS)
_interp2d_locate_scalar_jit = jit(interp2d_locate_scalar_impl, **JIT_OPTIONS)
_interp2d_locate_array_jit = jit(interp2d_locate_array_impl, **JIT_OPTIONS)
_interp2d_eval_scalar_jit = jit(interp2d_eval_scalar, **JIT_OPTIONS)
_interp2d_eval_array_jit = jit(interp2d_eval_array_impl, **JIT_OPTIONS)
_interp2d_scalar_jit = jit(interp2d_scalar, **JIT_OPTIONS)
_interp2d_array_jit = jit(interp2d_array_impl, **JIT_OPTIONS)


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
        value = _interp1d_eval_scalar_jit(
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


def interp2d_eval(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Evaluate a bilinear interpolant from indices and weights.

    Parameters
    ----------
    index
        Integer NumPy array with shape ``sample_shape + (2,)`` containing valid
        lower-bound indices for each axis.
    weight
        Real NumPy array with the same shape as ``index`` containing lower-grid-point
        weights for each axis.
    fp
        Two-dimensional NumPy array with at least two function values on each axis.
    extrapolate
        Whether to evaluate the bilinear extrapolant outside either grid. If false,
        an exterior point receives ``NaN``.
    out
        Optional writable float64 buffer with ``sample_shape``. Not supported when
        ``index`` has shape ``(2,)`` and therefore describes one point.

    Returns
    -------
    A Python float for one point, a newly allocated float64 array for multiple
    points, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If an input or output buffer has an unsupported type or dtype, or ``out`` is
        supplied for one point.
    ValueError
        If input or output shapes are invalid, ``fp`` is not conformable, or ``out``
        is not writable.
    IndexError
        If any lower-bound index is outside the corresponding valid range.
    """
    if not isinstance(index, np.ndarray) or not isinstance(weight, np.ndarray):
        msg = 'index and weight must be NumPy arrays'
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

    _validate_real_array(fp, 'fp', ndim=2)
    if fp.shape[0] < 2 or fp.shape[1] < 2:
        msg = 'fp must have at least two values on each axis'
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
        value = _interp2d_eval_scalar_jit(index_work, weight_work, fp, extrapolate)
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


@numba_overload(interp1d_locate, jit_options=JIT_OPTIONS)
def _overload_interp1d_locate(
    x: Any,
    xp: Any,
    ilb: Any = 0,
    index_out: Any = None,
    weight_out: Any = None,
) -> Any:
    if _numba_real_scalar(x):
        if not _numba_none(index_out) or not _numba_none(weight_out):
            return None

        def impl(x, xp, ilb=0, index_out=None, weight_out=None):
            return interp1d_locate_scalar(x, xp, ilb)

        return impl
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
            return interp1d_eval_scalar(
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
    if _numba_real_scalar(x):
        if not _numba_none(out):
            return None

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
    if _numba_real_array(x):
        return interp1d_array
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
    if _numba_real_scalar(x0) and _numba_real_scalar(x1):
        return interp2d_locate_scalar
    if _numba_real_array(x0) and _numba_real_array(x1):
        return interp2d_locate_array
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

    if isinstance(index, types.Array) and index.ndim == 1:
        if not _numba_none(out):
            return None

        def impl(index, weight, fp, extrapolate=True, out=None):
            return interp2d_eval_scalar(index, weight, fp, extrapolate)

        return impl
    if isinstance(index, types.Array) and index.ndim > 1:
        return interp2d_eval_array
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
    if _numba_real_scalar(x0) and _numba_real_scalar(x1):
        if not _numba_none(out):
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
