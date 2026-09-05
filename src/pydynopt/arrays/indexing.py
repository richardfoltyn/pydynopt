"""Provide checked C-order conversion between flat and multidimensional indices.

- Normalize Python scalar, sequence, array, shape, axis, and output arguments.
- Dispatch ordinary Python and Numba-compiled calls to shared low-level kernels.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence
from math import prod
from typing import Any, overload as typing_overload

import numpy as np
from numpy.typing import NDArray

from pydynopt.numba import JIT_OPTIONS, jit, overload as numba_overload

from .numba.indexing import (
    ind2sub_array,
    ind2sub_array_impl,
    ind2sub_axis_array,
    ind2sub_axis_array_impl,
    ind2sub_axis_scalar,
    ind2sub_scalar,
    ind2sub_scalar_impl,
    sub2ind_array,
    sub2ind_array_impl,
    sub2ind_scalar,
)

__all__ = [
    'ind2sub',
    'sub2ind',
]

type IntegerScalar = int | np.integer[Any]
type IndexArray = NDArray[np.int64]
type Shape = Sequence[IntegerScalar]
type IndexInput = Sequence[IntegerScalar] | np.ndarray
type NestedIndexInput = Sequence[Sequence[IntegerScalar]]

_ind2sub_scalar_impl_jit = jit(ind2sub_scalar_impl, **JIT_OPTIONS)
_ind2sub_axis_scalar_jit = jit(ind2sub_axis_scalar, **JIT_OPTIONS)
_ind2sub_array_impl_jit = jit(ind2sub_array_impl, **JIT_OPTIONS)
_ind2sub_axis_array_impl_jit = jit(ind2sub_axis_array_impl, **JIT_OPTIONS)
_sub2ind_scalar_jit = jit(sub2ind_scalar, **JIT_OPTIONS)
_sub2ind_array_impl_jit = jit(sub2ind_array_impl, **JIT_OPTIONS)


def _normalize_integer(value: object, name: str) -> int:
    """Normalize a non-boolean integer scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'iu':
        msg = f'{name} must be an integer'
        raise TypeError(msg)
    return int(array.item())


def _normalize_shape(shape: Shape) -> tuple[int, ...]:
    """Normalize a non-empty one-dimensional positive integer shape."""
    try:
        array = np.asarray(shape, dtype=object)
    except (TypeError, ValueError) as exc:
        msg = 'shape must be a one-dimensional integer sequence'
        raise TypeError(msg) from exc
    if array.ndim != 1:
        msg = 'shape must be one-dimensional'
        raise ValueError(msg)
    if array.size == 0:
        msg = 'shape must contain at least one dimension'
        raise ValueError(msg)

    normalized = tuple(_normalize_integer(item, 'shape entry') for item in array)
    if any(dim <= 0 for dim in normalized):
        msg = 'shape dimensions must be strictly positive'
        raise ValueError(msg)
    return normalized


def _normalize_axis(axis: IntegerScalar | None, ndim: int) -> int | None:
    """Normalize an optional axis using NumPy's negative-axis convention."""
    if axis is None:
        return None
    normalized = _normalize_integer(axis, 'axis')
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        msg = f'axis {axis} is outside [-{ndim}, {ndim})'
        raise ValueError(msg)
    return normalized


def _normalize_integer_array(value: object, name: str) -> np.ndarray:
    """Normalize an integer sequence or array to a contiguous NumPy array."""
    try:
        if not isinstance(value, np.ndarray):
            probe = np.asarray(value, dtype=object)
            for item in probe.flat:
                _normalize_integer(item, name)
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must contain integers'
        raise TypeError(msg) from exc
    if array.dtype.kind not in 'iu':
        msg = f'{name} must contain integers'
        raise TypeError(msg)
    if array.ndim == 0:
        return array
    return np.ascontiguousarray(array)


def _prepare_index_output(
    out: np.ndarray | None,
    shape: tuple[int, ...],
) -> IndexArray:
    """Allocate or validate a writable output capable of holding ``int64``."""
    if out is None:
        return np.empty(shape, dtype=np.int64)
    if not isinstance(out, np.ndarray):
        msg = 'out must be a NumPy array'
        raise TypeError(msg)
    if out.shape != shape:
        msg = f'out must have shape {shape}, got {out.shape}'
        raise ValueError(msg)
    if out.dtype.kind not in 'iu':
        msg = 'out must have an integer dtype'
        raise TypeError(msg)
    if not np.can_cast(np.dtype(np.int64), out.dtype, casting='safe'):
        msg = f'out dtype {out.dtype} cannot safely represent int64'
        raise TypeError(msg)
    if not out.flags.writeable:
        msg = 'out must be writable'
        raise ValueError(msg)
    return out


@typing_overload
def ind2sub(
    indices: IntegerScalar,
    shape: Shape,
    axis: None = None,
    out: IndexArray | None = None,
) -> IndexArray: ...


@typing_overload
def ind2sub(
    indices: IntegerScalar,
    shape: Shape,
    axis: IntegerScalar,
    out: None = None,
) -> int: ...


@typing_overload
def ind2sub(
    indices: IndexInput,
    shape: Shape,
    axis: IntegerScalar | None = None,
    out: IndexArray | None = None,
) -> IndexArray: ...


def ind2sub(
    indices: IntegerScalar | IndexInput,
    shape: Shape,
    axis: IntegerScalar | None = None,
    out: IndexArray | None = None,
) -> int | IndexArray:
    """Convert C-order flat indices into dimension-first coordinates.

    Parameters
    ----------
    indices
        Flat index or indices to convert. For an input with shape ``S``, full
        coordinates have shape ``(len(shape), *S)``.
    shape
        Non-empty one-dimensional sequence of positive dimensions.
    axis
        Optional coordinate axis. Negative axes follow NumPy's normalization
        convention.
    out
        Optional output buffer of the exact result shape. It must be writable and
        safely represent ``int64`` values. A scalar index with an explicit axis
        does not support an output buffer.

    Returns
    -------
    A built-in ``int`` for a scalar index with an explicit axis, a newly allocated
    ``int64`` array otherwise, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If an argument has an invalid integer dtype or a scalar-valued call
        receives ``out``.
    ValueError
        If shape, axis, index bounds, or output shape and writability are invalid.
    """
    dimensions = _normalize_shape(shape)
    laxis = _normalize_axis(axis, len(dimensions))
    size = prod(dimensions)
    scalar = np.isscalar(indices) and not isinstance(indices, np.ndarray)

    if scalar:
        index = _normalize_integer(indices, 'indices')
        if index < 0 or index >= size:
            msg = f'indices must satisfy 0 <= indices < {size}'
            raise ValueError(msg)
        if laxis is not None:
            if out is not None:
                msg = 'scalar axis-selected ind2sub calls do not accept out'
                raise TypeError(msg)
            coord = _ind2sub_axis_scalar_jit(index, dimensions, laxis)
            return int(coord)

        result = _prepare_index_output(out, (len(dimensions),))
        _ind2sub_scalar_impl_jit(index, dimensions, result)
        return result

    index_array = _normalize_integer_array(indices, 'indices')
    if np.any(index_array < 0) or np.any(index_array >= size):
        msg = f'indices must satisfy 0 <= indices < {size}'
        raise ValueError(msg)

    if laxis is None:
        result_shape = (len(dimensions), *index_array.shape)
        result = _prepare_index_output(out, result_shape)
        _ind2sub_array_impl_jit(index_array, dimensions, result)
    else:
        result = _prepare_index_output(out, index_array.shape)
        _ind2sub_axis_array_impl_jit(index_array, dimensions, laxis, result)
    return result


@typing_overload
def sub2ind(
    coords: Sequence[IntegerScalar],
    shape: Shape,
    out: None = None,
) -> int: ...


@typing_overload
def sub2ind(
    coords: NestedIndexInput | np.ndarray,
    shape: Shape,
    out: IndexArray | None = None,
) -> int | IndexArray: ...


def sub2ind(
    coords: Sequence[IntegerScalar] | NestedIndexInput | np.ndarray,
    shape: Shape,
    out: IndexArray | None = None,
) -> int | IndexArray:
    """Convert dimension-first coordinates into C-order flat indices.

    Parameters
    ----------
    coords
        Coordinates to convert. A one-dimensional input of length ``len(shape)``
        describes one point. An input with shape ``(len(shape), *S)`` describes a
        batch with sample shape ``S``.
    shape
        Non-empty one-dimensional sequence of positive dimensions.
    out
        Optional output buffer with the batch sample shape. It must be writable
        and safely represent ``int64`` values. Single-point calls do not support
        an output buffer.

    Returns
    -------
    A built-in ``int`` for one point, a newly allocated ``int64`` array for a
    batch, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If an argument has an invalid integer dtype or a single-point call
        receives ``out``.
    ValueError
        If dimensions, coordinate bounds, or output shape and writability are
        invalid.
    """
    dimensions = _normalize_shape(shape)
    coord_array = _normalize_integer_array(coords, 'coords')
    if coord_array.ndim == 0:
        msg = 'coords must have at least one dimension'
        raise ValueError(msg)
    if coord_array.shape[0] != len(dimensions):
        msg = 'leading coordinate dimension must equal len(shape)'
        raise ValueError(msg)

    for j, dim in enumerate(dimensions):
        if np.any(coord_array[j] < 0) or np.any(coord_array[j] >= dim):
            msg = f'coordinates for axis {j} must satisfy 0 <= coord < {dim}'
            raise ValueError(msg)

    if coord_array.ndim == 1:
        if out is not None:
            msg = 'single-point sub2ind calls do not accept out'
            raise TypeError(msg)
        index = _sub2ind_scalar_jit(coord_array, dimensions)
        return int(index)

    result = _prepare_index_output(out, coord_array.shape[1:])
    _sub2ind_array_impl_jit(coord_array, dimensions, result)
    return result


def _numba_none(value: Any) -> bool:
    """Return whether a Numba overload argument represents ``None``."""
    from numba import types

    return value is None or isinstance(value, (types.NoneType, types.Omitted))


def _numba_integer_scalar(value: Any) -> bool:
    """Return whether a Numba type is an integer scalar."""
    from numba import types

    return types.unliteral(value) in types.integer_domain


def _numba_integer_array(value: Any) -> bool:
    """Return whether a Numba type is an integer array."""
    from numba import types

    return isinstance(value, types.Array) and value.dtype in types.integer_domain


def _numba_integer_sequence(value: Any, allow_empty: bool = False) -> bool:
    """Return whether a Numba tuple or list contains only integers."""
    from numba import types

    if isinstance(value, types.Array):
        return False
    items = getattr(value, 'types', None)
    if items is not None:
        return (allow_empty or len(items) > 0) and all(
            _numba_integer_scalar(item) for item in items
        )
    item = getattr(value, 'item_type', getattr(value, 'dtype', None))
    return item is not None and _numba_integer_scalar(item)


def _numba_shape(value: Any) -> bool:
    """Return whether a Numba type can represent a one-dimensional shape."""
    from numba import types

    if isinstance(value, types.Array):
        return value.ndim == 1 and value.dtype in types.integer_domain
    return _numba_integer_sequence(value)


def _numba_index_output(value: Any, ndim: int) -> bool:
    """Return whether a Numba output is omitted or writable ``int64``."""
    from numba import types

    if _numba_none(value):
        return True
    return (
        isinstance(value, types.Array)
        and value.dtype == types.int64
        and value.ndim == ndim
        and value.mutable
    )


@numba_overload(ind2sub, jit_options=JIT_OPTIONS)
def _overload_ind2sub(
    indices: Any,
    shape: Any,
    axis: Any = None,
    out: Any = None,
) -> Any:
    """Select an index-to-coordinate implementation for Numba argument types."""
    if not _numba_shape(shape):
        return None

    no_axis = _numba_none(axis)
    if not no_axis and not _numba_integer_scalar(axis):
        return None

    if _numba_integer_scalar(indices):
        if no_axis:
            if not _numba_index_output(out, 1):
                return None

            def impl(indices, shape, axis=None, out=None):
                return ind2sub_scalar(indices, shape, out)

            return impl
        if not _numba_none(out):
            return None

        def impl(indices, shape, axis=None, out=None):
            laxis = 0 if axis is None else axis
            return ind2sub_axis_scalar(indices, shape, laxis)

        return impl

    if not _numba_integer_array(indices):
        return None
    result_ndim = indices.ndim if not no_axis else indices.ndim + 1
    if not _numba_index_output(out, result_ndim):
        return None
    if no_axis:

        def impl(indices, shape, axis=None, out=None):
            return ind2sub_array(indices, shape, out)

        return impl

    def impl(indices, shape, axis=None, out=None):
        laxis = 0 if axis is None else axis
        return ind2sub_axis_array(indices, shape, laxis, out)

    return impl


@numba_overload(sub2ind, jit_options=JIT_OPTIONS)
def _overload_sub2ind(
    coords: Any,
    shape: Any,
    out: Any = None,
) -> Any:
    """Select a coordinate-to-index implementation for Numba argument types."""
    if not _numba_shape(shape):
        return None

    if _numba_integer_sequence(coords):
        if not _numba_none(out):
            return None

        def impl(coords, shape, out=None):
            return sub2ind_scalar(coords, shape)

        return impl

    if not _numba_integer_array(coords) or coords.ndim == 0:
        return None
    if coords.ndim == 1:
        if not _numba_none(out):
            return None

        def impl(coords, shape, out=None):
            return sub2ind_scalar(coords, shape)

        return impl

    if not _numba_index_output(out, coords.ndim - 1):
        return None
    return sub2ind_array
