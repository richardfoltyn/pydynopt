"""Provide Numba-compatible kernels for C-order index conversion.

The allocation wrappers create int64 outputs. In-place kernels require writable,
conformable integer output arrays. Shape, axis, index, and coordinate bounds are
checked because public Numba overloads call these kernels directly.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence

import numpy as np

from pydynopt.numba import JIT_OPTIONS_INLINE, register_jitable

__all__ = [
    'ind2sub_array',
    'ind2sub_array_impl',
    'ind2sub_axis_array',
    'ind2sub_axis_array_impl',
    'ind2sub_axis_scalar',
    'ind2sub_scalar',
    'ind2sub_scalar_impl',
    'sub2ind_array',
    'sub2ind_array_impl',
    'sub2ind_scalar',
]


@register_jitable(**JIT_OPTIONS_INLINE)
def _shape_size(shape: Sequence[int]) -> int:
    """Validate a shape and return its total size."""
    ndim = len(shape)
    if ndim == 0:
        msg = 'shape must contain at least one dimension'
        raise ValueError(msg)

    size = 1
    for j in range(ndim):
        dim = shape[j]
        if dim <= 0:
            msg = 'shape dimensions must be strictly positive'
            raise ValueError(msg)
        size *= dim
    return size


@register_jitable(**JIT_OPTIONS_INLINE)
def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize an axis and reject values outside the shape dimensions."""
    normalized = axis
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        msg = 'axis is outside the shape dimensions'
        raise ValueError(msg)
    return normalized


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_scalar_impl(
    indices: int,
    shape: Sequence[int],
    out: np.ndarray,
) -> None:
    """Unravel one flat index into a required output buffer.

    Parameters
    ----------
    indices
        Flat index satisfying ``0 <= indices < prod(shape)``.
    shape
        Non-empty sequence of positive dimensions.
    out
        Writable integer buffer with shape ``(len(shape),)``.

    Raises
    ------
    ValueError
        If ``shape`` or ``indices`` is invalid.
    """
    size = _shape_size(shape)
    if indices < 0 or indices >= size:
        msg = 'flat index is outside the valid range'
        raise ValueError(msg)

    val = indices
    for j in range(len(shape) - 1, -1, -1):
        dim = shape[j]
        out[j] = val % dim
        val //= dim


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_scalar(
    indices: int,
    shape: Sequence[int],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Unravel one flat index and optionally allocate its output.

    Parameters
    ----------
    indices
        Flat index satisfying ``0 <= indices < prod(shape)``.
    shape
        Non-empty sequence of positive dimensions.
    out
        Optional writable integer buffer with shape ``(len(shape),)``.

    Returns
    -------
    A newly allocated ``int64`` array or the supplied output buffer by identity.

    Raises
    ------
    ValueError
        If ``shape``, ``indices``, or the output shape is invalid.
    """
    expected = (len(shape),)
    result = np.empty(expected, dtype=np.int64) if out is None else out
    if result.shape != expected:
        msg = 'out has an invalid shape'
        raise ValueError(msg)
    ind2sub_scalar_impl(indices, shape, result)
    return result


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_axis_scalar(
    indices: int,
    shape: Sequence[int],
    axis: int,
) -> int:
    """Unravel one flat index along one axis.

    Parameters
    ----------
    indices
        Flat index satisfying ``0 <= indices < prod(shape)``.
    shape
        Non-empty sequence of positive dimensions.
    axis
        Coordinate axis. Negative values follow NumPy's normalization convention.

    Returns
    -------
    The coordinate along ``axis``.

    Raises
    ------
    ValueError
        If ``shape``, ``indices``, or ``axis`` is invalid.
    """
    size = _shape_size(shape)
    if indices < 0 or indices >= size:
        msg = 'flat index is outside the valid range'
        raise ValueError(msg)

    laxis = _normalize_axis(axis, len(shape))
    stride = 1
    for j in range(len(shape) - 1, laxis, -1):
        stride *= shape[j]
    return int((indices // stride) % shape[laxis])


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_array_impl(
    indices: np.ndarray,
    shape: Sequence[int],
    out: np.ndarray,
) -> None:
    """Unravel flat indices into a required dimension-first output buffer.

    Parameters
    ----------
    indices
        Flat indices satisfying ``0 <= indices < prod(shape)``.
    shape
        Non-empty sequence of positive dimensions.
    out
        Writable integer buffer with shape ``(len(shape), *indices.shape)``. Every
        element is overwritten.

    Raises
    ------
    ValueError
        If ``shape`` or an index is invalid.
    """
    size = _shape_size(shape)
    ndim = len(shape)
    n = indices.size

    for i in range(n):
        val = indices.flat[i]
        if val < 0 or val >= size:
            msg = 'flat index is outside the valid range'
            raise ValueError(msg)
        for j in range(ndim - 1, -1, -1):
            dim = shape[j]
            out.flat[j * n + i] = val % dim
            val //= dim


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_array(
    indices: np.ndarray,
    shape: Sequence[int],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Unravel flat indices and optionally allocate their output.

    Parameters
    ----------
    indices
        Flat indices satisfying ``0 <= indices < prod(shape)``.
    shape
        Non-empty sequence of positive dimensions.
    out
        Optional writable integer buffer with shape
        ``(len(shape), *indices.shape)``.

    Returns
    -------
    A newly allocated ``int64`` array or the supplied output buffer by identity.

    Raises
    ------
    ValueError
        If ``shape``, an index, or the output shape is invalid.
    """
    expected = (len(shape), *indices.shape)
    result = np.empty(expected, dtype=np.int64) if out is None else out
    if result.shape != expected:
        msg = 'out has an invalid shape'
        raise ValueError(msg)
    ind2sub_array_impl(indices, shape, result)
    return result


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_axis_array_impl(
    indices: np.ndarray,
    shape: Sequence[int],
    axis: int,
    out: np.ndarray,
) -> None:
    """Unravel flat indices along one axis into a required output buffer.

    Parameters
    ----------
    indices
        Flat indices satisfying ``0 <= indices < prod(shape)``.
    shape
        Non-empty sequence of positive dimensions.
    axis
        Coordinate axis. Negative values follow NumPy's normalization convention.
    out
        Writable integer buffer with the same shape as ``indices``. Every element
        is overwritten.

    Raises
    ------
    ValueError
        If ``shape``, an index, or ``axis`` is invalid.
    """
    size = _shape_size(shape)
    laxis = _normalize_axis(axis, len(shape))
    stride = 1
    for j in range(len(shape) - 1, laxis, -1):
        stride *= shape[j]

    for i in range(indices.size):
        val = indices.flat[i]
        if val < 0 or val >= size:
            msg = 'flat index is outside the valid range'
            raise ValueError(msg)
        out.flat[i] = (val // stride) % shape[laxis]


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_axis_array(
    indices: np.ndarray,
    shape: Sequence[int],
    axis: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Unravel flat indices along one axis and optionally allocate output.

    Parameters
    ----------
    indices
        Flat indices satisfying ``0 <= indices < prod(shape)``.
    shape
        Non-empty sequence of positive dimensions.
    axis
        Coordinate axis. Negative values follow NumPy's normalization convention.
    out
        Optional writable integer buffer with the same shape as ``indices``.

    Returns
    -------
    A newly allocated ``int64`` array or the supplied output buffer by identity.

    Raises
    ------
    ValueError
        If ``shape``, an index, ``axis``, or the output shape is invalid.
    """
    result = np.empty(indices.shape, dtype=np.int64) if out is None else out
    if result.shape != indices.shape:
        msg = 'out has an invalid shape'
        raise ValueError(msg)
    ind2sub_axis_array_impl(indices, shape, axis, result)
    return result


@register_jitable(**JIT_OPTIONS_INLINE)
def sub2ind_scalar(
    coords: Sequence[int] | np.ndarray,
    shape: Sequence[int],
) -> int:
    """Ravel one coordinate sequence into a C-order flat index.

    Parameters
    ----------
    coords
        One coordinate per dimension, each within the corresponding bound.
    shape
        Non-empty sequence of positive dimensions.

    Returns
    -------
    The C-order flat index.

    Raises
    ------
    ValueError
        If ``shape``, the coordinate count, or a coordinate is invalid.
    """
    _shape_size(shape)
    ndim = len(shape)
    if len(coords) != ndim:
        msg = 'coordinate count must equal the number of dimensions'
        raise ValueError(msg)

    index = 0
    for j in range(ndim):
        coord = coords[j]
        if coord < 0 or coord >= shape[j]:
            msg = 'coordinate is outside the valid range'
            raise ValueError(msg)
        index = index * shape[j] + coord
    return int(index)


@register_jitable(**JIT_OPTIONS_INLINE)
def sub2ind_array_impl(
    coords: np.ndarray,
    shape: Sequence[int],
    out: np.ndarray,
) -> None:
    """Ravel dimension-first coordinate batches into a required output buffer.

    Parameters
    ----------
    coords
        Coordinates with shape ``(len(shape), *S)`` and valid bounds.
    shape
        Non-empty sequence of positive dimensions.
    out
        Writable integer buffer with sample shape ``S``. Every element is
        overwritten.

    Raises
    ------
    ValueError
        If ``shape``, the leading coordinate dimension, or a coordinate is invalid.
    """
    _shape_size(shape)
    ndim = len(shape)
    if coords.ndim < 2 or coords.shape[0] != ndim:
        msg = 'leading coordinate dimension must equal the number of dimensions'
        raise ValueError(msg)

    n = coords.size // ndim
    for i in range(n):
        index = 0
        for j in range(ndim):
            coord = coords.flat[j * n + i]
            if coord < 0 or coord >= shape[j]:
                msg = 'coordinate is outside the valid range'
                raise ValueError(msg)
            index = index * shape[j] + coord
        out.flat[i] = index


@register_jitable(**JIT_OPTIONS_INLINE)
def sub2ind_array(
    coords: np.ndarray,
    shape: Sequence[int],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Ravel coordinate batches and optionally allocate their output.

    Parameters
    ----------
    coords
        Coordinates with shape ``(len(shape), *S)`` and valid bounds.
    shape
        Non-empty sequence of positive dimensions.
    out
        Optional writable integer buffer with sample shape ``S``.

    Returns
    -------
    A newly allocated ``int64`` array or the supplied output buffer by identity.

    Raises
    ------
    ValueError
        If ``coords``, ``shape``, a coordinate, or the output shape is invalid.
    """
    if coords.ndim < 2:
        msg = 'batched coordinates must have at least two dimensions'
        raise ValueError(msg)
    expected = coords.shape[1:]
    result = np.empty(expected, dtype=np.int64) if out is None else out
    if result.shape != expected:
        msg = 'out has an invalid shape'
        raise ValueError(msg)
    sub2ind_array_impl(coords, shape, result)
    return result
