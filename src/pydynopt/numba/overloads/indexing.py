"""Provide Numba-compatible overloads for NumPy indexing functions.

- Convert between flat and multidimensional indices.
- Support the subset of NumPy indexing options required by Numba code.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy import ravel_multi_index, typing as npt, unravel_index

from pydynopt.numba import JIT_OPTIONS, overload

__all__ = ['ravel_multi_index', 'unravel_index']


def _unravel_index_array(
    indices: npt.ArrayLike,
    shape: Sequence[int],
    order: str = 'C',
) -> npt.NDArray[Any]:
    """Convert flat indices to an array of coordinate arrays.

    This is a Numba-compatible implementation of ``np.unravel_index`` for
    array-valued indices.

    Parameters
    ----------
    indices
        Flat indices to convert.
    shape
        Shape of the target array.
    order
        Index order, either ``'C'`` or ``'F'``.

    Returns
    -------
    Coordinate array whose leading axis indexes dimensions.
    """
    order = order.upper()

    lindices = np.atleast_1d(indices)
    lindices_flat = lindices.reshape((-1,))

    unravel_ndim = len(shape)
    unravel_dims = np.empty(unravel_ndim, dtype=np.int64)
    # Copy dimensions in a loop because constructing an array from an array
    # can fail in Numba mode.
    for i in range(unravel_ndim):
        unravel_dims[i] = shape[i]
    unravel_size = int(np.prod(unravel_dims))

    # Tuple concatenation is supported by Numba; starred unpacking is not.
    coords_shp = (unravel_ndim,) + tuple(lindices.shape)  # noqa: RUF005
    coords = np.empty(coords_shp, dtype=lindices.dtype)
    coords_flat = coords.reshape((unravel_ndim, -1))

    idx_start = unravel_ndim - 1 if order == 'C' else 0
    idx_step = -1 if order == 'C' else 1

    for i in range(lindices.size):
        val = lindices_flat[i]

        if val < 0 or val >= unravel_size:
            raise ValueError('Invalid flat index')

        idx = idx_start

        for _ in range(unravel_ndim):
            tmp = val / unravel_dims[idx]
            coords_flat[idx, i] = val % unravel_dims[idx]
            val = tmp
            idx += idx_step

    return coords


def _unravel_index_scalar(
    indices: int,
    shape: Sequence[int],
    order: str = 'C',
) -> npt.NDArray[Any]:
    """Convert a flat scalar index to a coordinate array.

    This Numba-compatible implementation dispatches through the array-valued
    overload of ``np.unravel_index``.

    Parameters
    ----------
    indices
        Flat index to convert.
    shape
        Shape of the target array.
    order
        Index order, either ``'C'`` or ``'F'``.

    Returns
    -------
    One-dimensional array containing the coordinate of each dimension.
    """
    indices1d = np.array([indices])

    coords2d: Any = np.unravel_index(indices1d, shape, order)  # type: ignore
    coords = coords2d[:, 0]

    return coords


@overload(unravel_index, jit_options=JIT_OPTIONS)
def unravel_index_generic(
    indices: Any,
    shape: Any,
    order: Any = 'C',
) -> Callable[..., npt.NDArray[Any]] | None:
    """Select an ``unravel_index`` implementation for Numba argument types.

    Parameters
    ----------
    indices
        Numba type describing the flat indices.
    shape
        Numba type describing the target shape.
    order
        Numba type describing the index order.

    Returns
    -------
    Matching overload implementation, or ``None`` for unsupported types.
    """
    from numba import types

    f = None
    if isinstance(indices, types.Integer):  # type: ignore
        f = _unravel_index_scalar
    elif isinstance(indices, types.Array):
        f = _unravel_index_array

    return f


def _ravel_multi_index_array(
    multi_index: npt.NDArray[Any],
    dims: Sequence[int],
    mode: str = 'raise',
    order: str = 'C',
) -> npt.NDArray[Any]:
    """Convert an array of coordinates to flat indices.

    This Numba-compatible partial implementation of ``np.ravel_multi_index``
    requires one coordinate dimension per row and supports only ``mode='raise'``.

    Parameters
    ----------
    multi_index
        Coordinate array whose leading axis indexes dimensions.
    dims
        Size of each dimension.
    mode
        Boundary mode. Only ``'raise'`` is supported.
    order
        Index order, which is accepted for API compatibility and ignored.

    Returns
    -------
    Array of flat indices.
    """
    ravel_dims = np.empty(len(dims), dtype=np.int64)
    for i, d in enumerate(dims):
        ravel_dims[i] = d

    ravel_ndim = ravel_dims.size

    dtype = multi_index.dtype
    # Flatten all remaining axes; the leading axis represents dimensions.
    lmulti_index_flat = multi_index.reshape((ravel_ndim, -1))

    one = np.ones(1, dtype=ravel_dims.dtype)
    iwork = np.hstack((one, ravel_dims[:0:-1]))
    ravel_strides = np.cumprod(iwork)[::-1]

    shp_indices = tuple(multi_index.shape[1:]) if multi_index.ndim >= 2 else (1,)

    indices = np.empty(shp_indices, dtype=dtype)
    indices_flat = indices.reshape((-1,))
    N = lmulti_index_flat.shape[-1]

    mode = mode.upper()
    if mode != 'RAISE':
        raise NotImplementedError("mode='raise' required")

    for k in range(N):
        raveled = 0

        for i in range(ravel_ndim):
            m = ravel_dims[i]
            j = lmulti_index_flat[i, k]

            if j < 0 or j >= m:
                raise ValueError('Invalid multi-index')

            raveled += j * ravel_strides[i]

        indices_flat[k] = raveled

    return indices


def _ravel_multi_index_array1d(
    multi_index: npt.NDArray[Any],
    dims: Sequence[int],
    mode: str = 'raise',
    order: str = 'C',
) -> Any:
    """Convert a one-dimensional coordinate array to a flat index.

    Parameters
    ----------
    multi_index
        One-dimensional coordinate array.
    dims
        Size of each dimension.
    mode
        Boundary mode. Only ``'raise'`` is supported.
    order
        Index order, accepted for API compatibility.

    Returns
    -------
    Flat index for the supplied coordinates.
    """
    lmulti_index = multi_index.reshape((-1, 1))
    indices = np.ravel_multi_index(  # type: ignore
        lmulti_index, dims, mode, order
    )

    index = indices[0]
    return index


def _ravel_multi_index(
    multi_index: Any,
    dims: Sequence[int],
    mode: str = 'raise',
    order: str = 'C',
) -> Any:
    """Convert a coordinate sequence to a flat index.

    Parameters
    ----------
    multi_index
        Coordinate sequence.
    dims
        Size of each dimension.
    mode
        Boundary mode. Only ``'raise'`` is supported.
    order
        Index order, accepted for API compatibility.

    Returns
    -------
    Flat index for the supplied coordinates.
    """
    lmulti_index = np.array(multi_index)

    # Dispatch to the one-dimensional array implementation above.
    index = np.ravel_multi_index(  # type: ignore
        lmulti_index, dims, mode, order
    )
    return index


@overload(ravel_multi_index, jit_options=JIT_OPTIONS)
def ravel_multi_index_generic(
    multi_index: Any,
    dims: Any,
    mode: Any = 'raise',
    order: Any = 'C',
) -> Callable[..., Any]:
    """Select a ``ravel_multi_index`` implementation for Numba types.

    Parameters
    ----------
    multi_index
        Numba type describing the coordinates.
    dims
        Numba type describing dimension sizes.
    mode
        Numba type describing the boundary mode.
    order
        Numba type describing the index order.

    Returns
    -------
    Matching overload implementation.
    """
    from numba import types

    if isinstance(multi_index, types.Array) and multi_index.ndim >= 2:
        f = _ravel_multi_index_array
    elif isinstance(multi_index, types.Array) and multi_index.ndim == 1:
        f = _ravel_multi_index_array1d
    else:
        f = _ravel_multi_index

    return f
