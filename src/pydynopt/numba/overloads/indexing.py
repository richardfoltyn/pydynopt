"""
Overloads for Numpy indexing functions for Numba code.
"""


import numpy as np
from numpy import ravel_multi_index, unravel_index

from pydynopt.numba import JIT_OPTIONS, overload

__all__ = ['unravel_index', 'ravel_multi_index']


def _unravel_index_array(indices, shape, order='C'):
    """
    Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Numba-fiable implementation of np.unravel_index() for array-valued arguments

    Parameters
    ----------
    indices
    shape
    order

    Returns
    -------

    """

    order = order.upper()

    lindices = np.atleast_1d(indices)
    lindices_flat = lindices.reshape((-1,))

    unravel_ndim = len(shape)
    unravel_dims = np.empty(unravel_ndim, dtype=np.int64)
    # Copy over dimensions in a loop, since creating an array with np.array()
    # with an argument that already is an array seems to fail in Numba mode.
    for i in range(unravel_ndim):
        unravel_dims[i] = shape[i]
    unravel_size = int(np.prod(unravel_dims))

    coords_shp = (unravel_ndim,) + tuple(lindices.shape)
    coords = np.empty(coords_shp, dtype=lindices.dtype)
    coords_flat = coords.reshape((unravel_ndim, -1))

    idx_start = unravel_ndim - 1 if order == 'C' else 0
    idx_step = -1 if order == 'C' else 1

    for i in range(lindices.size):
        val = lindices_flat[i]

        if val < 0 or val >= unravel_size:
            raise ValueError('Invalid flat index')

        idx = idx_start

        for j in range(0, unravel_ndim):
            tmp = val / unravel_dims[idx]
            coords_flat[idx, i] = val % unravel_dims[idx]
            val = tmp
            idx += idx_step

    return coords


def _unravel_index_scalar(indices, shape, order='C'):
    """
    Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Numba-fiable implementation of np.unravel_index() for scalar arguments

    Parameters
    ----------
    indices
    shape
    order

    Returns
    -------

    """

    indices1d = np.array([indices])

    coords2d = np.unravel_index(indices1d, shape, order)
    coords = coords2d[:, 0]

    return coords


@overload(unravel_index, jit_options=JIT_OPTIONS)
def unravel_index_generic(indices, shape, order='C'):

    from numba import types

    f = None
    if isinstance(indices, types.Integer):
        f = _unravel_index_scalar
    elif isinstance(indices, types.Array):
        f = _unravel_index_array

    return f


def _ravel_multi_index_array(multi_index, dims, mode='raise', order='C'):
    """
    Converts a tuple of index arrays into an array of flat indices.

    Numba-fiable partial implementation of np.ravel_multi_index() for array-valued arguments.

    Parameters
    ----------
    multi_index : array_like
        Array of indices where each row represents a dimension.
        Unlike the NumPy implementation, a tuple of arrays is not supported.
    dims : array_like of int
    mode : str
        Required to be 'raise' but otherwise ignored
    order : str
        Ignored, only present to ensure compatible function signatures.

    Returns
    -------

    """

    # Convert dims which might be any sequence to a numpy array
    ravel_dims = np.empty(len(dims), dtype=np.int64)
    for i, d in enumerate(dims):
        ravel_dims[i] = d

    ravel_ndim = ravel_dims.size

    dtype = multi_index.dtype
    # Flatten out all remaining axes. Leading axis represents dimensions to be raveled.
    lmulti_index_flat = multi_index.reshape((ravel_ndim, -1))

    one = np.ones(1, dtype=ravel_dims.dtype)
    iwork = np.hstack((one, ravel_dims[:0:-1]))
    ravel_strides = np.cumprod(iwork)[::-1]

    if multi_index.ndim >= 2:
        shp_indices = tuple(multi_index.shape[1:])
    else:
        shp_indices = (1,)

    indices = np.empty(shp_indices, dtype=dtype)
    indices_flat = indices.reshape((-1,))
    N = lmulti_index_flat.shape[-1]

    mode = mode.upper()

    if mode != 'RAISE':
        raise NotImplementedError("mode='raise' required")

    MODE_RAISE = 0
    MODE_WRAP = 1
    MODE_CLIP = 2

    imode = MODE_RAISE

    for k in range(N):
        raveled = 0

        for i in range(ravel_ndim):
            m = ravel_dims[i]
            j = lmulti_index_flat[i, k]

            if imode == MODE_RAISE:
                if j < 0 or j >= m:
                    raise ValueError('Invalid multi-index')

            raveled += j * ravel_strides[i]

        indices_flat[k] = raveled

    return indices


def _ravel_multi_index_array1d(multi_index, dims, mode='raise', order='C'):

    lmulti_index = multi_index.reshape((-1, 1))
    indices = np.ravel_multi_index(lmulti_index, dims, mode, order)

    index = indices[0]

    return index


def _ravel_multi_index(multi_index, dims, mode='raise', order='C'):

    lmulti_index = np.array(multi_index)

    # This should call the _array1d implementation above
    index = np.ravel_multi_index(lmulti_index, dims, mode, order)

    return index


@overload(ravel_multi_index, jit_options=JIT_OPTIONS)
def ravel_multi_index_generic(multi_index, dims, mode='raise', order='C'):

    from numba import types

    f = None
    if isinstance(multi_index, types.Array) and multi_index.ndim >= 2:
        f = _ravel_multi_index_array
    elif isinstance(multi_index, types.Array) and multi_index.ndim == 1:
        f = _ravel_multi_index_array1d
    else:
        f = _ravel_multi_index

    return f
