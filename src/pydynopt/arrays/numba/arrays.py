"""
Module implementing basic array creation and manipulation routines that
can be compiled by Numba.

Author: Richard Foltyn
"""


import numpy as np

from pydynopt.numba import register_jitable


def _insert(arr, obj, values, axis=None):
    """
    Insert values before the given indices.

    Implements mostly Numpy-compatible replacement for np.insert() that can
    be compiled using Numba.

    Notes
    -----
    -   This implementation ignores the axis argument.
    -   Only integer-valued index arrays are supported.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    obj : np.ndarray or int
        Object that defines the index or indices before which values is
        inserted. Must be integer-valued!
    values : np.ndarray or numeric
        Values to insert into `arr`.
    axis : object
        Ignored.

    Returns
    -------
    out : np.ndarray
        A copy of arr with values inserted.
    """

    lobj = np.asarray(obj)
    lvalues = np.asarray(values)

    if lobj.ndim > 1:
        raise ValueError('Unsupported array dimension')

    if (lobj.ndim != lvalues.ndim) or (lobj.size != lvalues.size):
        raise ValueError('Array dimension or shape mismatch')

    N = arr.shape[0]
    Nnew = lobj.size
    Nout = N + Nnew
    out = np.empty(Nout, dtype=arr.dtype)

    indices = np.empty(Nnew, dtype=np.int64)
    indices[:] = lobj
    indices[indices < 0] += N

    # Use stable sorting algorithm such that the sort order of identical
    # elements in indices is predictably the same as in np.insert()
    iorder = np.argsort(indices, kind='mergesort')
    indices[iorder] += np.arange(Nnew)

    mask_old = np.ones(Nout, dtype=np.bool_)
    mask_old[indices] = False

    out[mask_old] = arr
    out[indices] = lvalues

    return out


def _unravel_index_array(indices, shape, order='C'):

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

    coords_shp = (unravel_ndim, ) + tuple(lindices.shape)
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

    indices1d = np.array(indices, dtype=np.int64)

    coords2d = np.unravel_index(indices1d, shape, order)
    coords = coords2d[:, 0]

    return coords


def _ravel_multi_index_array(multi_index, dims, mode='raise', order='C'):

    ravel_dims = np.empty(len(dims), dtype=np.int64)
    for i, d in enumerate(dims):
        ravel_dims[i] = d

    ravel_ndim = ravel_dims.shape[0]

    dtype = multi_index.dtype
    lmulti_index_flat = multi_index.reshape((ravel_ndim, -1))

    one = np.ones(1, dtype=ravel_dims.dtype)
    iwork = np.hstack((one, ravel_dims[:0:-1]))
    ravel_strides = np.cumprod(iwork)[::-1]

    if multi_index.ndim >= 2:
        shp_indices = tuple(multi_index.shape[1:])
    else:
        shp_indices = (1, )

    indices = np.empty(shp_indices, dtype=dtype)
    indices_flat = indices.reshape((-1, ))
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


def ind2sub_array(indices, shape, out=None):
    """
    Converts a flat index or array of flat indices into a tuple of coordinate
    arrays.

    Equivalent to Numpy's unravel_index(), but with fewer features and thus
    hopefully faster.

    Parameters
    ----------
    indices : array_like
        An integer array whose elements are indices into the flattened version
        of an array of dimensions `shape`.
    shape : array_like
        The shape of the array to use for unraveling indices.
    out : np.ndarray or None
        Optional output array (only Numpy arrays supported in Numba mode)

    Returns
    -------
    coords : np.ndarray
        Array of coordinates
    """

    unravel_ndim = len(shape)
    N = len(indices)

    if out is not None:
        coords = out
    else:
        coords = np.empty((unravel_ndim, N), dtype=indices.dtype)

    coords = ind2sub_array_impl(indices, shape, coords)

    return coords


@register_jitable(nogil=True, parallel=False)
def ind2sub_array_impl(indices, shape, out):
    """
    Implementation for array-values ind2sub().
    """

    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    N = len(indices)

    idx_start = unravel_ndim - 1
    idx_step = -1

    for i in range(N):
        val = indices[i]

        if val < 0 or val >= unravel_size:
            raise ValueError('Invalid flat index')

        idx = idx_start

        for j in range(0, unravel_ndim):
            k = shape[idx]
            tmp = val/k
            out[idx, i] = val%k
            val = tmp
            idx += idx_step

    return out


def ind2sub_scalar(indices, shape, out=None):
    """
    Converts a flat index tuple of coordinates.

    Equivalent to Numpy's unravel_index(), but with fewer features and thus
    hopefully faster.

    Parameters
    ----------
    indices : int
        Indices into the flattened version of an array of dimensions `shape`.
    shape : array_like
        The shape of the array to use for unraveling indices.
    out : np.ndarray or None
        Optional output array (only Numpy arrays supported in Numba mode)

    Returns
    -------
    coords : np.ndarray
        Array of coordinates
    """

    unravel_ndim = len(shape)

    if out is not None:
        coords = out
    else:
        coords = np.empty((unravel_ndim, ), dtype=np.int64)

    coords = ind2sub_scalar_impl(indices, shape, coords)

    return coords


@register_jitable(nogil=True, parallel=False)
def ind2sub_scalar_impl(indices, shape, out):

    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    idx_start = unravel_ndim - 1
    idx_step = -1

    val = indices

    if val < 0 or val >= unravel_size:
        raise ValueError('Invalid flat index')

    idx = idx_start

    for j in range(0, unravel_ndim):
        k = shape[idx]
        tmp = val / k
        out[idx] = val % k
        val = tmp
        idx += idx_step

    return out
