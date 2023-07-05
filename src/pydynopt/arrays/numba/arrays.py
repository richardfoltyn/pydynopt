"""
Module implementing basic array creation and manipulation routines that
can be compiled by Numba.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

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


def ind2sub_array(indices, shape, axis=None, out=None):
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
    axis : int, optional
        Ignored, only present to ensure compatible function signatures.
    out : np.ndarray or None
        Optional output array (only Numpy arrays supported in Numba mode)

    Returns
    -------
    coords : np.ndarray
        Array of coordinates
    """

    unravel_ndim = len(shape)
    n = len(indices)

    if out is not None:
        coords = out
    else:
        coords = np.empty((unravel_ndim, n), dtype=indices.dtype)

    coords = ind2sub_array_impl(indices, shape, 0, coords)

    return coords


@register_jitable(nogil=True, parallel=False)
def ind2sub_array_impl(indices, shape, axis, out):
    """
    Implementation for array-values ind2sub().
    """

    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    n = len(indices)

    for i in range(n):
        val = indices[i]

        if val < 0 or val >= unravel_size:
            raise ValueError('Invalid flat index')

        for j in range(unravel_ndim-1, -1, -1):
            k = shape[j]
            tmp = val/k
            out[j, i] = val % k
            val = tmp

    return out


def ind2sub_axis_array(indices, shape, axis=None, out=None):
    """
    Converts a flat index or array of flat indices into a coordinate
    arrays for the given axis.

    Parameters
    ----------
    indices : array_like
        An integer array whose elements are indices into the flattened version
        of an array of dimensions `shape`.
    shape : array_like
        The shape of the array to use for unraveling indices.
    axis : int, optional
        Axis along which coordinate array should be returned. If None,
        the coordinates for the leading axis are returned.
    out : np.ndarray or None
        Optional output array (only Numpy arrays supported in Numba mode)

    Returns
    -------
    coords : np.ndarray
        Array of coordinates
    """

    n = len(indices)
    laxis = 0 if axis is None else axis

    if out is not None:
        coords = out
    else:
        coords = np.empty((n, ), dtype=indices.dtype)

    coords = ind2sub_axis_array_impl(indices, shape, laxis, coords)

    return coords


@register_jitable(nogil=True, parallel=False)
def ind2sub_axis_array_impl(indices, shape, axis, out):
    """
    Implementation for array-valued ind2sub(..., axis, out).
    """

    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    n = len(indices)

    for i in range(n):
        val = indices[i]

        if val < 0 or val >= unravel_size:
            raise ValueError('Invalid flat index')

        for j in range(unravel_ndim - 1, -1, -1):
            k = shape[j]
            tmp = val / k
            if j == axis:
                out[i] = val % k
                break
            val = tmp

    return out


def ind2sub_scalar(indices, shape, axis=None, out=None):
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
    axis : int, optional
        Ignored, only present to ensure compatible function signatures.
    out : np.ndarray, optional
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
        coords = np.empty((unravel_ndim, ), dtype=np.asarray(indices).dtype)

    coords = ind2sub_scalar_impl(indices, shape, 0, coords)

    return coords


@register_jitable(nogil=True, parallel=False)
def ind2sub_scalar_impl(indices, shape, axis, out):
    """
    Implementation routine for ind2sub() with scalar arguments.
    """

    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    val = indices

    if val < 0 or val >= unravel_size:
        raise ValueError('Invalid flat index')

    for j in range(unravel_ndim-1, -1, -1):
        k = shape[j]
        tmp = val / k
        out[j] = val % k
        val = tmp

    return out


@register_jitable(nogil=True, parallel=False)
def ind2sub_axis_scalar(indices, shape, axis, out):
    """
    Converts a flat index into a coordinate for the given axis.

    Wrapper around implementation routine to support `out` arguments that
    are not None.

    Parameters
    ----------
    indices : int
        Index into the flattened version of an array of dimensions `shape`.
    shape : array_like
        The shape of the array to use for unraveling indices.
    axis : int, optional
        Axis along which coordinate should be returned. If None,
        the coordinates for the leading axis are returned.
    out : np.ndarray
        Array to store coordinate along requested axis as its first element.

    Returns
    -------
    int
        Coordinate along the requested axis.
    """

    lout = ind2sub_axis_scalar_impl(indices, shape, axis)
    out[0] = lout
    return lout


@register_jitable(nogil=True, parallel=False)
def ind2sub_axis_scalar_impl(indices, shape, axis=None, out=None):
    """
    Converts a flat index into a coordinate for the given axis.

    Parameters
    ----------
    indices : int
        Index into the flattened version of an array of dimensions `shape`.
    shape : array_like
        The shape of the array to use for unraveling indices.
    axis : int, optional
        Axis along which coordinate should be returned. If None,
        the coordinates for the leading axis are returned.
    out : np.ndarray, optional
         Ignored, only present to ensure compatible function signatures.

    Returns
    -------
    int
        Coordinate along the requested axis.
    """

    laxis = 0 if axis is None else axis

    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    val = indices

    if val < 0 or val >= unravel_size:
        raise ValueError('Invalid flat index')

    lout = 0

    for j in range(unravel_ndim-1, -1, -1):
        k = shape[j]
        tmp = val / k
        if j == laxis:
            lout = val % k
            break
        val = tmp

    return lout


def sub2ind_array(coords, shape, out=None):
    """
    Converts an array of indices (coordinates) into a multi-dimensional array
    into an array of flat indices.

    Parameters
    ----------
    coords : np.ndarray
        2-dimensional integer array of coordinates. Each row contains the
        coordinates for one dimension.
    shape : array_like
        Shape of array into which indices from `coords` apply.
    out : np.ndarray or None
        Optional output array of flat indices.

    Returns
    -------
    out : np.ndarray
        Array of indices into flatted array.
    """

    if out is not None:
        sub2ind_array_impl(coords, shape, out)
        return out
    else:
        shp = coords.shape[1:]
        lout = np.empty(shp, dtype=coords.dtype)
        sub2ind_array_impl(coords, shape, lout)
        return lout


@register_jitable(nogil=True, parallel=False)
def sub2ind_array_impl(coords, shape, out):
    """
    Implementation of sub2ind with mandatory `out` argument.

    Parameters
    ----------
    coords : np.ndarray
        2-dimensional integer array of coordinates. Each row contains the
        coordinates for one dimension.
    shape : array_like
        Shape of array into which indices from `coords` apply.
    out : np.ndarray
        Array of indices into flatted array.
    """

    ndim = len(shape)
    stride = np.empty(ndim, dtype=np.int64)
    stride[-1] = 1

    for j in range(1, ndim):
        stride[ndim - j - 1] = shape[ndim-j] * stride[ndim - j]

    out[...] = 0
    out_flat = out.reshape((-1, ))
    coords_flat = coords.reshape((-1, ndim))

    N = coords_flat.shape[0]

    for i in range(N):
        for j in range(ndim):
            stride_j = stride[j]

            k = coords_flat[i, j]
            if k < 0 or k >= shape[j]:
                raise ValueError('Invalid coordinates')
            out_flat[i] += k * stride_j


@register_jitable(nogil=True, parallel=False)
def sub2ind_scalar(coords, shape, out=None):
    """
    Convert a tuple of indices (coordinates) into a multi-dimension array
    to an index into a flat array.

    Parameters
    ----------
    coords : np.ndarray
        1d-array of indices (coordinates) into multi-dimensional array.
    shape : array_like
        Shape of array into which indices from `coords` apply.
    out : object
        Ignored, only present for API-compativility with array-values
        version of this function.

    Returns
    -------
    out : int
        Index into flat array.
    """

    ndim = len(shape)
    if len(coords) != ndim:
        raise ValueError('Incompatible coordinate array size')
    stride = np.empty(ndim, dtype=np.int64)
    stride[-1] = 1

    for j in range(1, ndim):
        stride[ndim-j-1] = shape[ndim-j] * stride[ndim-j]

    lidx = 0

    for j in range(ndim):
        k = coords[j]
        if k < 0 or k >= shape[j]:
            raise ValueError('Invalid coordinates')
        lidx += coords[j] * stride[j]

    return lidx

