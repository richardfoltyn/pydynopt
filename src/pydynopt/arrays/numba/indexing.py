import numpy as np

from pydynopt.numba import JIT_OPTIONS_INLINE, register_jitable


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


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_array_impl(indices, shape, axis, out):
    """
    Implementation for array-valued ind2sub().
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
        coords = np.empty((n,), dtype=indices.dtype)

    coords = ind2sub_axis_array_impl(indices, shape, laxis, coords)

    return coords


@register_jitable(**JIT_OPTIONS_INLINE)
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
        coords = np.empty((unravel_ndim,), dtype=np.asarray(indices).dtype)

    coords = ind2sub_scalar_impl(indices, shape, 0, coords)

    return coords


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_scalar_impl(indices, shape, axis, out):
    """
    Implementation routine for ind2sub() with scalar arguments.
    """

    unravel_ndim = len(shape)
    val = indices

    for j in range(unravel_ndim - 1, -1, -1):
        k = shape[j]
        tmp = val // k
        out[j] = val % k
        val = tmp

    if val >= shape[0]:
        raise ValueError('Invalid flat index')

    return out


@register_jitable(**JIT_OPTIONS_INLINE)
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


@register_jitable(**JIT_OPTIONS_INLINE)
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

    for j in range(unravel_ndim - 1, -1, -1):
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


@register_jitable(**JIT_OPTIONS_INLINE)
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
        stride[ndim - j - 1] = shape[ndim - j] * stride[ndim - j]

    out[...] = 0
    out_flat = out.reshape((-1,))
    coords_flat = coords.reshape((-1, ndim))

    N = coords_flat.shape[0]

    for i in range(N):
        for j in range(ndim):
            stride_j = stride[j]

            k = coords_flat[i, j]
            if k < 0 or k >= shape[j]:
                raise ValueError('Invalid coordinates')
            out_flat[i] += k * stride_j


@register_jitable(**JIT_OPTIONS_INLINE)
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

    lidx = 0
    stride_ = 1
    for j in range(ndim - 1, -1, -1):
        lidx += coords[j] * stride_
        stride_ *= shape[j]

    return lidx
