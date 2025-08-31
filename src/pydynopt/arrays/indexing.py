import numpy as np

from pydynopt.arrays.numba.indexing import (
    ind2sub_array,
    ind2sub_array_impl,
    ind2sub_axis_array,
    ind2sub_axis_array_impl,
    ind2sub_axis_scalar,
    ind2sub_axis_scalar_impl,
    ind2sub_scalar,
    ind2sub_scalar_impl,
    sub2ind_array,
    sub2ind_array_impl,
    sub2ind_scalar,
)
from pydynopt.numba import JIT_OPTIONS, JIT_OPTIONS_INLINE, jit, overload

ind2sub_scalar_jit = jit(ind2sub_scalar, **JIT_OPTIONS)
ind2sub_array_jit = jit(ind2sub_array, **JIT_OPTIONS)
ind2sub_scalar_impl_jit = jit(ind2sub_scalar_impl, **JIT_OPTIONS)
ind2sub_array_impl_jit = jit(ind2sub_array_impl, **JIT_OPTIONS)
ind2sub_axis_scalar_jit = jit(ind2sub_axis_scalar, **JIT_OPTIONS)
ind2sub_axis_array_jit = jit(ind2sub_axis_array, **JIT_OPTIONS)
ind2sub_axis_scalar_impl_jit = jit(ind2sub_axis_scalar_impl, **JIT_OPTIONS)
ind2sub_axis_array_impl_jit = jit(ind2sub_axis_array_impl, **JIT_OPTIONS)

sub2ind_scalar_jit = jit(sub2ind_scalar, **JIT_OPTIONS)
sub2ind_array_jit = jit(sub2ind_array, **JIT_OPTIONS)


def ind2sub(indices, shape, axis=None, out=None):
    """
    Converts a flat index or array of flat indices into a tuple of coordinate
    arrays.

    Equivalent to Numpy's unravel_index(), but with fewer features and thus
    hopefully faster.

    Parameters
    ----------
    indices : int or array_like
        An integer or integer array whose elements are indices into the
        flattened version of an array of dimensions `shape`.
    axis : int, optional
        If not None, restricts the return array to contain only the coordinates
        along `axis`.
    shape : array_like
        The shape of the array to use for unraveling indices.
    out : np.ndarray or None
        Optional output array (only Numpy arrays supported in Numba mode)

    Returns
    -------
    coords : int or np.ndarray
        Array of coordinates
    """

    if np.isscalar(indices):
        if axis is None:
            if out is None:
                # Pre-allocate array here so we don't have array dtype
                # conflicts in the JIT-able code.
                out = np.empty(len(shape), dtype=np.asarray(indices).dtype)

            out = ind2sub_scalar_impl_jit(indices, shape, axis, out)
        else:
            if out is not None:
                # Implementation routine ignores out argument and only returns
                # a scalar.
                out = ind2sub_axis_scalar_impl_jit(indices, shape, axis)
            else:
                # JIT-able routine writes index into first element of out!
                out = ind2sub_axis_scalar_jit(indices, shape, axis, out)
    else:
        if out is None:
            shp = (len(indices),)
            if axis is None:
                # With axis=None, output will be a 2d-array.
                shp += (len(shape),)

        if axis is None:
            out = ind2sub_array_impl_jit(indices, shape, axis, out)
        else:
            out = ind2sub_axis_array_impl_jit(indices, shape, axis, out)

    return out


@overload(ind2sub, jit_options=JIT_OPTIONS_INLINE)
def ind2sub_impl_generic(indices, shape, axis, out):
    """
    Return JIT-able function when all arguments are present.

    The routine requires that the `out` argument is not explicitly passed
    as None, since the JIT-able functions returned here for the most part
    assume that the `out` array is allocated (except for the scalar case
    when axis=None).

    """

    from numba import types

    f = None
    if isinstance(indices, types.Integer):
        if axis is not None and out is not None:
            f = ind2sub_axis_scalar
        elif axis is not None:
            f = ind2sub_scalar_impl
    elif isinstance(indices, types.Array):
        if axis is not None and out is not None:
            f = ind2sub_axis_array_impl
        elif out is not None:
            f = ind2sub_array_impl

    return f


@overload(ind2sub, jit_options=JIT_OPTIONS_INLINE)
def ind2sub_generic(indices, shape, axis=None, out=None):
    """
    Return JIT-able function appropriate for the given arguments.

    If both axis and out are provided, this function will
    not be called in the first place! We therefore only need to handle
    the case when one of them is missing.
    """

    from numba import types

    f = None
    if isinstance(indices, types.Integer):
        if axis is not None:
            # out missing, will be ignored by implementation
            f = ind2sub_axis_scalar_impl
        else:
            # axis is missing, so implementation will return 1d-array with
            # all coordinates. out array will be allocated if missing.
            f = ind2sub_scalar
    elif isinstance(indices, types.Array):
        if axis is not None:
            # out missing, will be allocated on demand
            f = ind2sub_axis_array
        else:
            # axis is missing, so implementation will return 2d-array with
            # all coordinates. out array will be allocated if missing.
            f = ind2sub_array

    return f


def sub2ind(coords, shape, out=None):
    """
    Converts an array of indices (coordinates) into a multi-dimensional array
    into an array of flat indices.

    Parameters
    ----------
    coords : array_like
        Integer array of coordinates. Coordinates for each dimension are
        arranged along the first axis.
    shape : array_like
        Shape of array into which indices from `coords` apply.
    out : np.ndarray or None
        Optional output array of flat indices.

    Returns
    -------
    out : np.ndarray
        Array of indices into flatted array.
    """

    coords = np.atleast_1d(coords)

    if coords.ndim == 1:
        out = sub2ind_scalar_jit(coords, shape, out)
    elif coords.ndim == 2:
        out = sub2ind_array_jit(coords, shape, out)

    return out


@overload(sub2ind, jit_options=JIT_OPTIONS_INLINE)
def sub2ind_generic(coords, shape, out=None):

    from numba import types

    from .numba.indexing import sub2ind_array, sub2ind_scalar

    f = None
    if out is None:
        if isinstance(coords, types.Array):
            if coords.ndim == 1:
                f = sub2ind_scalar
            elif coords.ndim >= 2:
                f = sub2ind_array
        else:
            # probably coords argument is a tuple or some other array-like
            # object that can be indexed. Assume it's one-dimensional.
            f = sub2ind_scalar

    return f


@overload(sub2ind, jit_options=JIT_OPTIONS_INLINE)
def sub2ind_impl_generic(coords, shape, out):
    from numba import types

    f = None
    if isinstance(coords, types.Array) and out is not None:
        if coords.ndim >= 2:
            f = sub2ind_array_impl

    return f
