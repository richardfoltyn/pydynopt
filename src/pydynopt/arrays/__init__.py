"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from numpy import insert, unravel_index, ravel_multi_index
from pydynopt.numba import overload

from .numba import _insert
from .base import powerspace, logspace

from .base import ind2sub, sub2ind

__all__ = ['insert', 'unravel_index', 'ravel_multi_index',
           'powerspace', 'logspace', 'ind2sub', 'sub2ind']

JIT_OPTIONS = {'parallel': False, 'nogil': True, 'cache': True}


@overload(insert, jit_options=JIT_OPTIONS)
def insert_generic(arr, obj, values, axis=None):
    from numba import types

    f = None
    if isinstance(obj, types.Integer) and isinstance(values, types.Number):
        f = _insert
    elif isinstance(obj, types.Array) and isinstance(values, types.Array):
        if obj.ndim <= 1:
            f = _insert

    return f


@overload(ravel_multi_index, jit_options=JIT_OPTIONS)
def ravel_multi_index_generic(multi_index, dims, mode='raise', order='C'):

    from numba import types
    from .numba.arrays import _ravel_multi_index, _ravel_multi_index_array
    from .numba.arrays import _ravel_multi_index_array1d

    f = None
    if isinstance(multi_index, types.Array) and multi_index.ndim >= 2:
        f = _ravel_multi_index_array
    elif isinstance(multi_index, types.Array) and multi_index.ndim == 1:
        f = _ravel_multi_index_array1d
    else:
        f = _ravel_multi_index

    return f


@overload(unravel_index, jit_options=JIT_OPTIONS)
def unravel_index_generic(indices, shape, order='C'):
    
    from numba import types

    from .numba.arrays import _unravel_index_scalar, _unravel_index_array

    f = None
    if isinstance(indices, types.Integer):
        f = _unravel_index_scalar
    elif isinstance(indices, types.Array):
        f = _unravel_index_array

    return f


@overload(ind2sub, jit_options=JIT_OPTIONS)
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
            from .numba.arrays import ind2sub_axis_scalar as f
        elif axis is not None:
            from .numba.arrays import ind2sub_scalar_impl as f
    elif isinstance(indices, types.Array):
        if axis is not None and out is not None:
            from .numba.arrays import ind2sub_axis_array_impl as f
        elif out is not None:
            from .numba.arrays import ind2sub_array_impl as f

    return f


@overload(ind2sub, jit_options=JIT_OPTIONS)
def ind2sub_generic(indices, shape, axis=None, out=None):
    """
    Return JIT-able function appropriate for the given arguments.

    If neither both axis and out are provided, this function will
    not be called in the first place! We therefore only need to handle
    the case when one of them is missing.
    """

    from numba import types

    f = None
    if isinstance(indices, types.Integer):
        if axis is not None:
            # out missing, will be ignored by implementation
            from .numba.arrays import ind2sub_axis_scalar_impl as f
        else:
            # axis is missing, so implementation will return 1d-array with
            # all coordinates. out array will be allocated if missing.
            from .numba.arrays import ind2sub_scalar as f
    elif isinstance(indices, types.Array):
        if axis is not None:
            # out missing, will be allocated on demand
            from .numba.arrays import ind2sub_axis_array as f
        else:
            # axis is missing, so implementation will return 2d-array with
            # all coordinates. out array will be allocated if missing.
            from .numba.arrays import ind2sub_array as f

    return f


@overload(sub2ind, jit_options=JIT_OPTIONS)
def sub2ind_generic(coords, shape, out=None):

    from numba import types

    from .numba.arrays import sub2ind_array, sub2ind_scalar

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


@overload(sub2ind, jit_options=JIT_OPTIONS)
def sub2ind_impl_generic(coords, shape, out):
    from numba import types

    from .numba.arrays import sub2ind_array_impl

    f = None
    if isinstance(coords, types.Array) and out is not None:
        if coords.ndim >= 2:
            f = sub2ind_array_impl

    return f

