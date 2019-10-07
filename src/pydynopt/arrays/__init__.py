"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from numpy import insert, unravel_index, ravel_multi_index
from pydynopt.numba import overload

from .numba import _insert
from .base import powerspace

from .base import ind2sub, sub2ind

__all__ = ['insert', 'unravel_index', 'ravel_multi_index',
           'powerspace', 'ind2sub', 'sub2ind']

JIT_OPTIONS = {'parallel': False, 'nogil': True, 'cache': True}


@overload(insert, jit_options=JIT_OPTIONS)
def insert_generic(arr, obj, values, axis=None):
    from numba.types import Integer, Number
    from numba.types.npytypes import Array

    f = None
    if isinstance(obj, Integer) and isinstance(values, Number):
        f = _insert
    elif isinstance(obj, Array) and isinstance(values, Array):
        if obj.ndim <= 1:
            f = _insert

    return f


@overload(ravel_multi_index, jit_options=JIT_OPTIONS)
def ravel_multi_index_generic(multi_index, dims, mode='raise', order='C'):

    from numba.types.npytypes import Array
    from .numba.arrays import _ravel_multi_index, _ravel_multi_index_array
    from .numba.arrays import _ravel_multi_index_array1d

    f = None
    if isinstance(multi_index, Array) and multi_index.ndim >= 2:
        f = _ravel_multi_index_array
    elif isinstance(multi_index, Array) and multi_index.ndim == 1:
        f = _ravel_multi_index_array1d
    else:
        f = _ravel_multi_index

    return f


@overload(unravel_index, jit_options=JIT_OPTIONS)
def unravel_index_generic(indices, shape, order='C'):
    from numba.types import Integer
    from numba.types.npytypes import Array

    from .numba.arrays import _unravel_index_scalar, _unravel_index_array

    f = None
    if isinstance(indices, Integer):
        f = _unravel_index_scalar
    elif isinstance(indices, Array):
        f = _unravel_index_array

    return f


@overload(ind2sub, jit_options=JIT_OPTIONS)
def ind2sub_impl_generic(indices, shape, out):
    from numba.types import Integer
    from numba.types.npytypes import Array

    from .numba.arrays import ind2sub_array_impl, ind2sub_scalar_impl

    f = None
    if isinstance(indices, Integer) and out is not None:
        f = ind2sub_scalar_impl
    elif isinstance(indices, Array) and out is not None:
        f = ind2sub_array_impl

    return f


@overload(ind2sub, jit_options=JIT_OPTIONS)
def ind2sub_generic(indices, shape, out=None):

    from numba.types import Integer
    from numba.types.npytypes import Array

    from .numba.arrays import ind2sub_array, ind2sub_scalar

    f = None
    if isinstance(indices, Integer) and out is None:
        f = ind2sub_scalar
    elif isinstance(indices, Array) and out is None:
        f = ind2sub_array

    return f


@overload(sub2ind, jit_options=JIT_OPTIONS)
def sub2ind_generic(coords, shape, out=None):

    from numba.types.npytypes import Array

    from .numba.arrays import sub2ind_array, sub2ind_scalar

    f = None
    if isinstance(coords, Array) and out is None:
        if coords.ndim == 1:
            f = sub2ind_scalar
        elif coords.ndim >= 2:
            f = sub2ind_array

    return f


@overload(sub2ind, jit_options=JIT_OPTIONS)
def sub2ind_impl_generic(coords, shape, out):
    from numba.types.npytypes import Array

    from .numba.arrays import sub2ind_array_impl

    f = None
    if isinstance(coords, Array) and out is not None:
        if coords.ndim >= 2:
            f = sub2ind_array_impl

    return f

