"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from numpy import insert, unravel_index, ravel_multi_index
from pydynopt.numba import overload

from .numba import _insert
from .base import powerspace

__all__ = ['insert', 'unravel_index', 'ravel_multi_index',
           'powerspace']

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
