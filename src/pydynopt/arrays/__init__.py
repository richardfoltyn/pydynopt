"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from numpy import insert
from pydynopt.numba import overload

from .numba import _insert_scalar, _insert_array

__all__ = ['insert']


@overload(insert)
def insert_generic(arr, obj, values, axis=None):
    from numba.types import Integer, Number
    from numba.types.npytypes import Array

    f = None
    if isinstance(obj, Integer) and isinstance(values, Number):
        f = _insert_scalar
    elif isinstance(obj, Array) and isinstance(values, Array):
        f = _insert_array

    return f
