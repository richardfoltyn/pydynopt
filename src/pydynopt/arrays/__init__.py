"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from numpy import insert
from pydynopt.numba import overload

from .numba import _insert
from .base import powerspace

__all__ = ['insert', 'powerspace']


@overload(insert)
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
