"""
Helper routines to facilitate Python and Numba-compatible code.

Author: Richard Foltyn
"""

import numpy as np

from . import overload


def to_array(obj, dtype=None):
    """
    Wrapper around np.array() to be used in pure-Python code.

    Parameters
    ----------
    obj :
    dtype

    Returns
    -------
    x : np.ndarray
    """

    x = np.array(obj, dtype=dtype)
    return x


def to_array_iterable(obj, dtype=None):
    """
    Helper routine to convert tuples and lists to 1d-arrays.

    Parameters
    ----------
    obj : tuple or list
    dtype

    Returns
    -------
    x : np.ndarray
    """

    n = len(obj)
    if dtype is not None:
        ldtype = dtype
    else:
        ldtype = np.float64

    x = np.empty((n, ), dtype=ldtype)

    for i in range(n):
        x[i] = obj[i]

    return x


def to_array_default(obj, dtype=None):
    """
    Helper function to convert objects to Numpy arrays in Numba code.

    Parameters
    ----------
    obj
    dtype

    Returns
    -------
    x : np.ndarray
    """

    if dtype is None:
        ldtype = np.float64
    else:
        ldtype = dtype

    x = np.array(obj, dtype=ldtype)
    return x


@overload(to_array, jit_options={'nogil': True, 'parallel': False})
def array_generic(obj, dtype=None):

    from numba.types import UniTuple, List

    f = to_array_default
    if isinstance(obj, (UniTuple, List)):
        f = to_array_iterable

    return f

