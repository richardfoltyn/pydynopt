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


def create_numba_instance(obj, attrs=None):
    """
    Automatically create numba-fied instance of a given object.
    The routine attempts to automatically generate a class signature that
    can be used to instantiate a numba-fied object.

    Parameters
    ----------
    obj : object
    attrs : array_like, optional
        List of attributes of `obj` that should be included in the Numba-fied
        class definition.

    Returns
    -------

    """

    from pydynopt.numba import jitclass, boolean
    from pydynopt.numba import int8, int32, int64
    from pydynopt.numba import float32, float64

    if attrs is None:
        # Assume that all object attributes should be included as long as they
        # are not for internal use or None.
        attrs = [attr for attr in dir(obj)
                 if not attr.startswith('_') and getattr(obj, attr) is not None]

    # Empty init, expected by Numba jitclass()
    def __init__(self):
        pass

    __dict__ = {'__init__': __init__, '__module__': obj.__class__.__module__}

    # Create class name with Numba suffix
    name = obj.__class__.__name__ + 'Numba'
    cls = type(name, (), __dict__)

    signature = []

    types_python = {int: int64,
                    float: float64,
                    bool: boolean}

    types_numpy = {np.int8: int8,
                   np.int32: int32,
                   np.int64: int64,
                   np.float32: float32,
                   np.float64: float64,
                   np.bool: boolean,
                   np.bool_: boolean}

    for attr in attrs:
        # Note: we excluded None-values attributes above
        value = getattr(obj, attr)
        t = type(value)

        if hasattr(value, 'dtype'):
            # List of equivalent types
            keys = tuple(k for k in types_numpy if k == value.dtype)
            if len(keys) == 0:
                msg = 'Unsupported Numpy dtype {}'.format(value.dtype)
                raise ValueError(msg)

            nbtype = types_numpy[keys[0]]

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    signature.append((attr, nbtype))
                elif value.ndim == 1:
                    signature.append((attr, nbtype[::1]))
                else:
                    dims = (slice(None, None), ) * (value.ndim - 1)
                    dims += (slice(None, None, 1), )
                    signature.append((attr, nbtype[dims]))
            else:
                # scalar type
                signature.append((attr, nbtype))

        elif t in types_python:
            nbtype = types_python[t]
            signature.append((attr, nbtype))

    cls_nb = jitclass(signature)(cls)

    obj_nb = cls_nb()

    return obj_nb
