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


def create_numba_instance(obj, attrs=None, init=True, copy=False):
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
    init : bool, optional
        If true, additionally initialize the attribute values in the Numba
        instance with values from original objects.
    copy : bool, optional
        If true, create copies of container objects such as tuples or Numpy
        arrays when initializing attributes of Numba instance.

    Returns
    -------

    """

    import sys

    from pydynopt.numba import jitclass, boolean
    from pydynopt.numba import int8, int32, int64
    from pydynopt.numba import float32, float64
    from pydynopt.numba import from_dtype

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

    def process_ndarray(value):
        try:
            nbtype = from_dtype(value.dtype)
        except:
            msg = 'Unsupported Numpy dtype {}'.format(value.dtype)
            print(msg, file=sys.stderr)
            return

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                signature.append((attr, nbtype))
            elif value.ndim == 1:
                if value.flags.c_contiguous:
                    signature.append((attr, nbtype[::1]))
                else:
                    msg = f'Array {attr} is not C-contiguous'
                    print(msg, file=sys.stderr)
                    signature.append((attr, nbtype[:]))
            else:
                if value.flags.c_contiguous:
                    dims = (slice(None, None),)*(value.ndim - 1)
                    dims += (slice(None, None, 1),)
                    signature.append((attr, nbtype[dims]))
                else:
                    msg = f'Array {attr} is not C-contiguous'
                    print(msg, file=sys.stderr)
                    dims = (slice(None, None),)*value.ndim
                    signature.append((attr, nbtype[dims]))
        else:
            # scalar type
            signature.append((attr, nbtype))

    for attr in attrs:
        # Note: we excluded None-values attributes above
        value = getattr(obj, attr)
        t = type(value)

        if hasattr(value, 'dtype'):
            process_ndarray(value)
        elif isinstance(value, (list, tuple)):
            # Convert to Numpy array in numba instance
            value = np.asarray(value)
            process_ndarray(value)
        elif t in types_python:
            nbtype = types_python[t]
            signature.append((attr, nbtype))

    cls_nb = jitclass(signature)(cls)

    obj_nb = cls_nb()

    if init:
        copy_attributes(obj, obj_nb, copy=copy)

    return obj_nb


def copy_attributes(src, dst, copy=True):
    """
    Copy attributes from src that at also present in dst into dst.

    Parameters
    ----------
    src : object
    dst : object
    copy : bool
        If true, copy array-valued attributes instead of referencing the
        original array.

    Returns
    -------
    params : NumbaParams
    """
    for attr in dir(dst):
        if not attr.startswith('_') and hasattr(src, attr):
            x = getattr(src, attr)
            if x is not None:
                if (copy and not np.isscalar(x)) or isinstance(x, (tuple, list)):
                    x = np.copy(x)
                setattr(dst, attr, x)

    return dst
