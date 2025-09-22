"""
Helper routines to facilitate Python and Numba-compatible code.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import sys
from collections.abc import Sequence
from typing import Optional, Any

import numpy as np

from . import overload, JIT_OPTIONS
from ..utils import anything_to_tuple


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

    x = np.empty((n,), dtype=ldtype)

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


@overload(to_array, jit_options=JIT_OPTIONS)
def array_generic(obj, dtype=None):

    from numba import types

    f = to_array_default
    if isinstance(obj, (types.UniTuple, types.List)):
        f = to_array_iterable

    return f


def create_numba_instance(
    obj,
    attrs: Optional[str | Sequence[str]] = None,
    exclude: Optional[str | Sequence[str]] = None,
    init: bool = True,
    copy: bool = False,
    cache: Optional[type] = None,
    return_type: bool = False
):
    """
    Automatically create numba-fied instance of a given object.
    The routine attempts to automatically generate a class signature that
    can be used to instantiate a numba-fied object.

    If the given object is already a compiled Numba type, it is immediately
    returned as-is.

    Parameters
    ----------
    obj : object
    attrs : str or Sequence of str, optional
        List of attributes of `obj` that should be included in the Numba-fied
        class definition.
    exclude : str or Sequence of str, optional
        List of attributes of `obj` that should be excluded in the Numba-fied
        class definition.
    init : bool, optional
        If true, additionally initialize the attribute values in the Numba
        instance with values from original objects.
    copy : bool, optional
        If true, create copies of container objects such as tuples or Numpy
        arrays when initializing attributes of Numba instance.
    cache: type, optional
        If not None, use given object as Numba-fied instance and optionally update
        its attributes. Avoids building a signature and creating a new type.
    return_type : bool
        If True, return the dynamically constructed type.

    Returns
    -------
    object
        Instance of compiled Numba type.
    """

    from pydynopt.numba import jitclass
    from pydynopt.numba import has_numba

    # if this already is a compiled instance, return it immediately
    if not has_numba or hasattr(obj, '_numba_type_'):
        if return_type:
            return obj, obj.__class__
        else:
            return obj

    # object is not an instance of a Numba type, we need to build
    # signature for jitclass().
    attrs = anything_to_tuple(attrs)
    if not attrs:
        # Check whether class has NUMBA_ATTRS attribute which contains
        # the attributes to be included in Numbafied instance.
        if attrs := getattr(obj.__class__, 'NUMBA_ATTRS', None):
            attrs = anything_to_tuple(attrs)
            # Keep only existing attributes
            present = dir(obj)
            attrs = tuple(attr for attr in attrs if attr in present)
        else:
            # Assume that all object attributes should be included as long as
            # they are not for internal use or None or callable
            attrs = tuple(
                attr
                for attr in dir(obj)
                if not attr.startswith('_')
                and getattr(obj, attr) is not None
                and not callable(getattr(obj, attr))
            )

    exclude = anything_to_tuple(exclude)
    if exclude:
        attrs = tuple(attr for attr in attrs if attr not in exclude)
    elif exclude := getattr(obj.__class__, 'NUMBA_ATTRS_EXCLUDE', None):
        exclude = anything_to_tuple(exclude)
        attrs = tuple(attr for attr in attrs if attr not in exclude)

    # Automatically exclude meta-attributes
    attrs = [attr for attr in attrs if not attr.startswith('NUMBA_ATTRS')]

    # Empty init, expected by Numba jitclass()
    def __init__(self):
        pass

    if cache is not None:
        cls_nb = cache
        obj_nb = cls_nb()
    else:
        __dict__ = {'__init__': __init__, '__module__': obj.__class__.__module__}

        # Create class name with Numba suffix
        name = obj.__class__.__name__ + 'Numba'
        cls = type(name, (), __dict__)

        signature = _build_signature(obj, attrs)

        cls_nb = jitclass(signature)(cls)

        obj_nb = cls_nb()

    if init:
        copy_attributes(obj, obj_nb, attrs=attrs, copy=copy)

    if return_type:
        return obj_nb, cls_nb
    else:
        return obj_nb


def _build_signature(obj, attrs: Sequence[str]):
    """
    Build a signature for numba.jitclass from the list of attributes names
    and their associated types.

    Parameters
    ----------
    obj : object
    attrs : list of str

    Returns
    -------
    signature : list
        List of tuples containing (name, type) pairs.
    """

    from pydynopt.numba import boolean, int64, float64
    from pydynopt.numba import from_dtype
    from numba.types import UniTuple

    signature = []

    types_python = {int: int64, float: float64, bool: boolean}

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
                    dims = (slice(None, None),) * (value.ndim - 1)
                    dims += (slice(None, None, 1),)
                    signature.append((attr, nbtype[dims]))
                else:
                    msg = f'Array {attr} is not C-contiguous'
                    print(msg, file=sys.stderr)
                    dims = (slice(None, None),) * value.ndim
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
        elif isinstance(value, list):
            # Convert to Numpy array in numba instance
            value = np.asarray(value)
            process_ndarray(value)
        elif t in types_python:
            nbtype = types_python[t]
            signature.append((attr, nbtype))
        elif isinstance(value, tuple):
            # Check for uniform types
            is_unif = all(type(i) == type(value[0]) for i in value)
            if is_unif:
                value = np.asarray(value)
                t = UniTuple(from_dtype(value.dtype), len(value))
            else:
                # Store as numpy array instead
                process_ndarray(np.asarray(value))
            signature.append((attr, t))

    return signature


def copy_attributes(
    src: Any, dst: Any, attrs: Optional[Sequence[str]] = None, copy: bool = True
) -> Any:
    """
    Copy attributes from src that at also present in dst into dst.

    Parameters
    ----------
    src : object
    dst : object
    attrs : Sequence of str, optional
    copy : bool
        If true, copy array-valued attributes instead of referencing the
        original array.

    Returns
    -------
    params : NumbaParams
    """

    if attrs is None:
        attrs = [k for k in dir(dst) if not k.startswith('_') and hasattr(src, k)]

    for attr in attrs:
        x = getattr(src, attr)
        if x is None:
            continue

        if isinstance(x, np.ndarray) and copy:
            x = np.copy(x)
        elif np.isscalar(x):
            # Assign as is
            pass
        elif isinstance(x, list):
            # Always creates a copy
            x = np.asarray(x)
        elif isinstance(x, tuple) and copy:
            x = copy.copy(x)
        setattr(dst, attr, x)

    return dst
