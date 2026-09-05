"""Provide helpers for code that supports both Python and Numba execution.

- Convert Python values to arrays through Numba-compatible implementations.
- Build and initialize dynamic Numba ``jitclass`` instances.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Sequence
from copy import copy as shallow_copy
import sys
from typing import Any

import numpy as np
from numpy import typing as npt

from ..utils import anything_to_tuple
from . import JIT_OPTIONS, overload


def to_array(obj: Any, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
    """Convert an object to a NumPy array in pure-Python code.

    Parameters
    ----------
    obj
        Object to convert.
    dtype
        Data type for the resulting array.

    Returns
    -------
    Array containing the input data.
    """
    x = np.array(obj, dtype=dtype)
    return x


def to_array_iterable(
    obj: Sequence[Any], dtype: npt.DTypeLike | None = None
) -> npt.NDArray[Any]:
    """Convert a tuple or list to a one-dimensional array.

    Parameters
    ----------
    obj
        Values to convert.
    dtype
        Data type for the resulting array.

    Returns
    -------
    One-dimensional array containing the input values.
    """
    n = len(obj)
    ldtype = dtype if dtype is not None else np.float64

    x = np.empty((n,), dtype=ldtype)

    for i in range(n):
        x[i] = obj[i]

    return x


def to_array_default(obj: Any, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
    """Convert an object to an array using a Numba-compatible default.

    Parameters
    ----------
    obj
        Object to convert.
    dtype
        Data type for the resulting array. Defaults to ``float64``.

    Returns
    -------
    Array containing the input data.
    """
    ldtype = np.float64 if dtype is None else dtype

    x = np.array(obj, dtype=ldtype)
    return x


@overload(to_array, jit_options=JIT_OPTIONS)
def array_generic(obj: Any, dtype: Any = None) -> Callable[..., npt.NDArray[Any]]:
    """Select a Numba-compatible implementation of ``to_array``.

    Parameters
    ----------
    obj
        Numba type describing the object to convert.
    dtype
        Numba type describing the requested array data type.

    Returns
    -------
    Implementation suitable for the supplied Numba types.
    """
    from numba import types

    f = to_array_default
    if isinstance(obj, (types.UniTuple, types.List)):
        f = to_array_iterable

    return f


def create_numba_instance(
    obj: Any,
    attrs: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
    init: bool = True,
    copy: bool = False,
    cache: type[Any] | None = None,
    return_type: bool = False,
) -> Any:
    """Create a Numba-compatible instance from an object.

    The function generates a ``jitclass`` specification from selected attributes.
    An object that is already compiled is returned unchanged.

    Parameters
    ----------
    obj
        Object to convert.
    attrs
        Attributes to include in the Numba class definition.
    exclude
        Attributes to exclude from the Numba class definition.
    init
        Whether to initialize attributes from the original object.
    copy
        Whether to copy container-valued attributes during initialization.
    cache
        Existing Numba class to instantiate instead of creating a new class.
    return_type
        Whether to return the dynamically constructed type with the instance.

    Returns
    -------
    Compiled instance, optionally paired with its dynamically constructed type.
    """
    from pydynopt.numba import has_numba, jitclass

    # If this already is a compiled instance, return it immediately.
    if not has_numba or hasattr(obj, '_numba_type_'):
        if return_type:
            return obj, obj.__class__
        return obj

    # Build a specification for jitclass() when the object is not compiled.
    attrs = anything_to_tuple(attrs, force=True)
    if not attrs:
        # Prefer an explicit class-level list of attributes when available.
        if attrs := getattr(obj.__class__, 'NUMBA_ATTRS', None):
            attrs = anything_to_tuple(attrs, force=True)
            present = dir(obj)
            attrs = tuple(attr for attr in attrs if attr in present)
        else:
            # Include public, non-null, non-callable object attributes.
            attrs = tuple(
                attr
                for attr in dir(obj)
                if not attr.startswith('_')
                and getattr(obj, attr) is not None
                and not callable(getattr(obj, attr))
            )

    exclude = anything_to_tuple(exclude, force=True)
    if exclude:
        attrs = tuple(attr for attr in attrs if attr not in exclude)
    elif exclude := getattr(obj.__class__, 'NUMBA_ATTRS_EXCLUDE', None):
        exclude = anything_to_tuple(exclude, force=True)
        attrs = tuple(attr for attr in attrs if attr not in exclude)

    # Exclude class-level Numba configuration attributes.
    attrs = [attr for attr in attrs if not attr.startswith('NUMBA_ATTRS')]

    def __init__(self: Any) -> None:
        """Initialize an empty dynamically generated class."""

    if cache is not None:
        cls_nb = cache
        obj_nb = cls_nb()
    else:
        __dict__: dict[str, Any] = {
            '__init__': __init__,
            '__module__': obj.__class__.__module__,
        }

        name = obj.__class__.__name__ + 'Numba'
        cls = type(name, (), __dict__)

        signature = _build_signature(obj, attrs)
        cls_nb = jitclass(signature)(cls)
        obj_nb = cls_nb()

    if init:
        copy_attributes(obj, obj_nb, attrs=attrs, copy=copy)

    if return_type:
        return obj_nb, cls_nb
    return obj_nb


def _build_signature(obj: Any, attrs: Sequence[str]) -> list[tuple[str, Any]]:
    """Build a Numba ``jitclass`` specification from object attributes.

    Parameters
    ----------
    obj
        Object whose attributes determine the specification.
    attrs
        Attribute names to include.

    Returns
    -------
    Pairs containing each attribute name and its Numba type.
    """
    from numba import types
    from numba.core.errors import NumbaNotImplementedError

    from pydynopt.numba import boolean, float64, from_dtype, int64

    signature: list[tuple[str, Any]] = []
    types_python: dict[type[Any], Any] = {
        int: int64,
        float: float64,
        bool: boolean,
    }

    def process_ndarray(value: Any) -> None:
        """Append a specification entry for an array or NumPy scalar."""
        try:
            nbtype = from_dtype(value.dtype)
        except NumbaNotImplementedError:
            msg = f'Unsupported NumPy dtype {value.dtype}'
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
            elif value.flags.c_contiguous:
                dims = (slice(None, None),) * (value.ndim - 1)
                dims += (slice(None, None, 1),)
                signature.append((attr, nbtype[dims]))
            else:
                msg = f'Array {attr} is not C-contiguous'
                print(msg, file=sys.stderr)
                dims = (slice(None, None),) * value.ndim
                signature.append((attr, nbtype[dims]))
        else:
            signature.append((attr, nbtype))

    for attr in attrs:
        # None-valued attributes were excluded while selecting attributes.
        value = getattr(obj, attr)
        t = type(value)

        if hasattr(value, 'dtype'):
            process_ndarray(value)
        elif isinstance(value, list):
            process_ndarray(np.asarray(value))
        elif t in types_python:
            nbtype = types_python[t]
            signature.append((attr, nbtype))
        elif isinstance(value, tuple):
            is_unif = all(type(i) is type(value[0]) for i in value)
            if is_unif:
                value = np.asarray(value)
                tuple_type = types.UniTuple
                t = tuple_type(from_dtype(value.dtype), len(value))
                signature.append((attr, t))
            else:
                process_ndarray(np.asarray(value))

    return signature


def copy_attributes(
    src: Any,
    dst: Any,
    attrs: Sequence[str] | None = None,
    copy: bool = True,
) -> Any:
    """Copy attributes shared by a source and destination object.

    Parameters
    ----------
    src
        Object from which attributes are read.
    dst
        Object to which attributes are written.
    attrs
        Attribute names to copy. Shared public attributes are used by default.
    copy
        Whether to copy array- and tuple-valued attributes.

    Returns
    -------
    Destination object with the selected attributes assigned.
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
            pass
        elif isinstance(x, list):
            x = np.asarray(x)
        elif isinstance(x, tuple) and copy:
            x = shallow_copy(x)
        setattr(dst, attr, x)

    return dst
