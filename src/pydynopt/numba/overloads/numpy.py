"""
Overloads for NumPy functions not (fully) supported by Numba. This includes functions
which are only supported for a sub-set of arguments.

Author: Richard Foltyn

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/
"""
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
from numpy import cumsum, insert

from pydynopt.numba import JIT_OPTIONS, jit, overload

__all__ = ['cumsum', 'insert']


def cumsum_dispatch(x: np.ndarray, axis: Optional[int] = None) -> Union[callable, None]:
    """
    Overload for numpy.cumsum() with second argument (axis) given, which is not
    supported by Numba.

    Parameters
    ----------
    x : np.ndarray
    axis : int, optional

    Returns
    -------
    callable or None
    """

    if axis is None:
        return np.cumsum
    elif x.ndim == 2 and axis is not None:

        def _impl(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
            xout = np.empty_like(x)
            if axis == 0:
                xout[0] = x[0]
                for i in range(1, x.shape[0]):
                    xout[i] = xout[-1] + x[i]
            else:
                # Note that function is guaranteed to be called only of axis is not
                # None, otherwise the dispatcher does not return it for the given
                # arguments.
                xout[:, 0] = x[:, 0]
                for j in range(1, x.shape[1]):
                    xout[:, j] = xout[:, j - 1] + x[:, j]
            return xout

        return _impl


def overload_cumsum(jit_options: Optional[Mapping] = None):
    """
    Overload np.cumsum() for 2d arrays if current version of Numba does not support
    axis argument.

    Parameters
    ----------
    jit_options : mapping
        JIT options passed to Numba's overload()

    """

    try:

        def f(x, axis):
            return np.cumsum(x, axis=axis)

        kw = dict(jit_options) if jit_options else JIT_OPTIONS
        kw["nopython"] = True

        fjit = jit(f, **kw)
        fjit(np.zeros((2, 2)), axis=1)
        # If this compiles and runs, current Numba version supports cumsum with axis
        # argument. Nothing else needs to be done
        return
    except:
        # Ignore error, return custom implementation
        pass

    overload(np.cumsum, jit_options=jit_options)(cumsum_dispatch)


def _insert(arr, obj, values, axis=None):
    """
    Insert values before the given indices.

    Implements mostly Numpy-compatible replacement for np.insert() that can
    be compiled using Numba.

    Notes
    -----
    - This implementation ignores the axis argument.
    - Only integer-valued index arrays are supported.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    obj : np.ndarray or int
        Object that defines the index or indices before which values is
        inserted. Must be integer-valued!
    values : np.ndarray or numeric
        Values to insert into `arr`.
    axis : object
        Ignored.

    Returns
    -------
    out : np.ndarray
        A copy of arr with values inserted.
    """

    lobj = np.asarray(obj)
    lvalues = np.asarray(values)

    if lobj.ndim > 1:
        raise ValueError('Unsupported array dimension')

    if (lobj.ndim != lvalues.ndim) or (lobj.size != lvalues.size):
        raise ValueError('Array dimension or shape mismatch')

    N = arr.shape[0]
    Nnew = lobj.size
    Nout = N + Nnew
    out = np.empty(Nout, dtype=arr.dtype)

    indices = np.empty(Nnew, dtype=np.int64)
    indices[:] = lobj
    indices[indices < 0] += N

    # Use stable sorting algorithm such that the sort order of identical
    # elements in indices is predictably the same as in np.insert()
    iorder = np.argsort(indices, kind='mergesort')
    indices[iorder] += np.arange(Nnew)

    mask_old = np.ones(Nout, dtype=np.bool_)
    mask_old[indices] = False

    out[mask_old] = arr
    out[indices] = lvalues

    return out


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
