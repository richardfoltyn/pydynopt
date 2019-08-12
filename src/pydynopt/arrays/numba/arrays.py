"""
Module implementing basic array creation and manipulation routines that
can be compiled by Numba.

Author: Richard Foltyn
"""


import numpy as np


def _insert(arr, obj, values, axis=None):
    """
    Insert values before the given indices.

    Implements mostly Numpy-compatible replacement for np.insert() that can
    be compiled using Numba.

    Notes
    -----
    -   This implementation ignores the axis argument.
    -   Only integer-valued index arrays are supported.

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


