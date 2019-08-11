"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""


import numpy as np
from numpy import insert


def _insert_array(arr, obj, values, axis=None):

    if obj.shape[0] != values.shape[0]:
        raise ValueError('shape mismatch')

    N = arr.shape[0]
    Nnew = obj.shape[0]
    Nout = N + Nnew
    out = np.empty(Nout, dtype=arr.dtype)

    indices = np.empty(Nnew, dtype=np.int64)
    indices[:] = obj
    indices[indices < 0] += N

    # Use stable sorting algorithm such that the sort order of identical
    # elements in indices is predictably the same as in np.insert()
    iorder = np.argsort(indices, kind='mergesort')
    indices[iorder] += np.arange(Nnew)

    mask_old = np.ones(Nout, dtype=np.bool_)
    mask_old[indices] = False

    out[mask_old] = arr
    out[indices] = values

    return out


def _insert_scalar(arr, obj, values, axis=None):

    obj1d = np.array([obj], dtype=np.int64)
    values1d = np.array([values], dtype=arr.dtype)

    # Call Numpy's insert() function. If this is executed in a compiled
    # instance of _insert_scalar(), this call should automatically be
    # redirected to _insert_array() defined above.
    return insert(arr, obj1d, values1d, axis)

