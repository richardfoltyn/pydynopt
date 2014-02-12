from __future__ import division, print_function, absolute_import
import numpy as np

import pydynopt.utils.cutils as cu


def cartesian_op(a_tup, axis=0, op=None, dtype=None):

    in_arrays = [np.atleast_2d(x).swapaxes(axis, 0) for x in a_tup]
    in_dim = np.array([x.shape for x in in_arrays], dtype=np.uint32)

    ncols, nrows = np.prod(in_dim[:, 1]), np.sum(in_dim[:, 0])

    if dtype is None:
        dtype = a_tup[0].dtype

    res = np.empty((nrows, ncols), dtype=dtype)

    cu._cartesian_cimpl(in_arrays, res, in_dim)

    res = res.swapaxes(axis, 0)

    if op is not None:
        res = op(res, axis=axis)

    return res

if __name__ == '__main__':
    arr1, arr2, arr3, arr4 = np.arange(10), np.arange(20), np.arange(30), \
        np.arange(40)
    for i in range(1000):
        cartesian_op((arr1, arr2, arr3, arr4))
