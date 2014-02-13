from __future__ import division, print_function, absolute_import
import numpy as np

import pydynopt.utils.cutils as cu


def cartesian_op(a_tup, axis=0, op=None, dtype=None):

    in_arrays = [np.asmatrix(x) for x in a_tup]
    in_dim = np.array([x.shape for x in in_arrays], dtype=np.uint32, order='C')

    ncols, nrows = np.prod(in_dim[:, 1]), np.sum(in_dim[:, 0])

    if dtype is None:
        dtype = a_tup[0].dtype

    res = np.empty((nrows, ncols), dtype=dtype, order='C')

    if dtype == np.int64:
        cu._cartesian_cimpl_int64(in_arrays, res, in_dim)


    if op is not None:
        res = op(res, axis=axis)

    return res

if __name__ == '__main__':
    arr1, arr2, arr3, arr4 = np.arange(100), np.arange(100), np.arange(100), \
        np.arange(40)
    cartesian_op((arr1, arr2, arr3, arr4))
