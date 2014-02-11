from __future__ import division, print_function, absolute_import
import numpy as np

import pydynopt.utils.cutils as cu


def cartesian_op(a_tup, axis=0, op=None, dtype=None):

    na = len(a_tup)

    ncols, nrows = 1, 0
    dtypes = np.empty((na, ), dtype=object)
    in_arrays = np.empty((na, ), dtype=object)

    for (i, v) in enumerate(a_tup):
        in_arrays[i] = np.atleast_2d(v).swapaxes(axis, 0)
        ncols *= in_arrays[i].shape[1]
        nrows += in_arrays[i].shape[0]

    res = np.empty((nrows, ncols), dtype=dtype)

    cu._cartesian_cimpl(in_arrays[0], in_arrays[1], out=res, axis=axis)

    if op is not None:
        res = op(res, axis=axis)

    return res