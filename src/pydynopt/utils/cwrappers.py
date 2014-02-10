from __future__ import division, print_function, absolute_import
import numpy as np

import pydynopt.utils.cutils as cu


def cartesian_op(a_tup, axis=0, op=None, dtype=None):
    if dtype == np.int64 or dtype == np.int:
        res = cu._c_cartesian_int64(a_tup, axis)

    if op is not None:
        res = op(res, axis=axis)

    return res