"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import numpy as np


def cartesian_op(a_tup, axis=0, op=None, dtype=None):
    # assert axis <= 1

    na = len(a_tup)

    dim = np.empty((na, 2), dtype=np.uint32)
    in_arrays = np.empty((na, ), dtype=object)

    for (i, v) in enumerate(a_tup):
        in_arrays[i] = np.atleast_2d(v).swapaxes(axis, 0)
        dim[i] = in_arrays[i].shape

    cumc = np.cumprod(dim[:, 1], axis=0)
    crd = np.zeros((dim.shape[0] + 1, ), dtype=np.uint32)
    crd[1:] = np.cumsum(dim[:, 0])
    reps = np.ones_like(cumc, dtype=np.uint32)
    tiles = np.ones_like(reps)
    reps = cumc[-1] / cumc
    tiles[1:] = cumc[:-1]

    if dtype is None:
        dtype = a_tup[0].dtype
    out = np.empty((crd[-1], cumc[-1]), dtype=dtype)

    for (i, v) in enumerate(in_arrays):
        out[crd[i]:crd[i+1]] = \
            np.tile(v.repeat(reps[i], axis=1), (1, tiles[i]))

    out = out.swapaxes(axis, 0)

    if op is not None:
        out = op(out, axis=axis)

    return out


