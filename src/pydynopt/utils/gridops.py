from __future__ import division, absolute_import, print_function

import numpy as np
from numpy import searchsorted, interp
from math import log

from scipy.optimize import root


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


def makegrid(start, stop, num, logs=True, insert_vals=None, log_shift=0,
             x0=None, num_at_x0=None):
    if insert_vals:
        insert_vals = np.atleast_1d(insert_vals)

    if logs:
        if num_at_x0 is not None and x0 is not None:
            frac0 = num_at_x0 / num

            f = lambda x: log(x0 + x) - log(start + x) - \
                          frac0 * (log(stop + x) - log(start + x))

            sol = root(f, 0 - start + 1, tol=1e-8)
            assert sol.success
            log_shift = sol.x

        lstart, lstop = log(start + log_shift), log(stop + log_shift)
    else:
        log_shift = 0
        lstart, lstop = start, stop

    rem = 0 if insert_vals is None else len(insert_vals)

    grid = np.linspace(lstart, lstop, num - rem)
    if logs:
        grid = np.exp(grid) - log_shift

    if insert_vals is not None and len(insert_vals) > 0:
        idx_insert = np.searchsorted(grid, insert_vals) + 1
        grid = np.insert(grid, idx_insert, insert_vals)

    # there may be some precision issues resulting in
    # x != exp(log(x + log_shift) - log_shift
    # so we replace the start and stop values with the requested values
    grid[0] = start
    grid[-1] = stop

    return grid


