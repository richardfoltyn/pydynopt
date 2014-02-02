from __future__ import division, absolute_import, print_function

import numpy as np
from numpy import searchsorted, interp
from math import log

from scipy.optimize import root

# from .cutils import _interp_grid_prob


def interp_grid_prob(vals, grid_vals):
    v = np.asarray(vals).flatten()
    g = np.asarray(grid_vals).flatten()
    i_low = np.maximum(searchsorted(g, v, side='right') - 1, 0)
    max_idx = len(g) - 1
    fp = np.arange(len(g))
    p_high = interp(v, g, fp) - fp[i_low]

    assert np.all(p_high >= 0) and np.all(p_high <= 1)

    i_high = np.where(i_low < max_idx, i_low + 1, i_low)
    p_low = 1 - p_high
    vs = vals.shape
    return i_low.reshape(vs), i_high.reshape(vs), p_low.reshape(vs), \
           p_high.reshape(vs)


# def interp_grid_prob2(vals, grid_vals):
#     v = np.asarray(vals).flatten()
#     g = np.asarray(grid_vals).flatten()
#     i_low, i_high, p_low, p_high = _interp_grid_prob(v, g)
#     vs = vals.shape
#     return i_low.reshape(vs), i_high.reshape(vs), p_low.reshape(vs), p_high.reshape(vs)


def cartesian_op(a_tup, axis=0, op=None):
    assert axis <= 1

    lengths = []
    in_arrays = []
    out_arrays = []
    for (i, v) in enumerate(a_tup):
        in_arrays.append(np.atleast_2d(v))
        lengths.append(in_arrays[-1].shape[1 - axis])

    for (i, v) in enumerate(in_arrays):
        tile_by = [1, 1]
        tile_by[1 - axis] = np.prod(lengths[:i])
        rep_by = np.prod(lengths[i + 1:])

        vv = v.repeat(rep_by, axis=(1 - axis))
        vv = np.tile(vv, tuple(tile_by))
        out_arrays.append(vv)

    res = np.concatenate(tuple(out_arrays), axis=axis)

    if op is not None:
        res = np.atleast_2d(op(res, axis=axis)).swapaxes(0, axis)

    return res


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


def makegrid_mirrored(start, stop, num, around, endpoint=True,
                      logs=False, log_shift=1):
    assert start <= around <= stop

    adj_stop = stop - around
    adj_start = abs(start - around)
    adj_around = 0
    adj_num = num if not endpoint else num - 1

    f = lambda x: np.log(x + log_shift) if logs else lambda y: y
    finv = lambda x: np.exp(x) - log_shift if logs else lambda y: y

    frac = min(f(adj_start), f(adj_stop)) / (f(adj_stop) + f(adj_start))
    t_num = np.ceil(frac * adj_num)
    frm, to = f(adj_around), min(f(adj_start), f(adj_stop))
    step = np.linspace(frm, to, t_num, retstep=True)
    grid_short = finv(np.linspace(frm, to, t_num))
    to = frm + step * (adj_num - t_num + 1) * 1.5
    grid_long = finv(np.arange(frm, to, step)[1:adj_num - t_num])

    if endpoint:
        grid_long = np.append(grid_long, stop)

    sgn_long = np.sign(adj_stop - adj_start)

    grid_low = (- sgn_long) * grid_short + around
    grid_high = sgn_long * grid_long + around
    grid = np.hstack(grid_low, grid_high)

    assert np.all(start <= grid) and np.all(grid <= stop)
    assert len(grid) == num

    return grid
