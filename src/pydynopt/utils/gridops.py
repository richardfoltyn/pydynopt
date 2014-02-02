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
                      logs=False, log_shift=1, retaround=False):

    assert start <= around <= stop and start < stop

    # adjust grid just that around = 0
    adj_start, adj_stop = abs(start - around), stop - around
    adj_around = 0
    # check whether we actually need to manually include endpoint of
    # 'shorter' arm of the grid. This is not the case if the endpoint
    # coincides with 'around' and is thus on the grid in any case.
    need_endpoint = (adj_stop > 0 and abs(adj_start) > 0) and endpoint

    adj_num = num if not need_endpoint else num - 1

    if logs:
        f = lambda x: np.log(x + log_shift)
        finv = lambda x: np.exp(x) - log_shift
    else:
        f = finv = lambda x: x

    # boundaries of long segment
    frm, to = f(adj_around), max(f(adj_start), f(adj_stop))

    # compute fraction of grid elements that will be located on the
    # 'long' segment
    if adj_start == 0 or adj_stop == 0:
        frac = 1
    else:
        frac = (to - frm) / (f(adj_stop) + f(adj_start) - 2 * frm)

    # number of elements on the 'long' segment of the grid
    # make sure there is space left in case we need to add an endpoint value!
    num_long = min(np.ceil(frac * num), adj_num)

    grid_long, step = np.linspace(frm, to, num_long, retstep=True)
    grid_long = finv(grid_long)
    grid_short = grid_long[1:adj_num - num_long + 1]

    if need_endpoint:
        grid_short = np.append(grid_short, min(adj_start, adj_stop))

    sgn_long = np.sign(adj_stop - adj_start)

    # grid_short = (- sgn_long) * grid_short + around
    # grid_long = sgn_long * grid_long + around
    if sgn_long > 0:
        grid = np.hstack((-grid_short[::-1], grid_long)) + around
        # ensure that endpoint of the longer side of the grid has exactly the
        # requested value. This may not be the case due to lack of precision,
        # in particular when log transformation is applied.
        grid[-1] = stop
        if need_endpoint or adj_start == 0:
            grid[0] = start
        around_idx = len(grid_short)
    else:
        grid = np.hstack((- grid_long[::-1], grid_short)) + around
        grid[0] = start
        if need_endpoint or adj_stop == 0:
            grid[-1] = stop
        around_idx = len(grid_long) - 1

    # TODO: eventually remove these assertions as they are covered in unit
    # test for some specific scenarios
    assert np.all(start <= grid) and np.all(grid <= stop)

    if retaround:
        return grid, around_idx
    else:
        return grid
