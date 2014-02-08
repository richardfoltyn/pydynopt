from __future__ import division, absolute_import, print_function

import numpy as np
from numpy import searchsorted, interp
from math import log

from scipy.optimize import root

# from .cutils import _interp_grid_prob


def interp_grid_prob(vals, grid_vals):
    v = np.reshape(vals, (-1,))
    g = np.reshape(grid_vals, (-1,))
    i_low = np.fmax(searchsorted(g, v, side='right') - 1, 0)
    max_idx = len(g) - 1
    fp = np.arange(max_idx + 1)
    p_high = interp(v, g, fp) - fp[i_low]

    # assert np.all(p_high >= 0) and np.all(p_high <= 1)

    i_high = np.fmin(i_low + 1, max_idx)
    p_low = 1 - p_high
    vs = vals.shape
    return i_low.reshape(vs), i_high.reshape(vs), p_low.reshape(vs), \
           p_high.reshape(vs)


# def interp_grid_prob2(vals, grid_vals):
#     v = np.reshape(vals, (-1,))
#     g = np.reshape(grid_vals, (-1,))
#     i_low = np.fmax(searchsorted(g, v, side='right') - 1, 0)
#     max_idx = len(g) - 1
#     i_high = np.fmin(i_low + 1, max_idx)
#
#     gl = g[i_low]
#     p_high = np.empty_like(i_low, dtype=float)
#     p_high[i_high != i_low] = (v - gl) / (g[i_high] - gl)
#     p_high[i_high == i_low] = 0
#
#     # assert np.all(p_high >= 0) and np.all(p_high <= 1)
#     p_low = 1 - p_high
#     vs = vals.shape
#     return i_low.reshape(vs), i_high.reshape(vs), p_low.reshape(vs), p_high.reshape(vs)


# def cartesian_op(a_tup, axis=0, op=None):
#     assert axis <= 1
#
#     lengths = []
#     in_arrays = []
#     out_arrays = []
#     for (i, v) in enumerate(a_tup):
#         in_arrays.append(np.atleast_2d(v))
#         lengths.append(in_arrays[-1].shape[1 - axis])
#
#     for (i, v) in enumerate(in_arrays):
#         tile_by = [1, 1]
#         tile_by[1 - axis] = np.prod(lengths[:i])
#         rep_by = np.prod(lengths[i + 1:])
#
#         vv = v.repeat(rep_by, axis=(1 - axis))
#         vv = np.tile(vv, tuple(tile_by))
#         out_arrays.append(vv)
#
#     res = np.concatenate(tuple(out_arrays), axis=axis)
#
#     if op is not None:
#         res = np.atleast_2d(op(res, axis=axis)).swapaxes(0, axis)
#
#     return res


def cartesian_op(a_tup, axis=0, op=None, dtype=None):
    # assert axis <= 1

    na = len(a_tup)

    dim = np.empty((na, 2), dtype=np.uint32)
    dtypes = np.empty((na, ), dtype=object)
    in_arrays = np.empty((na, ), dtype=object)

    for (i, v) in enumerate(a_tup):
        in_arrays[i] = np.atleast_2d(v).swapaxes(axis, 0)
        dim[i] = in_arrays[i].shape
        dtypes[i] = in_arrays[i].dtype

    cumc = np.cumprod(dim[:, 1], axis=0)
    crd = np.zeros((dim.shape[0] + 1, ), dtype=np.uint32)
    crd[1:] = np.cumsum(dim[:, 0])
    reps = np.ones_like(cumc, dtype=np.uint32)
    tiles = np.ones_like(reps)
    reps = cumc[-1] / cumc
    tiles[1:] = cumc[:-1]

    if dtype is None:
        dtype = np.find_common_type(dtypes, [])
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
