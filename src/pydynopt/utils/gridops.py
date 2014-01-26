from __future__ import division, absolute_import, print_function

import numpy as np
from numpy import searchsorted, interp


def interp_grid_prob(vals, grid_vals):
    vals = np.atleast_1d(vals)
    inext_low = searchsorted(grid_vals, vals, side='right') - 1
    max_idx = grid_vals.shape[0] - 1
    fp = np.arange(grid_vals.shape[0])
    pnext_low = interp(vals, grid_vals, fp) - fp[inext_low]

    assert np.all(pnext_low >= 0) and np.all(pnext_low <= 1)

    inext_high = np.where(inext_low < max_idx, inext_low + 1, inext_low)
    pnext_high = 1 - pnext_low

    return inext_low, inext_high, pnext_low, pnext_high


def makegrid(a_tup, axis=0, op=None):

    assert axis <= 1

    lengths = []
    in_arrays = []
    out_arrays = []
    for (i, v) in enumerate(a_tup):
        in_arrays.append(np.atleast_2d(v))
        lengths.append(in_arrays[-1].shape[1-axis])

    for (i, v) in enumerate(in_arrays):
        tile_by = [1, 1]
        tile_by[1-axis] = np.prod(lengths[:i])
        rep_by = np.prod(lengths[i+1:])

        vv = v.repeat(rep_by, axis=(1-axis))
        vv = np.tile(vv, tuple(tile_by))
        out_arrays.append(vv)

    res = np.concatenate(tuple(out_arrays), axis=axis)

    if op is not None:
        res = op(res, axis=axis)

    return res