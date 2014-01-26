from __future__ import division, absolute_import, print_function

import numpy as np
from numpy import searchsorted, interp


def interp_grid_prob(vals, grid_vals):
    inext_low = searchsorted(grid_vals, vals)
    max_idx = grid_vals.shape[0] - 1
    fp = np.arange(vals.shape[0])
    pnext_low = interp(vals, grid_vals, fp) - fp

    assert np.all(pnext_low >= 0) and np.all(pnext_low <= 1)

    inext_high = np.where(inext_low < max_idx, inext_low + 1, inext_low)
    pnext_high = 1 - pnext_low

    return inext_low, inext_high, pnext_low, pnext_high


def makegrid(x, y, axis=1):

    x = np.matrix(x)
    y = np.matrix(y)

    how = x.shape
    how[1-axis] = 1

    yy = np.tile(y, how)
    xx = x.repeat(y.shape[axis], axis=axis)

    res = np.vstack((xx, yy))
    return res