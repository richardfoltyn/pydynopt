from __future__ import division, absolute_import, print_function

import numpy as np
from numpy import searchsorted, interp
from math import log

from scipy.optimize import root

import unittest2 as ut


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
        res = op(res, axis=axis)

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

        start = log(start + log_shift)
        stop = log(stop + log_shift)
    else:
        log_shift = 0

    rem = 0 if insert_vals is None else len(insert_vals)

    grid = np.linspace(start, stop, num - rem)
    if logs:
        grid = np.exp(grid) - log_shift

    if insert_vals is not None and len(insert_vals) > 0:
        idx_insert = np.searchsorted(grid, insert_vals) + 1
        grid = np.insert(grid, idx_insert, insert_vals)

    return grid


def makegrid_mirrored(start, stop, num, around, logs=False):
    assert start <= around <= stop

    start = np.min(start, around)
    frac_above = (stop - around) / (stop - start)
    tmp_n = np.ceil(max(frac_above, 1 - frac_above) * num)
    grid_above = makegrid(around, max(stop, abs(start)), tmp_n, logs=logs,
                          log_shift=1)
    grid_below = -1 * grid_above[-1:0:-1]

    grid = np.hstack((grid_below[start <= grid_below],
                      grid_above[grid_above <= stop]))

    return grid


class _TestInterpGrid(ut.TestCase):

    def setUp(self):
        self.vals = np.random.rand(10)
        self.grid = np.array([0.0, 1.0])

    def test_interp_grid_prod(self):
        ilow, ihigh, plow, phigh = interp_grid_prob(self.vals, self.grid)

        self.assertTrue(ilow.shape == ihigh.shape == plow.shape == phigh.shape)
        self.assertTrue(ilow.shape == self.vals.shape)
        self.assertTrue(np.all(np.abs(plow + phigh - 1) < 1e-12))
        self.assertTrue(np.all(np.abs(self.vals - phigh) < 1e-12))
        self.assertTrue(np.all(ilow <= ihigh))

if __name__ == "__main__":
    ut.main()