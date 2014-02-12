from __future__ import division, absolute_import, print_function

import numpy as np
import unittest2 as ut

from pydynopt.utils import *


class TestInterpGrid(ut.TestCase):

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


class TestCartesianOp(ut.TestCase):
    def setUp(self):
        self.arr1 = np.arange(4)
        self.arr2 = np.arange(4, 7)
        self.arr3 = np.arange(7, 9)
        self.arr4 = np.arange(10).reshape((2, 5))
        self.arr5 = np.arange(10, 19).reshape((3, 3))

    def test_dimensions(self):

        out_len = np.prod((len(self.arr1), len(self.arr2),
                           len(self.arr3)))

        result = cartesian_op((self.arr1, self.arr2, self.arr3), axis=0)

        self.assertEqual(result.shape[0], 3, 'Output dimension test failed')
        self.assertEqual(result.shape[1], out_len, 'Output dimension test '
                                                   'failed')

        result = cartesian_op((self.arr1.reshape((-1, 1)),
                               self.arr2.reshape((-1, 1)),
                               self.arr3.reshape((-1, 1))), axis=1)

        self.assertEqual(result.shape[1], 3, 'Output dimension test failed')
        self.assertEqual(result.shape[0], out_len, 'Output dimension test '
                                                   'failed')

        result = cartesian_op((self.arr1, self.arr2, self.arr3), axis=0,
                              op=np.prod)
        self.assertEqual(result.shape[0], out_len, 'Output dimension test failed')

        result = cartesian_op((self.arr1.reshape((-1, 1)),
                               self.arr2.reshape((-1, 1)),
                               self.arr3.reshape((-1, 1))), axis=1,
                              op=np.prod)

        self.assertEqual(result.shape[0], out_len, 'Output dimension test failed')

        result = cartesian_op((self.arr4, self.arr5), axis=0)
        self.assertEqual(result.shape[0], self.arr4.shape[0] +
                                          self.arr5.shape[0])
        self.assertEqual(result.shape[1], self.arr4.shape[1] *
                                        self.arr5.shape[1])


class Test_makegrid_mirrored(ut.TestCase):
    def setUp(self):
        pass

    def test_simple(self):
        start, around, stop = 0, 1, 10
        num = 20

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        start, around, stop = -10, 0, 5
        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

    def test_log(self):
        start, around, stop = 0, 1, 10
        num = 20

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        start, around, stop = -15, 0, 5
        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        start, around, stop = 0, 10, 15
        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=True, log_shift=5)
        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

    def test_random(self):
        start, around, stop = np.sort(np.random.randn(3) * 10)
        num = 100

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=False)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

    def test_boundary(self):
        start, around, stop = 0, 0, 10
        num = 20

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        start, around, stop = -10, 0, 0

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True)
        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=True)

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

        # Test log-shifted transformation with log_shift > 1 and one-sided
        # grid
        start, around, stop = 0, 0, 10
        lshift = 5
        grid, idx0 = makegrid_mirrored(start, stop, num, around,
                                       retaround=True, logs=True,
                                       log_shift=lshift)

        grid2 = np.exp(np.linspace(np.log(start + lshift),
                                   np.log(stop+lshift), num=num)) -lshift
        # adjust start / stop values as makegrid_mirrored does
        grid2[0], grid2[-1] = start, stop
        self.assertTrue(np.all(grid == grid2))

        self._basic_checks(start, stop, around, num, grid, idx0)
        self._compare_elements(around, grid, idx0)

    def _basic_checks(self, start, stop, around, num, grid, idx0):
        self.assertEqual(grid[idx0], around)
        self.assertEqual(num, len(grid))
        self.assertEqual(start, grid[0])
        self.assertEqual(stop, grid[-1])
        self.assertTrue(np.all(start <= grid))
        self.assertTrue(np.all(grid <= stop))
        # make sure the sequence of grid points is non-decreasing
        self.assertTrue(np.all(grid[1:] >= grid[:-1]))

    def _compare_elements(self, around, grid, idx0, delta=1e-8):
        num = len(grid)
        for i in range(num):
            if idx0 - i < 1 or idx0 + i >= num - 1:
                break
            self.assertAlmostEqual(grid[idx0 + i] - around,
                             around - grid[idx0 - i], delta=delta)