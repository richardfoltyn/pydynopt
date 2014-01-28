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
        self.first = np.arange(4)
        self.second = np.arange(4, 7)
        self.third = np.arange(7, 9)

    def test_dimensions(self):
        out_len = np.prod((len(self.first), len(self.second),
                           len(self.third)))

        result = cartesian_op((self.first, self.second, self.third), axis=0)

        self.assertEqual(result.shape[0], 3L, 'Output dimension test failed')
        self.assertEqual(result.shape[1], out_len, 'Output dimension test '
                                                   'failed')

        result = cartesian_op((self.first.reshape((-1, 1)),
                               self.second.reshape((-1, 1)),
                               self.third.reshape((-1, 1))), axis=1)

        self.assertEqual(result.shape[1], 3L, 'Output dimension test failed')
        self.assertEqual(result.shape[0], out_len, 'Output dimension test '
                                                   'failed')

        result = cartesian_op((self.first, self.second, self.third), axis=0,
                              op=np.prod)
        self.assertEqual(result.shape[0], 1L, 'Output dimension test failed')
        self.assertEqual(result.shape[1], out_len, 'Output dimension test failed')

        result = cartesian_op((self.first.reshape((-1, 1)),
                               self.second.reshape((-1, 1)),
                               self.third.reshape((-1, 1))), axis=1,
                              op=np.prod)

        self.assertEqual(result.shape[0], out_len, 'Output dimension test failed')
        self.assertEqual(result.shape[1], 1L, 'Output dimension test failed')
