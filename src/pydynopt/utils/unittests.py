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
