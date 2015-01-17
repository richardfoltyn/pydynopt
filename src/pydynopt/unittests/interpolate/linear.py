__author__ = 'Richard Foltyn'


import unittest2 as ut
import numpy as np

from pydynopt.interpolate import interp1d_linear


class BilinearTest(ut.TestCase):

    def setUp(self):
        self.xp = np.sort(np.random.randn(10))

        self.x0 = np.linspace(np.min(self.xp), np.max(self.xp), 10)

        # define some linear functions
        self.flin1 = lambda x: 2.23 * x - 123
        self.flin2 = lambda x: -234.5 * x + 123.123
        self.flin3 = lambda x: np.ones_like(x, dtype=np.float) * 100.0

        # define some non-linear functions
        self.f1 = lambda x: np.sin(x) - np.cos(x) / 2
        self.f2 = lambda x: np.exp(x) - np.power(x, 2)
        self.f3 = lambda x: np.log(np.abs(x))

    def test_constant(self):

        z = np.zeros_like(self.xp)

        zhat = interp1d_linear(self.x0, self.xp, z)

        self.assertTrue(np.max(np.abs(zhat - 0)) <= 1e-9)

    def test_extrapolate(self):
        """
        Verify that extrapolation for linear functions works exactly.
        """

        x0 = np.hstack((-10 * self.x0, self.x0, 10 * self.x0))
        for f in (self.flin1, self.flin2, self.flin3):
            fp = f(self.xp)
            fx = f(x0)

            fx_hat = interp1d_linear(x0, self.xp, fp)

            self.assertTrue(np.max(np.abs((fx_hat - fx)/fx)) < 1e-9)

    def test_identity(self):
        """
        Check whether interpolating exactly at the nodes returns function
        value at nodes passed into interpolator.
        """

        for f in (self.flin1, self.flin2, self.flin3):
            fp = f(self.xp)

            fx_hat = interp1d_linear(self.xp, self.xp, fp)

            self.assertTrue(np.max(np.abs((fx_hat - fp) / fp)) < 1e-12)

    def test_numpy(self):
        """
        Check whether interpolated results coincide with np.interp results
        for some linear and non-linear functions.
        """
        # Note: np.interp does not support extrapolation!
        for f in (self.flin1, self.flin2, self.flin3, self.f1, self.f2,
                  self.f3):

            fp = f(self.xp)
            fx_np = np.interp(self.x0, self.xp, fp)
            fx_hat = interp1d_linear(self.x0, self.xp, fp)

            self.assertTrue(np.max(np.abs((fx_hat-fx_np))) < 1e-12)

