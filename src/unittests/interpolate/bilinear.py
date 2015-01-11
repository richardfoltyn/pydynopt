__author__ = 'Richard Foltyn'


import unittest2 as ut
import numpy as np

from scipy.interpolate import interp2d, bisplev, bisplrep

from pydynopt.interpolate import interp_bilinear




# Utility function used to compute cartesian product
def cartesian(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    b1 = np.tile(b, (1, a.shape[1]))
    a1 = np.repeat(a, b.shape[1], axis=1)

    return np.vstack((a1, b1))


def get_z(f, x, y):
    return np.array([[f(u, v) for v in y] for u in x])


class BilinearTest(ut.TestCase):

    def setUp(self):
        self.x = np.sort(np.random.randn(5))
        self.y = np.sort(np.random.randn(5))

        self.x0 = np.linspace(np.min(self.x), np.max(self.x), 10)
        self.y0 = np.linspace(np.min(self.y), np.max(self.y), 10)

        # define some 2D linear functions
        self.flin2d1 = lambda x, y: 2.23 * x - 1.45 * y - 123
        self.flin2d2 = lambda x, y: -234.5 * x - 1239 * y + 123.123
        self.flin2d3 = lambda x, y: 12312.123 * x - 1e8 * y - 1e9

        def func1(a, b):
            return np.exp(a) + np.power(b, 2)

        self.f1 = func1

        z = [[func1(u, v) for v in self.y] for u in self.x]
        self.z = np.asarray(z)

    def test_extrapolate_linear_1d(self):
        """
        Verify that extrapolation for linear 'marginal' functions works exactly.
        """

        f = lambda u, v: u
        x = np.arange(10, dtype=np.float)
        z = get_z(f, x,  np.zeros_like(x))

        x0 = np.arange(-5, 15, dtype=np.float)
        f0 = interp_bilinear(x0, x0, x, x, z)

        self.assertTrue(np.max(np.abs(x0 - f0)) < 1e-9)

        # repeat in y direction
        f = lambda u, v: v
        z = get_z(f, np.zeros_like(x), x)
        f0 = interp_bilinear(x0, x0, x, x, z)

        self.assertTrue(np.max(np.abs(f0 - x0)) < 1e-9)

    def test_extrapolate_linear_2d(self):
        """
        Verify that extrapolation for linear 2D functions works *exactly*
        """

        x0 = np.hstack((-10 * self.x0, self.x0, 10 * self.x0))
        y0 = np.hstack((-10 * self.y0, self.y0, 10 * self.y0))

        for f in (self.flin2d1, self.flin2d2, self.flin2d3):
            z = get_z(f, self.x, self.y)

            z0_true = np.diag(get_z(f, x0, y0))
            z0_hat = interp_bilinear(x0, y0, self.x, self.y, z)

            rerr = np.max(np.abs((z0_hat - z0_true)/z0_true))
            self.assertTrue(rerr < 1e-10)

    def test_1d(self):
        """
        Check whether interpolation for 'margins' returns same result as
        numpy implementation.
        """

        f = lambda x, y: x
        z = get_z(f, self.x, self.y)

        y0 = np.zeros_like(self.x0)

        # Note: np.interp does not extrapolate, so compare only inside domain!
        zhat_np = np.interp(self.x0, self.x, z[:, 0])
        zhat = interp_bilinear(self.x0, y0, self.x, self.y, z)

        self.assertTrue(np.max(np.abs(zhat-zhat_np)) < 1e-9)

        # repeat for y direction too
        f = lambda x, y: y
        z = get_z(f, self.x, self.y)
        x0 = np.zeros_like(self.y0)

        zhat_np = np.interp(self.y0, self.y, z[0])
        zhat = interp_bilinear(x0, self.y0, self.x, self.y, z)

        self.assertTrue(np.max(np.abs(zhat-zhat_np)) < 1e-9)

    def test_linear_2d(self):
        """
        Check whether linear 2D function is corretly interpolated.
        """

        f = lambda x, y: 2*x + 4*y - 5
        z = get_z(f, self.x, self.y)

        xy = cartesian(self.x0, self.y0)
        ztrue = np.array([f(u, v) for u, v in xy.T])
        zhat = interp_bilinear(xy[0], xy[1], self.x, self.y, z)

        self.assertTrue(np.max(np.abs(zhat - ztrue)) < 1e-9)

    def test_quadratic_2d(self):
        """
        Check whether quadratic 2D function is correctly interpolated.
        """

        f = lambda x, y: 2*x + 4*y - 5 + x * y / 2
        z = get_z(f, self.x, self.y)

        xy = cartesian(self.x0, self.y0)
        ztrue = np.array([f(u, v) for u, v in xy.T])
        zhat = interp_bilinear(xy[0], xy[1], self.x, self.y, z)

        self.assertTrue(np.max(np.abs(zhat - ztrue)) < 1e-9)


    def test_identity(self):
        """
        Check whether interpolating exactly at the nodes returns function
        value at nodes passed into interpolator.
        """

        xy0 = cartesian(self.x, self.y)

        zinterp = interp_bilinear(xy0[0], xy0[1], self.x, self.y,
                                  self.z)

        zinterp = zinterp.reshape((self.x.shape[0], -1))

        self.assertTrue(np.max(np.abs(zinterp - self.z)) < 1e-9)
