__author__ = 'Richard Foltyn'


import numpy as np

from pydynopt.interpolate import interp3d_trilinear, interp1d_linear, \
    interp2d_bilinear

from pydynopt.unittests.interpolate.common import AbstractTest, extend

# create arrays of knots this long
xp_len = 5
# Length of arrays of points where to interpolate
x_len = 10


class TrilinearTest(AbstractTest):

    def setUp(self):
        self.xp1 = np.sort(np.random.randn(xp_len) * 10)
        self.xp2 = np.sort(np.random.randn(xp_len) * 10)
        self.xp3 = np.sort(np.random.randn(xp_len) * 10)

        self.xp = (self.xp1, self.xp2, self.xp3)

        self.x1 = np.linspace(np.min(self.xp1), np.max(self.xp1), x_len)
        self.x2 = np.linspace(np.min(self.xp2), np.max(self.xp2), x_len)
        self.x3 = np.linspace(np.min(self.xp3), np.max(self.xp3), x_len)

        self.x = (self.x1, self.x2, self.x3)

    def test_dimensions(self):
        f = lambda x, y, z: np.zeros_like(x)
        self._test_dimensions(interp3d_trilinear, f, self.xp)

    def test_identity(self):
        """
        Test whether interpolating exactly at interpolation nodes gives
        correct result.
        """

        f = lambda x, y, z: np.exp(x/10) + np.cos(y) + np.log(np.abs(z))

        self._test_equality(interp3d_trilinear, f, self.xp, self.xp)

    def test_constant(self):

        const_val = np.asscalar(np.random.randn(1))
        f = lambda x, y, z: np.ones_like(x) * const_val
        
        self._test_equality(interp3d_trilinear, f, self.xp, self.x)

    def test_margins_1d(self):
        """
        Verify that interpolation and extrapolation for linear 'marginal'
        functions works exactly.
        """

        coefs = np.random.randn(2)

        fm = lambda v: coefs[0] * v + coefs[1]

        x1 = extend(self.x1)
        x2 = extend(self.x2)
        x3 = extend(self.x3)

        x = (x1, x2, x3)

        assert x1.shape == x2.shape == x3.shape

        ndim = len(self.xp)
        for i in range(ndim):
            f = lambda *x: fm(x[i])
            # test against np.interp, without extrapolation
            self._test_margin(interp3d_trilinear, f, self.xp, self.x,
                              marg=i, f_interp_marg=np.interp, f_marg=fm)

            # test 1d linear extrapolation too, using our own interp1d
            self._test_margin(interp3d_trilinear, f, self.xp, x,
                              marg=i, f_interp_marg=interp1d_linear, f_marg=fm)

    def test_margins_2d(self):
        """
        Verify that interpolation and extrapolation for linear 'marginal'
        functions works exactly.
        """

        # Test linear function
        coefs = np.random.randn(3)
        fm = lambda v, w: coefs[0] * v + coefs[1] * w + coefs[2]

        x = extend(self.x)

        ndim = len(self.xp)

        for i in range(ndim):
            m1, m2 = np.sort(tuple(set(range(ndim)) - set((i, ))))
            f = lambda *x: fm(x[m1], x[m2])
            # test against interp2d_bilinear for both inter- and extrapolation
            self._test_margin(interp3d_trilinear, f, self.xp, x,
                              marg=(m1, m2), f_marg=fm,
                              f_interp_marg=interp2d_bilinear)

        # Bilinear interpolation of a marginal function of the form
        # f(u, v) = a * u + b * v + c * u * v + d
        # should work exactly too.

        coefs = np.random.randn(4)
        fm = lambda v, w: coefs[0] * v + coefs[1] * w + coefs[2]*w*v + coefs[3]

        for i in range(ndim):
            m1, m2 = np.sort(tuple(set(range(ndim)) - set((i, ))))
            f = lambda *x: fm(x[m1], x[m2])
            # test against interp2d_bilinear for both inter- and extrapolation
            self._test_margin(interp3d_trilinear, f, self.xp, x,
                              marg=(m1, m2), f_marg=fm,
                              f_interp_marg=interp2d_bilinear)

    def test_extrapolate(self):
        """
        Verify that linear 3D functions are extrapolated exactly.
        """
        x = extend(self.x)

        for i in range(10):
            c = np.random.randn(4)
            f = lambda u, v, w: c[0] * u + c[1] * v + c[2] * w + c[3]

            self._test_equality(interp3d_trilinear, f, self.xp, x)

    def test_interpolate(self):
        """
        Check whether functions of the form f(u,v,w) = ... with all interactions
        are intrapolated exactly.
        """

        for i in range(10):
            c = np.random.randn(7)
            f = lambda u, v, w: c[0] * u + c[1] * v + c[2] * w + \
                    c[3]*u*v + c[4]*u*w + c[4]*v*w + c[5]*u*v*w + c[6]

            self._test_equality(interp3d_trilinear, f, self.xp, self.x)