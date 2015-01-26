__author__ = 'Richard Foltyn'

import numpy as np
import pytest

from pydynopt.interpolate import interp1d_linear, interp2d_bilinear

import common


class TestBilinear(common.TestBase):

    @pytest.fixture
    def f_interp(self):
        return interp2d_bilinear

    @pytest.fixture
    def ndim(self):
        return 2

    @pytest.fixture(scope='module', params=range(3))
    def f_nonlinear(self, request):
        """
        Define some non-linear testing functions with 2D domain.
        :param request: py.test request object parametrizing fixture
        :return: ith non-linear function
        """
        if request.param == 0:
            return lambda u, v: np.exp(10 + u/10) + np.sin(v)
        elif request.param == 1:
            return lambda u, v: np.log(np.abs(100*u)) + np.power(v, 2)
        else:
            return lambda u, v: np.power(np.abs(u), 1/2) + abs(v)*np.sin(v)

    def test_margins_1d(self, data, f_interp, ndim):
        """
        Verify that interpolation and extrapolation for linear 'marginal'
        functions works exactly.

        Testing strategy: iterate or all 'marginal' 1D linear functions 'fm',
        creating a sequence of ndim-dimensional wrapper function which only
        pass dimension (0,), (1,) or (2,) to underlying 1D function. Verify
        that ndim-dimensional interpolation of wrapper function coincides
        with 1D interpolation of 'marginal' function.
        """

        xp, x = data
        x_ext = common.extend(x)

        for fm in common.func_generator(1):
            for j in range(ndim):
                f = lambda *u: fm(u[j])

                # test against np.interp, without extrapolation
                common.test_margin(f_interp, f, xp, x,
                                   marg=j, f_interp_marg=np.interp, f_marg=fm)

                # test 1d linear extrapolation too, using our own interp1d
                common.test_margin(f_interp, f, xp, x_ext,
                                   marg=j, f_interp_marg=interp1d_linear,
                                   f_marg=fm)
