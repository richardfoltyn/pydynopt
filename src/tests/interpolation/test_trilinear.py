__author__ = 'Richard Foltyn'

import numpy as np
import pytest

from pydynopt.interpolate import interp3d_trilinear, interp1d_linear, \
    interp2d_bilinear

import common
import test_bilinear as bilinear


class TestTrilinear(bilinear.TestBilinear):

    @pytest.fixture
    def f_interp(self):
        return interp3d_trilinear

    @pytest.fixture
    def ndim(self):
        return 3

    @pytest.fixture(scope='module', params=range(3))
    def f_nonlinear(self, request):
        if request.param == 0:
            return lambda u, v, w: np.exp(10 + u/10) + np.sin(v)*np.cos(w)
        elif request.param == 1:
            return lambda u, v, w: np.log(np.abs(100*u)) + np.power(v, 2)
        else:
            return lambda u, v, w: np.power(np.abs(u), 1/2) + abs(v)*np.sin(w)

    def test_margins_2d(self, data, f_interp, ndim):
        """
        Verify that interpolation and extrapolation for linear 'marginal'
        functions works exactly.

        Testing strategy: iterate over all bilinear functions with 2D domain
        which can be interpolated using the bilinear interpolator. Write a
        wrapper function with 3D domain, passing only dimensions (1,2), (0,
        2) and (0, 1) to bilinear 2D function. Check that the interpolated
        values using trilinear and bilinear interpolation coincide.
        """

        xp, x = data
        x_ext = common.extend(x)

        for fm in common.func_generator(2):
            # Iterate over excluded margin
            for j in range(ndim):
                m1, m2 = common.get_margins(ndim, j)
                f = lambda *x: fm(x[m1], x[m2])
                # test against interp2d_bilinear for both inter- and
                # extrapolation
                common.test_margin(f_interp, f, xp, x_ext,
                                   marg=(m1, m2), f_marg=fm,
                                   f_interp_marg=interp2d_bilinear, tol=1e-9)


# class Request:
#     pass
# if __name__ == '__main__':
#
#     request = Request()
#     test = TestTrilinear()
#     ndim = test.ndim()
#
#     f_interp = test.f_interp()
#
#     for l in (2, 11, 101):
#         request.param = l
#         data = test.data(request, ndim)
#         test.test_margins_2d(data, f_interp, ndim)