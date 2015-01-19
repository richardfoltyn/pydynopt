__author__ = 'Richard Foltyn'


import numpy as np
import pytest

from pydynopt.interpolate import interp1d_linear

import common


class TestLinear(common.TestBase):

    @pytest.fixture
    def f_interp(self):
        return interp1d_linear

    @pytest.fixture
    def ndim(self):
        return 1

    @pytest.fixture(scope='module', params=range(3))
    def f_nonlinear(self, request):
        if request.param == 0:
            return lambda u: np.exp(10 + u/10)
        elif request.param == 1:
            return lambda u: np.log(np.abs(100*u))
        else:
            return lambda u: np.power(np.abs(u), 1/2) * np.sin(u)


# class Request:
#     pass
#
# if __name__ == '__main__':
#
#     request = Request()
#     test = TestLinear()
#     ndim = test.ndim()
#
#     f_interp = test.f_interp()
#
#     for l in (2, 11, 101):
#         request.param = l
#         data = test.data(request, ndim)
#         test.test_extrapolate(data, f_interp, ndim)