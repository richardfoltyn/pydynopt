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

    @pytest.fixture(params=((2,), (11,), (100,)))
    def data_shape(self, request):
        return request.param

    @pytest.fixture(scope='module', params=range(3))
    def f_nonlinear(self, request):
        if request.param == 0:
            return lambda u: np.exp(10 + u/10)
        elif request.param == 1:
            return lambda u: np.log(np.abs(100*u))
        else:
            return lambda u: np.power(np.abs(u), 1/2) * np.sin(u)

