__author__ = 'Richard Foltyn'

import numpy as np
import pytest

from pydynopt.interpolate import interp3d_trilinear, interp1d_linear, \
    interp2d_bilinear

import pydynopt.unittests.interpolate.common as common

NDIM = 3


@pytest.fixture(scope='module', params=[2, 11, 100])
def data(request):
    xp = common.initiliaze_xp(NDIM, request.param)
    x = common.initialize_x(xp)

    return xp, x

@pytest.fixture(scope='module')
def f_interp():
    return interp3d_trilinear


def test_dimensions(data, f_interp):
    """
    Test whether array dimensions of return array is conformable, and whether
    exceptions are raised if input dimensions are non-conformable.
    """
    xp, _ = data

    f = lambda u, v, w: np.zeros_like(u)
    common.test_dimensions(f_interp, f, xp)


def test_identity(data, f_interp):
    """
    Test whether interpolating exactly at interpolation nodes gives
    correct result.
    """

    xp, x = data

    f = lambda x, y, z: np.exp(x/10) + np.cos(y) + np.log(np.abs(z))
    common.test_equality(f_interp, f, xp, xp)


def test_constant(data, f_interp):
    xp, x = data

    const_val = np.asscalar(np.random.randn(1))

    f = lambda u, v, w: np.ones_like(v) * const_val
    common.test_equality(f_interp, f, xp, x)


def test_margins_1d(data, f_interp):
    """
    Verify that interpolation and extrapolation for linear 'marginal'
    functions works exactly.
    """

    coefs = np.random.randn(2)
    fm = lambda v: coefs[0] * v + coefs[1]

    xp, x = data
    # extend beyond interval defined by xp to text extrapolation
    x_ext = common.extend(x)

    for i in range(NDIM):
        f = lambda *x: fm(x[i])
        # test against np.interp, without extrapolation
        common.test_margin(f_interp, f, xp, x,
                           marg=i, f_interp_marg=np.interp, f_marg=fm)

        # test 1d linear extrapolation too, using our own interp1d
        common.test_margin(f_interp, f, xp, x_ext,
                           marg=i, f_interp_marg=interp1d_linear, f_marg=fm)


def test_margins_2d(data, f_interp):
    """
    Verify that interpolation and extrapolation for linear 'marginal'
    functions works exactly.
    """

    # Test linear function
    coefs = np.random.randn(3)
    fm = lambda v, w: coefs[0] * v + coefs[1] * w + coefs[2]

    xp, x = data
    x_ext = common.extend(x)

    for i in range(NDIM):
        m1, m2 = common.get_margins(NDIM, i)
        f = lambda *x: fm(x[m1], x[m2])
        # test against interp2d_bilinear for both inter- and extrapolation
        common.test_margin(f_interp, f, xp, x_ext,
                           marg=(m1, m2), f_marg=fm,
                           f_interp_marg=interp2d_bilinear, tol=1e-9)

    # Bilinear interpolation of a marginal function of the form
    # f(u, v) = a * u + b * v + c * u * v + d
    # should work exactly too.

    coefs = np.random.randn(4)
    fm = lambda v, w: coefs[0] * v + coefs[1] * w + coefs[2]*w*v + coefs[3]

    for i in range(NDIM):
        m1, m2 = common.get_margins(NDIM, i)
        f = lambda *x: fm(x[m1], x[m2])
        # test against interp2d_bilinear for both inter- and extrapolation
        common.test_margin(f_interp, f, xp, x_ext,
                           marg=(m1, m2), f_marg=fm,
                           f_interp_marg=interp2d_bilinear)


def test_extrapolate(data, f_interp):
    """
    Verify that linear 3D functions are extrapolated exactly.
    """
    xp, x = data
    x_ext = common.extend(x)

    for i in range(10):
        c = np.random.randn(4)
        f = lambda u, v, w: c[0] * u + c[1] * v + c[2] * w + c[3]

        common.test_equality(f_interp, f, xp, x_ext, tol=1e-9)


def test_interpolate(data, f_interp):
    """
    Check whether functions of the form f(u,v,w) = ... with all interactions
    are intrapolated exactly.
    """

    xp, x = data

    for i in range(10):
        c = np.random.randn(7)
        f = lambda u, v, w: c[0] * u + c[1] * v + c[2] * w + \
                c[3]*u*v + c[4]*u*w + c[4]*v*w + c[5]*u*v*w + c[6]

        common.test_equality(f_interp, f, xp, x)