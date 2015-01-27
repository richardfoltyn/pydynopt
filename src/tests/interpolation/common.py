__author__ = 'Richard Foltyn'

import numpy as np
from numpy import linspace

import pytest

from collections import Iterable
from itertools import combinations


def f_array(f, *xp):
    xp = ascontainer(xp)
    if len(xp) > 1:
        mgrid = np.meshgrid(*xp, indexing='ij')
        return f(*mgrid)
    else:
        return f(*xp)


def ascontainer(x, container=tuple):
    if isinstance(x, (list, tuple)):
        return container(x)
    else:
        return container((x,))


def func_generator(n, low=1, high=None, coefs=None):
    """
    Generate a set of functions that are linear in all possible interactions
    between the dimensions spanning the domain

    Specifically, assume that n=2; then func_generator returns an iterator
    over the functions

    f(x1,x2) = a1 * x1 + c
    f(x1,x2) = a1 * x2 + c
    f(x1,x2) = a1 * x1 + a2 * x2 + c
    f(x1,x2) = a1 * x1 + a2 * x2 + a3 * x1 * x2 + c

    If no adequate list of coefficients (a1,a2,a3,c) is passed, these are
    drawn from a standard-normal distribution.

    :type n: int
    :type coefs: unknown or ndarray
    :return:
    """
    # With at most n dimensions, we get 2 ** n - 1 possible interaction terms,
    # and need one additional coefficient for the constant
    ncoef = 2 ** n
    if coefs is None or len(coefs) < ncoef:
        coefs = np.random.randn(ncoef)

    if high is None:
        high = n

    low = int(low)
    high = int(high)

    assert low >= 1 and high <= n

    for i in range(low, high + 1):
        for subset in combinations(range(n), i):
            # Build docstring to have some clue what function is built
            dstr = 'f(x) ='
            cidx = 0
            for j in range(len(subset)):
                for ia in combinations(subset, j + 1):
                    astr = ''
                    for k in ia:
                        astr += 'x{:d}*'.format(k)
                    # Remove last * character
                    astr = astr[:-1]
                    dstr += ' {:+5.4f}*{}'.format(coefs[cidx], astr)
                    cidx += 1
            dstr = '{} {:+5.4f}'.format(dstr, coefs[ncoef-1])

            # build function that contains all interactions of indices in
            # subset and a constant
            def f(*args):
                args = ascontainer(args)
                res = 0
                cidx = 0
                for j in range(len(subset)):
                    for ia in combinations(subset, j + 1):
                        ires = 1
                        for k in ia:
                            ires *= args[k]
                        res += ires * coefs[cidx]
                        cidx += 1
                return res + coefs[ncoef - 1]
            f.__doc__ = dstr
            yield f


class TestBase:

    @pytest.fixture
    def f_interp(self):
        raise NotImplementedError()

    @pytest.fixture
    def ndim(self):
        raise NotImplementedError()

    @pytest.fixture
    def f_linear(self):
        return func_generator(1)

    @pytest.fixture
    def f_bilinear(self):
        return func_generator(2)

    @pytest.fixture
    def f_trilinear(self):
        return func_generator(3)

    @pytest.fixture
    def f_zero(self):
        def f(*args):
            args = ascontainer(args)
            return np.zeros_like(args[0])
        return f

    @pytest.fixture
    def f_const(self):
        c = np.asscalar(np.random.randn(1))

        def f(*args):
            args = ascontainer(args)
            return np.ones_like(args[0]) * c
        return f

    @pytest.fixture
    def data_shape(self):
        raise NotImplementedError()

    @pytest.fixture
    def data(self, data_shape):

        xp = tuple(np.sort(np.random.randn(s)) for s in data_shape)
        x = tuple([linspace(np.min(z), np.max(z), 5) for z in xp])

        return xp, x

    def test_dimensions(self, data, f_interp, f_const):
        """
        Test whether array dimensions of return array is conformable, and whether
        exceptions are raised if input dimensions are non-conformable.
        """
        xp, _ = data

        test_dimensions(f_interp, f_const, xp)

    def test_identity(self, data, f_interp, f_nonlinear):
        """
        Test whether interpolating exactly at interpolation nodes gives
        correct result.
        """

        xp, _ = data
        # we need to do this on a cartesian product of dimensions in xp,
        # as the vectors in xp might be of different length (if we have
        # different number of knots in each dimension.

        if len(xp) > 1:
            x = np.meshgrid(*xp, indexing='ij')
            x = tuple(arr.reshape(-1) for arr in x)
        else:
            x = xp

        test_equality(f_interp, f_nonlinear, xp, x)

    def test_constant(self, data, f_interp, f_const):
        xp, x = data
        test_equality(f_interp, f_const, xp, x)

    def test_zero(self, data, f_interp, f_zero):
        xp, x = data
        test_equality(f_interp, f_zero, xp, x)

    def test_extrapolate(self, data, f_interp, ndim):
        """
        Verify that extrapolation of linear functions of appropriate
        dimension works.
        """
        xp, x = data
        x_ext = extend(x)

        for f in func_generator(ndim, high=1):
            test_equality(f_interp, f, xp, x_ext, tol=1e-9)

    def test_interpolate(self, data, f_interp, ndim):
        """
        Verify that interpolation of 'pseudo'-linear functions works exactly.
        """

        xp, x = data
        for f in func_generator(ndim):
            test_equality(f_interp, f, xp, x, tol=1e-9)


def get_margins(ndim, excluded):
    """
    Return tuple of non-excluded margins given that we have ndim dimensions.
    Excluded margins are zero-based!
    :param ndim:
    :param excluded:
    :return:
    """
    excluded = ascontainer(excluded)
    return tuple(np.sort(tuple(set(range(ndim)) - set(excluded))))


def extend(array_like, by_frac=0.4, add_n=None):
    """
    Extend array or sequence of arrays by (by_frac/2 * 100)% of the distance
    between the first and last element of each array by adding add_n//2
    elements to each end of the array.
    """

    arrays = ascontainer(array_like)

    arr_extended = list()
    for arr in arrays:
        n = add_n // 2 if add_n is not None else arr.shape[0] // 2

        diff = (arr[-1] - arr[0]) * by_frac / 2
        dx = diff/n

        arr1 = np.linspace(arr[0] - diff, arr[0] - dx, n)
        arr2 = np.linspace(arr[-1] + dx, arr[-1] + diff, n)

        arr_extended.append(np.sort(np.hstack((arr1, arr, arr2))))

    if not isinstance(array_like, (list, tuple)):
        return arr_extended[0]

    # convert to tuple if we received tuple
    if isinstance(array_like, tuple):
        arr_extended = tuple(arr_extended)

    return arr_extended


def test_dimensions(f_interp, f, xp, length=(1, 2, 10, 100, 1001)):
    """
    Verify that dimensions of input and output arrays are conformable,
    and that appropriate exceptions are raised when non-conformable
    arrays are passed as arguments.
    """

    xp = list(xp)
    fp = f_array(f, *xp)

    for n in length:
        x = [linspace(np.min(z), np.max(z), n) for z in xp]
        args = x + xp + [fp]
        fx = f_interp(*args)

        assert x[0].shape == fx.shape

        # skip this for n=1 arrays, as truncating arrays will result in
        # zero-length buffers, and we do not bother to catch this error
        if n == 1:
            continue

        if len(x) > 1:
            for i in range(len(x)):
                x_new = x.copy()
                xi = x[i].copy()
                x_new[i] = xi[:-1]
                args = x_new + xp + [fp]

                with pytest.raises(ValueError):
                    f_interp(*args)

        for i in range(len(xp)):
            xp_new = xp.copy()
            xpi = xp[i].copy()
            xp_new[i] = xpi[:-1]

            args = x + xp_new + [fp]
            with pytest.raises(ValueError):
                f_interp(*args)


def test_equality(f_interp, f, xp, x, tol=1e-10):
    """
    Verify that interpolated values are almost equal (i.e. abs.
    difference between actual and interpolated values is smaller than tol).
    """

    xp = tuple(xp)
    x = tuple(x)

    fp = f_array(f, *xp)
    args = x + xp + (fp, )
    fx_hat = f_interp(*args)

    # True values
    fx = f(*x)

    assert np.max(np.abs(fx-fx_hat)) < tol


def test_margin(f_interp, f, xp, x, marg,
                 f_interp_marg, f_marg, tol=1e-10):
    """
    Verify that 'marginal' interpolation works, i.e. keeping one or two
    margins constant and effectively testing on a lower-dimensional
    function (f is assumed to be lower-dimensional).
    """

    xp = ascontainer(xp)
    x = ascontainer(x)
    marg = ascontainer(marg)

    # check that we are actually testing lower-dimensional interpolation
    assert len(marg) < len(xp)

    xp_marg = list()
    x_marg = list()
    for i in marg:
        xp_marg.append(xp[i])
        x_marg.append(x[i])

    fp = f_array(f, *xp)
    args = x + xp + (fp, )
    fx = f_interp(*args)

    fp_marg = f_array(f_marg, *xp_marg)
    # interpolate using lower-dimensional function
    args = x_marg + xp_marg + [fp_marg]
    fx_marg = f_interp_marg(*args)

    assert np.max(np.abs(fx - fx_marg)) < tol


