__author__ = 'Richard Foltyn'

import numpy as np
from numpy import linspace

import unittest as ut
from collections import Iterable


def f_array(f, *xp):
    xp = ascontainer(xp)
    if len(xp) > 1:
        mgrid = np.meshgrid(*xp, indexing='ij')
        return f(*mgrid)
    else:
        return f(*xp)


def ascontainer(x, container=tuple):
    if isinstance(x, Iterable):
        return container(x)
    else:
        return container((x,))


def extend(array_like, by_frac=0.4, add_n=None):
    """
    Extend array or sequence of arrays by (by_frac/2 * 100)% of the distance
    between the first and last element of each array by adding add_n//2
    elements to each end of the array.
    """

    if not isinstance(array_like, (list, tuple)):
        arrays = [array_like]
    else:
        arrays = []
        arrays.extend(array_like)

    arr_extended = list()
    for arr in arrays:
        n = add_n // 2 if add_n is not None else arr.shape[0] // 2

        diff = (arr[-1] - arr[0]) * by_frac / 2
        dx = diff/n

        arr1 = np.linspace(arr[0] - diff, arr[0] - dx, n)
        arr2 = np.linspace(arr[-1] + dx, arr[-1] + diff, n)

        arr_extended.append(np.sort(np.hstack((arr1, arr, arr2))))

    if len(arr_extended) == 1 and not isinstance(array_like, list):
        arr_extended = arr_extended[0]

    if isinstance(array_like, tuple):
        arr_extended = tuple(arr_extended)

    return arr_extended


class AbstractTest(ut.TestCase):

    def _test_dimensions(self, f_interp, f, xp, length=(1, 2, 10, 100, 1001)):
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

            self.assertTrue(x[0].shape == fx.shape)

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

                    self.assertRaises(ValueError, f_interp, *args)

            for i in range(len(xp)):
                xp_new = xp.copy()
                xpi = xp[i].copy()
                xp_new[i] = xpi[:-1]

                args = x + xp_new + [fp]
                self.assertRaises(ValueError, f_interp, *args)

    def _test_equality(self, f_interp, f, xp, x, tol=1e-12):
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

        self.assertTrue(np.max(np.abs(fx-fx_hat)) < tol)

    def _test_margin(self, f_interp, f, xp, x, marg,
                     f_interp_marg, f_marg, tol=1e-12):
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

        self.assertTrue(np.max(np.abs(fx - fx_marg)) < tol)


