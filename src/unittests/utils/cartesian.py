from __future__ import division, absolute_import, print_function
__author__ = 'Richard Foltyn'

import numpy as np
import unittest2 as ut

from pydynopt.utils import cartesian, cartesian2d, _cartesian2d


# Benchmark method implemented in python
def py_cartesian(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    b1 = np.tile(b, (1, a.shape[1]))
    a1 = np.repeat(a, b.shape[1], axis=1)

    return np.vstack((a1, b1))


class CartesianTest(ut.TestCase):

    def get_arrays(self, dtype=np.int64):
        a = np.arange(15, dtype=dtype).reshape((3, -1))
        b = np.arange(6, dtype=dtype).reshape((2, -1))

        return a, b

    def check_dims(self, a, b, ab):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        self.assertEqual(ab.shape[0], a.shape[0] + b.shape[0])
        self.assertEqual(ab.shape[1], a.shape[1] * b.shape[1])

    def test_2d(self):

        a, b = self.get_arrays()

        ab1 = py_cartesian(a, b)
        ab2 = cartesian2d(a, b)

        self.assertTrue(np.all(ab1.shape == ab2.shape))
        self.check_dims(a, b, ab1)
        self.assertTrue(np.all(ab1 == ab2))

    def test_types(self):
        for t in (np.float32, np.float64, np.int16, np.int32, np.int64):
            a, b = self.get_arrays(dtype=t)

            ab = cartesian2d(a, b)
            self.assertEqual(t, ab.dtype)

    def test_1d(self):
        a = np.arange(10)
        b = np.arange(11, 15)

        ab1 = cartesian(a, b)
        ab2 = py_cartesian(a, b)

        self.check_dims(a, b, ab1)
        self.assertTrue(np.all(ab1.shape == ab2.shape))
        self.assertTrue(np.all(ab1 == ab2))

    def test_contiguous(self):
        a = np.ascontiguousarray(np.arange(15).reshape((3, -1)))
        b = np.ascontiguousarray(np.arange(6).reshape((2, -1)))

        ab = cartesian2d(a, b)

        self.assertTrue(np.isfortran(ab.T))

        a = np.array(np.arange(15).reshape((3, -1)), order='F')
        b = np.array(np.arange(6).reshape((2, -1)), order='F')

        ab = cartesian2d(a.T, b.T)

        self.assertTrue(np.isfortran(ab.T))

    def test_pathological(self):
        """
        Test some weird arrays which make little sense to pass as arguments
        """

        a = np.array([[1]])
        b = np.array([[1]])

        ab1 = cartesian2d(a, b)
        ab2 = py_cartesian(a, b)

        self.assertTrue(np.all(ab1.shape == ab2.shape))
        self.assertTrue(np.all(ab1 == ab2))
        self.check_dims(a, b, ab1)

        a = np.array([1])
        b = np.array([2])

        ab1 = cartesian(a, b)
        ab2 = py_cartesian(a, b)

        self.assertItemsEqual(ab1.shape, ab2.shape)
        self.assertItemsEqual(ab1, ab2)
        self.check_dims(a, b, ab1)

    def test_out(self):
        """
        Test whether passing `out` argument works.
        """

        a, b = self.get_arrays()
        ab2 = py_cartesian(a, b)
        ab1 = np.empty_like(ab2)

        cartesian2d(a, b, ab1)

        self.assertItemsEqual(ab1.shape, ab2.shape)
        self.assertTrue(np.all(ab1 == ab2))

    def test_out_dims(self):
        """
        Test whether function aborts when output array has wrong dimensions.
        """

        a, b = self.get_arrays()
        ab1 = py_cartesian(a, b)
        out = np.empty_like(ab1)

        self.assertRaises(ValueError, cartesian2d, a=a, b=b,
                          out=out[:-1, :-1])

        retval = _cartesian2d(a, b, out[:-1, :])
        self.assertEqual(retval, -1)