__author__ = 'Richard Foltyn'


import unittest2 as ut
import numpy as np

from pydynopt.interpolate import interp_bilinear


class BilinearTest(ut.TestCase):

    def setUp(self):
        self.a = np.random.randn(5)
        self.b = np.random.randn(5)

        def func1(a, b):
            return np.exp(a) + np.power(b, 2)

        self.f1 = func1

        z = [[func1(u, v) for v in self.b] for u in self.a]
        self.z = np.asarray(z)

    def test_random(self):

        a0 = self.a * 1.1
        b0 = self.b * .9

        zinterp = interp_bilinear(a0, b0, self.a, self.b, self.z)

        ztrue = [[self.f1(u, v) for v in b0] for u in a0]
        ztrue = np.asarray(ztrue)

        print(np.max(np.abs(zinterp - ztrue)))

        self.assertTrue(np.all(a0.shape == zinterp.shape))

    def test_self(self):

        zinterp = interp_bilinear(self.a, self.b, self.a, self.b,
                                  self.z)

        self.assertAlmostEqual(np.abs(np.max(zinterp - np.diag(self.z))))
