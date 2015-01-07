from __future__ import division, absolute_import, print_function
__author__ = 'Richard Foltyn'

import numpy as np
import unittest2 as ut

from pydynopt.utils import bsearch, BSearchFlag


class BSearchTest(ut.TestCase):

    def test_random(self):
        """
        Run tests with randomly generated arrays of floats, using both unique
        and non-unique array values.
        """
        arr1 = np.random.uniform(low=0, high=100, size=(100,))
        arr1 = np.sort(arr1)

        # test first/last mechanism for repeated values
        arr2 = np.repeat(arr1, 2)

        for arr in (arr1, arr2):
            idx = np.random.randint(low=0, high=arr.shape[0] - 1, size=(10, ))
            for key in arr[idx]:
                i = bsearch(arr, key, BSearchFlag.first)
                j = np.min(np.where(np.logical_and(arr <= key, arr >= key))[0])
                self.assertEqual(i, j)

                i = bsearch(arr, key, BSearchFlag.last)
                j = np.max(np.where(np.logical_and(arr <= key, arr >= key))[0])
                self.assertEqual(i, j)

    def test_first_last(self):
        """
        Test whether returned index is the same for 'first' and 'last' if
        values in array are unique.
        """
        arr = np.arange(10)

        # Exact matches
        for key in arr:
            first = bsearch(arr, key, BSearchFlag.first)
            last = bsearch(arr, key, BSearchFlag.last)
            self.assertEqual(first, last)

        # < matches
        for key in (1.5, 2.5, 10.5):
            first = bsearch(arr, key, BSearchFlag.first)
            last = bsearch(arr, key, BSearchFlag.last)
            self.assertEqual(first, last)

    def test_value_error(self):
        """
        Check whether ValueError is raised when passing a key that is smaller
        than first element in array.
        """
        arr = np.arange(10, 20)
        self.assertRaises(ValueError, bsearch, arr=arr, key=1)