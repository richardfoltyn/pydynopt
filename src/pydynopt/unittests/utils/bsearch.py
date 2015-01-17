from __future__ import division, absolute_import, print_function
__author__ = 'Richard Foltyn'

import numpy as np
import unittest2 as ut

from pydynopt.utils import bsearch, BSearchFlag, _bsearch


class BSearchTest(ut.TestCase):

    def test_random(self):
        """
        Run tests with randomly generated arrays of floats, using both unique
        and non-unique array values.
        """

        # include some pathological array sizes in tests
        for n in (1, 2, 3, 10, 101):
            arr1 = np.random.uniform(low=0, high=100, size=(n,))
            arr1 = np.sort(arr1)

            # test first/last mechanism for repeated values
            arr2 = np.repeat(arr1, 2)

            for arr in (arr1, arr2):

                if n <= 10:
                    # for small arrays, test for all indices
                    idx = np.arange(n)
                else:
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
        for n in (1, 2, 3, 10, 101):
            arr = np.arange(n)

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

    def test_equal(self):

        for n in (1, 2, 3, 10, 101):
            arr = np.ones((n,), dtype=np.float)

            i = bsearch(arr, 1.0, BSearchFlag.first)
            self.assertEqual(i, 0)

            i = bsearch(arr, 1.0, BSearchFlag.last)
            self.assertEqual(i, n - 1)

    def test_bsearch_last_random(self):
        # include some pathological array sizes in tests
        for n in (1, 2, 3, 10, 101):
            arr1 = np.random.uniform(low=0, high=100, size=(n,))
            arr1 = np.sort(arr1)

            # add repeated values
            arr2 = np.repeat(arr1, 2)

            for arr in (arr1, arr2):

                if n <= 10:
                    # for small arrays, test for all indices
                    idx = np.arange(n)
                else:
                    idx = np.random.randint(low=0, high=arr.shape[0] - 1, size=(10, ))

                for key in arr[idx]:
                    i = _bsearch(arr, key)
                    j = np.max(np.where(arr <= key)[0])
                    self.assertEqual(i, j)

    def test_bsearch_last_boundaries(self):
        for n in (1, 2, 3, 10, 101):
            arr = np.sort(np.random.uniform(low=1, high=100, size=(n,)))

            # should get -1 if key < arr[0]
            i = _bsearch(arr, arr[0] - 10)
            self.assertEqual(i, -1)

            # should get arr.shape[0] if key > arr[-1]
            i = _bsearch(arr, arr[-1] + 10)
            self.assertEqual(i, arr.shape[0])

            # should get arr.shape[0] - 1 if key = arr[-1]
            i = _bsearch(arr, arr[-1])
            self.assertEqual(i, arr.shape[0] - 1)

            # should get 0 if key == arr[0]
            i = _bsearch(arr, arr[0])
            self.assertEqual(i, 0)