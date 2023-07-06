
__author__ = 'Richard Foltyn'

import numpy as np
import unittest as ut

from pydynopt.utils import bsearch, bsearch_eq


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
                    i = bsearch_eq(arr, key, first=True)
                    j = np.min(np.where(np.logical_and(arr <= key, arr >= key))[0])
                    self.assertEqual(i, j)

                    i = bsearch_eq(arr, key, first=False)
                    j = np.max(np.where(np.logical_and(arr <= key, arr >= key))[0])
                    self.assertEqual(i, j)

    def test_first_last(self):
        """
        Test whether returned index is the same for 'first' and 'last' if
        values in array are unique.
        """
        for n in (1, 2, 3, 10, 101):
            arr = np.arange(n, dtype=np.float)

            # Exact matches
            for key in arr:
                first = bsearch_eq(arr, key, first=True)
                last = bsearch_eq(arr, key, first=False)
                self.assertEqual(first, last)

            # < matches
            for key in (1.5, 2.5, 10.5):
                first = bsearch_eq(arr, key, first=True)
                last = bsearch_eq(arr, key, first=False)
                self.assertEqual(first, last)

    def test_value_error(self):
        """
        Check whether ValueError is raised when passing a key that is smaller
        than first element in array.
        """
        arr = np.arange(10, 20)
        self.assertRaises(ValueError, bsearch_eq, arr=arr, key=1)

    def test_equal(self):

        for n in (1, 2, 3, 10, 101):
            arr = np.ones((n,), dtype=np.float)

            i = bsearch_eq(arr, 1.0, first=True)
            self.assertEqual(i, 0)

            i = bsearch_eq(arr, 1.0, first=False)
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
                    i = bsearch(arr, key)
                    j = np.max(np.where(arr <= key)[0])
                    self.assertEqual(i, j)

    def test_bsearch_last_boundaries(self):
        for n in (1, 2, 3, 10, 101):
            arr = np.sort(np.random.uniform(low=1, high=100, size=(n,)))

            # should get -1 if key < arr[0]
            i = bsearch(arr, arr[0] - 10)
            self.assertEqual(i, -1)

            # should get arr.shape[0] if key > arr[-1]
            i = bsearch(arr, arr[-1] + 10)
            self.assertEqual(i, arr.shape[0])

            # should get arr.shape[0] - 1 if key = arr[-1]
            i = bsearch(arr, arr[-1])
            self.assertEqual(i, arr.shape[0] - 1)

            # should get 0 if key == arr[0]
            i = bsearch(arr, arr[0])
            self.assertEqual(i, 0)