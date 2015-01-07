__author__ = 'Richard Foltyn'

from enum import IntEnum
from ._bsearch import _bsearch

import numpy as np


class BSearchFlag(IntEnum):
    first = 1
    last = 0


def bsearch(arr, key, which=BSearchFlag.first):
    if arr[0] > key:
        raise ValueError('arr[0] <= key required!')

    arr = np.atleast_1d(arr).ravel()

    return _bsearch(arr, key, 0, arr.shape[0] - 1, bool(which))