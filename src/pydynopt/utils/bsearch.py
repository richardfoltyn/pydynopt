__author__ = 'Richard Foltyn'

from enum import IntEnum
from ._bsearch import _bsearch


class BSearchFlag(IntEnum):
    first = 0
    last = 1


def bsearch(arr, key, which=BSearchFlag.first):
    return _bsearch(arr, key, bool(which))