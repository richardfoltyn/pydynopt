__author__ = 'Richard Foltyn'

from enum import IntEnum
import numpy as np

from cython import boundscheck, wraparound, cdivision

from ..common.types cimport int_real_t

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef unsigned long _bsearch_impl(int_real_t[:] arr, int_real_t key,
                            unsigned long lb, unsigned long ub,
                            bint first) nogil:

    if arr[0] > key:
        return -1

    if ub - lb <= 1:
        if arr[lb] != arr[ub]:
            return ub if arr[ub] <= key else lb
        else:
            return lb if first else ub


    cdef unsigned long midx = (ub + lb) // 2
    cdef int_real_t mval = arr[midx]

    if (key > mval and first) or (key >= mval and not first):
        return _bsearch_impl(arr, key, midx, ub, first)
    else:
        return _bsearch_impl(arr, key, lb, midx, first)


cpdef unsigned long _bsearch(int_real_t[:] arr, int_real_t key,
                             bint first) nogil:

    cdef unsigned long ifrom = 0, ito = arr.shape[0] - 1

    return _bsearch_impl(arr, key, ifrom, ito, first)


class BSearchFlag(IntEnum):
    first = 1
    last = 0


def bsearch(int_real_t[:] arr, int_real_t key, which=BSearchFlag.first):
    if arr[0] > key:
        raise ValueError('arr[0] <= key required!')

    cdef unsigned long ifrom = 0, ito = arr.shape[0] - 1

    return _bsearch_impl(arr, key, ifrom, ito, <bint>bool(which))