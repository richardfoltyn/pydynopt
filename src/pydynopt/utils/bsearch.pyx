# cython: profile = False

__author__ = 'Richard Foltyn'

from enum import IntEnum


from cython import boundscheck, wraparound, cdivision

from ..common.types cimport int_real_t

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef long _bsearch_eq(int_real_t[:] arr, int_real_t key,
                              bint first) nogil:

    if arr[0] > key:
        return -1

    cdef long lb = 0, ub = arr.shape[0] - 1

    if ub == 0:
        return 0

    cdef long midx

    # Algorithm: first get us into the correct 2-element segment of the array
    # by iterating until only two elements are left. Then pick whichever is
    # the right one according to the value of 'first'.
    if first:
        while ub - lb > 1:
            midx = (ub + lb) // 2
            if key > arr[midx]:
                lb = midx
            else:
                ub = midx

        if arr[lb] == arr[ub]:
            return lb
        else:
            return ub if arr[ub] <= key else lb
    else:
        while ub - lb > 1:
            midx = (ub + lb) // 2
            if key >= arr[midx]:
                lb = midx
            else:
                ub = midx

        if arr[lb] == arr[ub]:
            return ub
        else:
            return ub if arr[ub] <= key else lb


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef long _bsearch(int_real_t[:] arr, int_real_t key) nogil:


    cdef long lb = 0, ub = arr.shape[0]
    cdef long midx

    if key > arr[ub-1]:
        return ub

    while lb < ub:
        midx = (ub + lb) // 2
        if key >= arr[midx]:
            lb = midx + 1
        else:
            ub = midx

    return lb - 1



class BSearchFlag(IntEnum):
    first = 1
    last = 0


def bsearch(int_real_t[:] arr, int_real_t key, which=BSearchFlag.first):
    if arr[0] > key:
        raise ValueError('arr[0] <= key required!')

    return _bsearch_eq(arr, key, <bint>bool(which))