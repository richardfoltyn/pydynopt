
__author__ = 'Richard Foltyn'

from cython import boundscheck


cpdef long cy_bsearch_eq(int_real_t[:] arr, int_real_t key,
                              bint first) nogil:
    """
    Implementation of binary search that returns the index i such that
    arr[i] <= key. If arr contains a sequence of several arr[i] == key,
    then the min {i | arr[i] == key} is returned, otherwise
    max {i | arr[i] == key}.
    """

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

cdef long cy_bsearch(int_real_t *arr, int_real_t key, size_t length) nogil:
    """
    Returns index i such that arr[i] <= key < arr[i+1].
    Boundary conditions are handled as follows:
        1) if key < arr[0]  then cy_bsearch(arr, key) == -1
        2) if key = arr[-1] then cy_bsearch(arr, key) == arr.shape[0] - 1
        3) if key > arr[-1] then cy_bsearch(arr, key) == arr.shape[0]
    """

    cdef long lb = 0, ub = length
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

cdef long c_bsearch(double[::1] arr, double key) nogil:
    """
    Returns index i such that arr[i] <= key < arr[i+1].
    Boundary conditions are handled as follows:
        1) if key < arr[0]  then c_bsearch(arr, key) == -1
        2) if key = arr[-1] then c_bsearch(arr, key) == arr.shape[0] - 1
        3) if key > arr[-1] then c_bsearch(arr, key) == arr.shape[0]
    """

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

cdef inline size_t c_bsearch_lb(double[::1] arr, double key) nogil:

    cdef size_t midx, lb = 0, ub = arr.shape[0] - 1

    while ub > (lb + 1):
        midx = (ub + lb) // 2
        if arr[midx] > key:
            ub = midx
        else:
            lb = midx

    return lb

@boundscheck(True)
def bsearch_eq(int_real_t[:] arr, int_real_t key, first=True):
    if arr[0] > key:
        raise ValueError('arr[0] <= key required!')

    cdef bint cfirst = 1 if first else 0

    return cy_bsearch_eq(arr, key, cfirst)

@boundscheck(True)
def bsearch(int_real_t[:] arr, int_real_t key):
    return cy_bsearch(&(arr[0]), key, arr.shape[0])

def bsearch_lb(double[::1] arr, double key):
    return c_bsearch_lb(arr, key)