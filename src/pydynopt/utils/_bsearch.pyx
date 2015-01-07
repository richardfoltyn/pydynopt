from ..common.types cimport int_real_t

from cython import boundscheck, wraparound, cdivision


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef unsigned int _bsearch(int_real_t[:] arr, int_real_t key,
                            unsigned int lb, unsigned int ub,
                            bint first) nogil except -1:

    if arr[0] > key:
        return -1

    if ub - lb <= 1:
        if arr[lb] != arr[ub]:
            return ub if arr[ub] <= key else lb
        else:
            return lb if first else ub


    cdef unsigned int midx = (ub + lb) // 2
    cdef int_real_t mval = arr[midx]

    if (key > mval and first) or (key >= mval and not first):
        return _bsearch(arr, key, midx, ub, first)
    else:
        return _bsearch(arr, key, lb, midx, first)