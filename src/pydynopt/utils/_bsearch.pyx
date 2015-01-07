
from cython cimport numeric

from ..common.types cimport int_real_t

from cython import boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
cpdef unsigned int _bsearch(int_real_t[:] arr, int_real_t key, bint which):

    cdef unsigned int n = arr.shape[0]
    cdef unsigned int midx = n // 2
    
    cdef unsigned int bound = 0
    cdef short dx = -1
    if which == 1:
        dx = 1
        bound = n-1
        
    cdef int_real_t mval = arr[midx]
    
    if mval > key:
        return _bsearch(arr[:midx], key, which)
    elif mval < key:
        return _bsearch(arr[midx:], key, which)
    else:
        while midx != bound and mval == arr[midx+dx]:
            midx += dx
        return midx
        