
from ..common.types cimport int_real_t

cpdef unsigned long _bsearch(int_real_t[:] arr, int_real_t key,
                            bint first) nogil except -1

cdef unsigned long _bsearch_impl(int_real_t[:] arr, int_real_t key,
                            unsigned long lb, unsigned long ub,
                            bint first) nogil except -1