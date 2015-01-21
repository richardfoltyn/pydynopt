
from pydynopt.common.types cimport int_real_t

cdef long _bsearch(int_real_t *arr, int_real_t key, unsigned long length) nogil

cpdef long _bsearch_eq(int_real_t[:] arr, int_real_t key,
                              bint first) nogil
