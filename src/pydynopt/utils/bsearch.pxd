
from pydynopt.common.types cimport int_real_t

cdef long cy_bsearch(int_real_t *arr, int_real_t key, unsigned long length) nogil

cpdef long cy_bsearch_eq(int_real_t[:] arr, int_real_t key,
                              bint first) nogil
