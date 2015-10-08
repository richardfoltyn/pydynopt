
from pydynopt.common.types cimport int_real_t

cdef long cy_bsearch(int_real_t *arr, int_real_t key, size_t length) nogil

cdef long c_bsearch(double[::1] arr, double key) nogil
cdef inline size_t c_bsearch_lb(double[::1], double) nogil

cpdef long cy_bsearch_eq(int_real_t[:] arr, int_real_t key,
                              bint first) nogil
