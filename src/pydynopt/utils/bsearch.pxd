
from ..common.types cimport int_real_t


cpdef long _bsearch_eq(int_real_t[:] arr, int_real_t key,
                              bint first) nogil
cpdef long _bsearch(int_real_t[:] arr, int_real_t key) nogil