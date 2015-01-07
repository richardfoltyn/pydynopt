
from ..common.types cimport int_real_t

cpdef unsigned int _bsearch(int_real_t[:] arr, int_real_t key,
                            unsigned int lb, unsigned int ub,
                            bint first) except -1