
from pydynopt.utils.bsearch cimport cy_bsearch

cdef struct interp_accel:
    size_t index

cdef inline size_t c_interp_bsearch(double *, double, size_t, size_t) nogil
cdef inline size_t c_interp_find(double *, double, size_t, interp_accel*) nogil
