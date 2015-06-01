
from pydynopt.utils.bsearch cimport cy_bsearch

cdef struct interp_accel:
    size_t index