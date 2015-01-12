
from ..common.types cimport real_t

cpdef int _interp1d_linear_vec(real_t[:] x0, real_t[:] xp,
                                     real_t[:] fp, real_t[:] out) nogil

cpdef inline real_t _interp1d_linear_impl(real_t x, real_t[:] xp,
                                         real_t[:] fp) nogil

cdef int _interp1d_linear(real_t x, real_t[:] xp, real_t[:] fp,
                                 real_t *out)