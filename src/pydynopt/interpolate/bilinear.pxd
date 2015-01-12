
from ..common.types cimport real_t

cdef int _interp2d_bilinear_vec(real_t[:] x0, real_t[:] y0,
        real_t[:] x, real_t[:] y,
        real_t[:, :] fval, real_t[:] out) nogil

cdef int _interp2d_bilinear(real_t x0, real_t y0,
        real_t[:] x, real_t[:] y,
        real_t[:, :] fval, real_t *out)