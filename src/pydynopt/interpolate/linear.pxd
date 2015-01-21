
from ..common.types cimport real_t

cdef inline long _find_lb(real_t *xp, real_t x, unsigned long length) nogil

cpdef inline real_t _interp1d_linear_impl(real_t x, real_t[:] xp,
                                          real_t[:] fp) nogil

cpdef int _interp1d_linear(real_t[:] x, real_t[:] xp, real_t[:] fp,
                              real_t [:]) nogil

cdef int _interp2d_bilinear(real_t[:] x0, real_t[:] y0,
        real_t[:] x, real_t[:] y,
        real_t[:, :] fval, real_t[:] out) nogil

cdef inline real_t _interp2d_bilinear_impl(real_t x, real_t y, real_t[:] xp,
            real_t[:] yp, real_t[:, :] fval) nogil

################################################################################
# Trilinear interpolation interfaces

cdef inline real_t _interp3d_trilinear_impl(real_t x, real_t y, real_t z,
        real_t[:] xp, real_t[:] yp, real_t[:] zp,
        real_t[:, :, :] fp) nogil

cdef int _interp3d_trilinear(real_t[:] x, real_t[:] y, real_t[:] z,
        real_t[:] xp, real_t[:] yp, real_t[:] zp,
        real_t[:, :, :] fp, real_t[:] out) nogil