
from pydynopt.common.ndarray_wrappers cimport make_ndarray
from pydynopt.utils.bsearch cimport _bsearch

from numpy cimport ndarray

cdef inline long cy_find_lb(double *xp, double x, unsigned long length) nogil

cdef inline double \
        cy_interp1d_linear(double, double[::1], double[:]) nogil

cdef int _interp2d_bilinear(double[:] x0, double[:] y0,
        double[::1] x, double[::1] y,
        double[:, :] fval, double[:] out) nogil

cdef inline double _interp2d_bilinear_impl(double x, double y, double[::1] xp,
            double[::1] yp, double[:, :] fval) nogil

################################################################################
# Trilinear interpolation interfaces

cdef inline double _interp3d_trilinear_impl(double x, double y, double z,
        double[::1] xp, double[::1] yp, double[::1] zp,
        double[:, :, :] fp) nogil

cdef int _interp3d_trilinear(double[:] x, double[:] y, double[:] z,
        double[::1] xp, double[::1] yp, double[::1] zp,
        double[:, :, :] fp, double[:] out) nogil