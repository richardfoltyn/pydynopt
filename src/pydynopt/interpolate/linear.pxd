
from pydynopt.common.ndarray_wrappers cimport make_ndarray
from pydynopt.utils.bsearch cimport cy_bsearch

from pydynopt.interpolate.search cimport interp_accel, c_interp_find

from numpy cimport ndarray

cdef inline long cy_find_lb(double *xp, double x, unsigned long length) nogil

cdef inline double \
        cy_interp1d_linear(double, double[::1], double[:]) nogil

cdef int c_interp2d_bilinear_vec(double[:], double[:],
        double[::1] x1p, double[::1] x2p,
        double[:, ::1] fval, double[:] out) nogil

cdef double c_interp2d_bilinear(
        double x, double y, double[::1] xp,
        double[::1] yp, double[:, ::1] fval,
        interp_accel *acc1, interp_accel *acc2) nogil

################################################################################
# Trilinear interpolation interfaces

cdef inline double _interp3d_trilinear_impl(double x, double y, double z,
        double[::1] xp, double[::1] yp, double[::1] zp,
        double[:, :, :] fp) nogil

cdef int _interp3d_trilinear(double[:] x, double[:] y, double[:] z,
        double[::1] xp, double[::1] yp, double[::1] zp,
        double[:, :, :] fp, double[:] out) nogil