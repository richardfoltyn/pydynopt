
cdef int _interp2d_bilinear_vec(double[:] x0, double[:] y0,
        double[:] x, double[:] y,
        double[:, :] fval, double[:] out) nogil

cdef int _interp2d_bilinear(double x0, double y0,
        double[:] x, double[:] y,
        double[:, :] fval, double *out)