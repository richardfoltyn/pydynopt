
cdef inline int _interp_bilinear_vec(double[:] x0, double[:] y0,
        double[:] x, double[:] y,
        double[:, :] fval, double[:] out) nogil

cdef inline int _interp_bilinear(double x0, double y0,
        double[:] x, double[:] y,
        double[:, :] fval, double *out)