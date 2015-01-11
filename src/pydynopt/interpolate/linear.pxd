


cdef int _interp1d_linear_vec(double[:] x0, double[:] xp,
                                     double[:] fp, double[:] out) nogil

cdef int _interp1d_linear(double x, double[:] xp, double[:] fp,
                                 double *out)