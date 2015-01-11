
from cython import boundscheck, wraparound, cdivision

import numpy as np

from ..utils.bsearch cimport _bsearch_impl

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline int _interp_bilinear_vec(double[:] x0, double[:] y0,
        double[:] x, double[:] y,
        double[:, :] fval, double[:] out) nogil:

    if x.shape[0] != fval.shape[0] or y.shape[0] != fval.shape[1]:
        return -1
    if x.shape[0] < 2 or y.shape[0] < 2:
        return -2
    if x0.shape[0] != y0.shape[0] != out.shape[0]:
        return -3

    cdef unsigned long xe = x.shape[0] - 1, ye = y.shape[0] - 1
    cdef unsigned long ifrom = 0
    cdef bint first = 1

    cdef double x_lb, x_ub, y_lb, y_ub
    cdef unsigned long ix_lb, iy_lb

    cdef double xi, yi
    cdef double wx, wy
    cdef double fx1, fx2

    for i in range(x0.shape[0]):

        xi, yi = x0[i], y0[i]

        if xi <= x[0]:
            ix_lb = 0
        elif xi >= x[xe]:
            ix_lb = xe - 1
        else:
            ix_lb = _bsearch_impl(x, xi, ifrom, xe, first)

        x_lb = x[ix_lb]
        x_ub = x[ix_lb + 1]

        if yi <= y[0]:
            iy_lb = 0
        elif yi >= y[ye]:
            iy_lb = ye - 1
        else:
            iy_lb = _bsearch_impl(y, yi, ifrom, ye, first)

        y_lb = y[iy_lb]
        y_ub = y[iy_lb + 1]

        wx = (xi - x_lb) / (x_ub - x_lb)
        fx1 = (1-wx) * fval[ix_lb, iy_lb] + wx * fval[ix_lb + 1, iy_lb]
        fx2 = (1-wx) * fval[ix_lb, iy_lb + 1] + wx * fval[ix_lb + 1, iy_lb + 1]

        wy = (yi - y_lb) / (y_ub - y_lb)
        out[i] = (1-wy) * fx1 + wy * fx2

    return 0


cdef inline int _interp_bilinear(double x0, double y0,
        double[:] x, double[:] y,
        double[:, :] fval, double *out):

    cdef double[:] xmv, ymv
    cdef double[:] outmv

    xmv = <double[:1]>&x0
    ymv = <double[:1]>&y0
    outmv = <double[:1]>out

    return _interp_bilinear_vec(xmv, ymv, x, y, fval, outmv)


def interp_bilinear(x0, y0, x, y, fval):

    x0 = np.atleast_1d(x0)
    y0 = np.atleast_1d(y0)

    out = np.empty_like(x0)

    retval = _interp_bilinear_vec(x0, y0, x, y, fval, out)
    if retval == -1:
        raise ValueError('Dimensions of x, y and fval not conformable')
    elif retval == -2:
        raise ValueError('Arrays x and y must contain at least 2 points!')
    elif retval == -3:
        raise ValueError('Dimensions of x0, y0 and output array not '
                         'conformable!')

    return out


