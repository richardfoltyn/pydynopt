
from cython import boundscheck, wraparound, cdivision

import numpy as np

from ..utils.bsearch cimport _bsearch_impl

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef int _interp2d_bilinear_vec(double[:] x0, double[:] y0,
        double[:] x, double[:] y,
        double[:, :] fval, double[:] out) nogil:

    # Some sanity checks: make sure dimension of input and output arrays 
    # match. We also require at least 2 points in each direction on the 
    # domain of f().
    if x.shape[0] != fval.shape[0] or y.shape[0] != fval.shape[1]:
        return -1
    if x.shape[0] < 2 or y.shape[0] < 2:
        return -2
    if x0.shape[0] != y0.shape[0] != out.shape[0]:
        return -3

    # store last valid indexes in x and y direction
    cdef unsigned long ix_last = x.shape[0] - 1, iy_last = y.shape[0] - 1
    # define additional paramaters for _bsearch_impl: starting index and 
    # first=1 so that the first index with arr[idx] <= key is returned. 
    cdef unsigned long ifrom = 0
    cdef bint first = 1

    cdef double x_lb, x_ub, y_lb, y_ub
    cdef unsigned long ix_lb, iy_lb

    cdef double xi, yi
    # interpolation weights in x and y direction
    cdef double xwgt, ywgt
    # interpolants in x direction evaluated at lower and upper y
    cdef double fx1, fx2

    # for each (x_i, y_i) combination where we compute interpolation,
    # we first need to identify bounding rectangle with indexes ix_lb,
    # iy_lb such that x0[ix_lb] <= xi < x0[ix_lb + 1] and
    # y0[iy_lb] <= yi < y0[iy_lb + 1].
    # Special care needs to be taken if xi or yi falls outside of the domain
    # defined by x and y arrays. In that case we perform linear extrapolation.

    for i in range(x0.shape[0]):

        xi, yi = x0[i], y0[i]

        if xi <= x[0]:
            ix_lb = 0
        elif xi >= x[ix_last]:
            ix_lb = ix_last - 1
        else:
            ix_lb = _bsearch_impl(x, xi, ifrom, ix_last, first)

        # lower and upper bounding indexes in x direction
        x_lb = x[ix_lb]
        x_ub = x[ix_lb + 1]

        if yi <= y[0]:
            iy_lb = 0
        elif yi >= y[iy_last]:
            iy_lb = iy_last - 1
        else:
            iy_lb = _bsearch_impl(y, yi, ifrom, iy_last, first)

        # lower and upper bounding indexes in y direction
        y_lb = y[iy_lb]
        y_ub = y[iy_lb + 1]

        xwgt = (xi - x_lb) / (x_ub - x_lb)
        fx1 = (1-xwgt) * fval[ix_lb, iy_lb] + xwgt * fval[ix_lb + 1, iy_lb]
        fx2 = (1-xwgt) * fval[ix_lb, iy_lb + 1] + xwgt * fval[ix_lb + 1, iy_lb + 1]

        ywgt = (yi - y_lb) / (y_ub - y_lb)
        out[i] = (1-ywgt) * fx1 + ywgt * fx2

    return 0


cdef int _interp2d_bilinear(double x0, double y0, double[:] x, double[:] y,
        double[:, :] fval, double *out):

    cdef double[:] xmv, ymv
    cdef double[:] outmv

    xmv = <double[:1]>&x0
    ymv = <double[:1]>&y0
    outmv = <double[:1]>out

    return _interp2d_bilinear_vec(xmv, ymv, x, y, fval, outmv)


def interp2d_bilinear(x0, y0, x, y, fval):

    x0 = np.atleast_1d(x0)
    y0 = np.atleast_1d(y0)

    out = np.empty_like(x0)

    retval = _interp2d_bilinear_vec(x0, y0, x, y, fval, out)
    if retval == -1:
        raise ValueError('Dimensions of x, y and fval not conformable')
    elif retval == -2:
        raise ValueError('Arrays x and y must contain at least 2 points!')
    elif retval == -3:
        raise ValueError('Dimensions of x0, y0 and output array not '
                         'conformable!')

    return out


