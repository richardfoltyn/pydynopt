
from cython import boundscheck, wraparound, cdivision

import numpy as np

from ..utils.bsearch cimport _bsearch_impl

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef int _interp1d_linear_vec(double[:] x0, double[:] xp,
        double[:] fp, double[:] out) nogil:

    # Some sanity checks: make sure dimension of input and output arrays 
    # match. We also require at least 2 points in each direction on the 
    # domain of f().
    if xp.shape[0] != fp.shape[0]:
        return -1
    if xp.shape[0] < 2:
        return -2
    if x0.shape[0] != out.shape[0]:
        return -3

    # store last valid indexes
    cdef unsigned long ix_last = xp.shape[0] - 1
    # define additional paramaters for _bsearch_impl: starting index and 
    # first=1 so that the first index with arr[idx] <= key is returned. 
    cdef unsigned long ifrom = 0
    cdef bint first = 1

    cdef double x_lb, x_ub,
    cdef unsigned long ix_lb

    cdef double xi
    # interpolation weights in x and y direction
    cdef double wgt

    for i in range(x0.shape[0]):

        xi = x0[i]

        if xi <= xp[0]:
            ix_lb = 0
        elif xi >= xp[ix_last]:
            ix_lb = ix_last - 1
        else:
            ix_lb = _bsearch_impl(xp, xi, ifrom, ix_last, first)

        # lower and upper bounding indexes in x direction
        x_lb = xp[ix_lb]
        x_ub = xp[ix_lb + 1]

        wgt = (xi - x_lb) / (x_ub - x_lb)
        out[i] = (1-wgt) * fp[ix_lb] + wgt * fp[ix_lb + 1]

    return 0


cdef int _interp1d_linear(double x, double[:] xp, double[:] fp, double *out):

    cdef double[:] xmv
    cdef double[:] outmv

    xmv = <double[:1]>&x
    outmv = <double[:1]>out

    return _interp1d_linear_vec(xmv, xp, fp, outmv)


def interp1d_linear(x, xp, fp):

    x = np.atleast_1d(x)
    out = np.empty_like(x)

    retval = _interp1d_linear_vec(x, xp, fp, out)
    if retval == -1:
        raise ValueError('Dimensions of xp and fp not conformable!')
    elif retval == -2:
        raise ValueError('Arrays xp must contain at least 2 points!')
    elif retval == -3:
        raise ValueError('Dimensions of x0 and output array not '
                         'conformable!')

    return out


