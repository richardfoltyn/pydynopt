
from cython import boundscheck, wraparound, cdivision

import numpy as np

from ..common.types cimport real_t
from ..utils.bsearch cimport _bsearch

from ..common.ndarray_wrappers cimport make_ndarray

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef int _interp1d_linear_vec(real_t[:] x, real_t[:] xp,
        real_t[:] fp, real_t[:] out) nogil:

    # Some sanity checks: make sure dimension of input and output arrays 
    # match. We also require at least 2 points in each direction on the 
    # domain of f().
    if xp.shape[0] != fp.shape[0]:
        return -1
    if xp.shape[0] < 2:
        return -2
    if x.shape[0] != out.shape[0]:
        return -3

    cdef unsigned long i
    cdef real_t xi

    for i in range(x.shape[0]):

        xi = x[i]
        out[i] = _interp1d_linear_impl(xi, xp, fp)

    return 0

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef inline real_t _interp1d_linear_impl(real_t x, real_t[:] xp,
                                         real_t[:] fp) nogil:

    cdef long ixp_last = xp.shape[0] - 1
    cdef real_t fx, x_lb, x_ub
    cdef long ix_lb
    # interpolation weight
    cdef real_t wgt

    if x <= xp[0]:
        ix_lb = 0
    elif x >= xp[ixp_last]:
        ix_lb = ixp_last - 1
    else:
        ix_lb = _bsearch(xp, x)

    # lower and upper bounding indexes
    x_lb = xp[ix_lb]
    x_ub = xp[ix_lb + 1]

    wgt = (x - x_lb) / (x_ub - x_lb)
    fx = (1-wgt) * fp[ix_lb] + wgt * fp[ix_lb + 1]

    return fx


cdef int _interp1d_linear(real_t x, real_t[:] xp, real_t[:] fp, real_t *out):

    cdef real_t[:] xmv
    cdef real_t[:] outmv

    xmv = <real_t[:1]>&x
    outmv = <real_t[:1]>out

    return _interp1d_linear_vec(xmv, xp, fp, outmv)


def interp1d_linear(real_t[:] x, real_t[:] xp, real_t[:] fp,
        real_t[:] out=None):
    """
    Compute the linearly interpolated values as x, given interpolations nodes xp
    with function values fp.

    Python-friendly wrapper for _interp1d_linear_vec()
    """

    cdef unsigned long nx = x.shape[0]
    if out is None:
        out = make_ndarray(1, nx, <real_t>x[0])

    retval = _interp1d_linear_vec(x, xp, fp, out)
    if retval == -1:
        raise ValueError('Dimensions of xp and fp not conformable!')
    elif retval == -2:
        raise ValueError('Arrays xp must contain at least 2 points!')
    elif retval == -3:
        raise ValueError('Dimensions of x and output array not '
                         'conformable!')

    return np.asarray(out)


