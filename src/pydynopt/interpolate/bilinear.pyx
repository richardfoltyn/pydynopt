
from cython import boundscheck, wraparound, cdivision

import numpy as np

from ..utils.bsearch cimport _bsearch
from ..common.types cimport real_t
from ..common.ndarray_wrappers cimport make_ndarray

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef int _interp2d_bilinear(real_t[:] x0, real_t[:] y0,
        real_t[:] x, real_t[:] y,
        real_t[:, :] fval, real_t[:] out) nogil:

    # Some sanity checks: make sure dimension of input and output arrays 
    # match. We also require at least 2 points in each direction on the 
    # domain of f().
    if x.shape[0] != fval.shape[0] or y.shape[0] != fval.shape[1]:
        return -1
    if x.shape[0] < 2 or y.shape[0] < 2:
        return -2
    if x0.shape[0] != y0.shape[0] != out.shape[0]:
        return -3

    # for each (x_i, y_i) combination where we compute interpolation,
    # we first need to identify bounding rectangle with indexes ix_lb,
    # iy_lb such that x0[ix_lb] <= xi < x0[ix_lb + 1] and
    # y0[iy_lb] <= yi < y0[iy_lb + 1].
    # Special care needs to be taken if xi or yi falls outside of the domain
    # defined by x and y arrays. In that case we perform linear extrapolation.

    cdef unsigned long i
    for i in range(x0.shape[0]):

        out[i] = _interp2d_bilinear_impl(x0[i], y0[i], x, y, fval)

    return 0

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline real_t _interp2d_bilinear_impl(real_t x, real_t y, real_t[:] xp, 
            real_t[:] yp, real_t[:, :] fval) nogil:
    
    # store last valid indexes in x and y direction
    cdef long ix_last = xp.shape[0] - 1, iy_last = yp.shape[0] - 1

    cdef real_t x_lb, x_ub, y_lb, y_ub
    cdef long ix_lb, iy_lb

    # interpolation weights in x and y direction
    cdef real_t xwgt, ywgt
    # interpolants in x direction evaluated at lower and upper y
    cdef real_t fx1, fx2
    
    if x <= xp[0]:
        ix_lb = 0
    elif x >= xp[ix_last]:
        ix_lb = ix_last - 1
    else:
        ix_lb = _bsearch(xp, x)

    # lower and upper bounding indexes in x direction
    x_lb = xp[ix_lb]
    x_ub = xp[ix_lb + 1]

    if y <= yp[0]:
        iy_lb = 0
    elif y >= yp[iy_last]:
        iy_lb = iy_last - 1
    else:
        iy_lb = _bsearch(yp, y)

    # lower and upper bounding indexes in y direction
    y_lb = yp[iy_lb]
    y_ub = yp[iy_lb + 1]

    xwgt = (x - x_lb) / (x_ub - x_lb)
    fx1 = (1-xwgt) * fval[ix_lb, iy_lb] + xwgt * fval[ix_lb + 1, iy_lb]
    fx2 = (1-xwgt) * fval[ix_lb, iy_lb + 1] + xwgt * fval[ix_lb + 1, iy_lb + 1]

    ywgt = (y - y_lb) / (y_ub - y_lb)
    
    return (1-ywgt) * fx1 + ywgt * fx2


def interp2d_bilinear(real_t[:] x0, real_t[:] y0, real_t[:] x, real_t[:] y,
                      real_t[:,:] fval, real_t[:] out=None):


    cdef unsigned long nx = x0.shape[0]
    if out is None:
        out = make_ndarray(1, nx, <real_t>x0[0])

    retval = _interp2d_bilinear(x0, y0, x, y, fval, out)
    if retval == -1:
        raise ValueError('Dimensions of x, y and fval not conformable')
    elif retval == -2:
        raise ValueError('Arrays x and y must contain at least 2 points!')
    elif retval == -3:
        raise ValueError('Dimensions of x0, y0 and output array not '
                         'conformable!')

    return np.asarray(out)


