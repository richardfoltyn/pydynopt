
from cython import boundscheck

import numpy as np

################################################################################

cdef inline long cy_find_lb(double *xp, double x, unsigned long length) nogil:
    """
    Find lower bound index i on xp such that x <= xp[i] when x is in the
    interval [xp[0],xp[-1])

    Boundary behavior:
        if x <= xp[0], return 0
        if x >= xp[-1], return xp.shape[0] - 2

    In both cases this is the desired behavior as then we need to
    extrapolate, and the slope for extrapolation is given by
        (fp[i+1]-fp[i])/(xp[i+1] - xp[i])
    """

    if x <= xp[0]:
        ixp_lb = 0
    elif x >= xp[length - 1]:
        ixp_lb = length - 2
    else:
        ixp_lb = _bsearch(xp, x, length)

    return ixp_lb

def find_lb(double[::1] xp, double x, unsigned long length):
    """
    Python-callable wrapper for cy_find_lb
    :param xp:
    :param x:
    :param length:
    :return:
    """
    return cy_find_lb(&(xp[0]), x, length)


################################################################################
# 1D linear interpolation

cdef inline double \
    cy_interp1d_linear(double x, double[::1] xp, double[:] fp) nogil:

    cdef double fx, x_lb
    cdef long ixp_lb, ixp_ub
    # interpolation weight
    cdef double slope

    ixp_lb = cy_find_lb(&(xp[0]), x, xp.shape[0])
    ixp_ub = ixp_lb + 1

    slope = (x - xp[ixp_lb]) / (xp[ixp_ub] - xp[ixp_lb])
    fx = (1 - slope) * fp[ixp_lb] + slope * fp[ixp_ub]

    return fx


def interp1d_linear(x, double[::1] xp, double[:] fp, out=None):
    """
    Compute the linearly interpolated values as x, given interpolations nodes xp
    with function values fp.
    """

    # Do some input argument error checking
    if xp.shape[0] != fp.shape[0]:
        raise ValueError("Non-conformable arguments 'xp' and 'fp'")

    if xp.shape[0] < 2:
        raise ValueError('Need at least 2 points to interpolate')

    cdef bint return_scalar = True
    x_arr = np.asarray(x)
    if isinstance(x, np.ndarray) or out is not None:
        return_scalar = False

    # ravel() does not copy data if array is C-contiguous (or 1-d),
    # but otherwise allocated memory!
    cdef double[::1] mv_x = x_arr.ravel()

    cdef double[::1] mv_out
    if out is None:
        out = np.empty(x_arr.shape, dtype=np.float)
        mv_out = out.ravel()
    else:
        if not isinstance(out, np.ndarray):
            raise ValueError('Argument out must be a ndarray')

        mv_out = out.ravel()
        if mv_out.shape[0] != mv_x.shape[0]:
            raise ValueError('Arguments \'x\' and \'out\' not conformable!')


    cdef unsigned long i
    for i in range(mv_x.shape[0]):
        mv_out[i] = cy_interp1d_linear(mv_x[i], xp, fp)

    if return_scalar:
        return mv_out[0]
    else:
        return out

################################################################################
# Bilinear interpolation

cdef int _interp2d_bilinear(double[:] x0, double[:] y0,
        double[::1] x, double[::1] y,
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


cdef inline double _interp2d_bilinear_impl(double x, double y, double[::1] xp,
            double[::1] yp, double[:, :] fval) nogil:
    """

    Note on implementation: This could alternatively be implemented as a
    composition of linear interpolations as
    f(x, y) = l(l(x,y0),l(x, y1))
    where y0 and y1 are the lower and upper bounds of y and l() is a 1D
    linear interpolation.
    However, then both
    l(x, y0) and l(x,y1) would find in bounding interval x0 <= x <= x1, thus
    performing a binary search twice!

    Hence we implement bilinear interpolation from scratch here and make sure that
    a binary search is performed only once for each dimension.
    """

    cdef double x_lb, x_ub, y_lb, y_ub
    cdef long ix_lb, iy_lb

    # interpolation weights in x and y direction
    cdef double xwgt, ywgt
    # interpolants in x direction evaluated at lower and upper y
    cdef double fx1, fx2

    ix_lb = cy_find_lb(&(xp[0]), x, xp.shape[0])
    iy_lb = cy_find_lb(&(yp[0]), y, yp.shape[0])

    # lower and upper bounding indexes in x direction
    x_lb = xp[ix_lb]
    x_ub = xp[ix_lb + 1]

    # lower and upper bounding indexes in y direction
    y_lb = yp[iy_lb]
    y_ub = yp[iy_lb + 1]

    xwgt = (x - x_lb) / (x_ub - x_lb)
    fx1 = (1-xwgt) * fval[ix_lb, iy_lb] + xwgt * fval[ix_lb + 1, iy_lb]
    fx2 = (1-xwgt) * fval[ix_lb, iy_lb + 1] + xwgt * fval[ix_lb + 1, iy_lb + 1]

    ywgt = (y - y_lb) / (y_ub - y_lb)

    return (1-ywgt) * fx1 + ywgt * fx2


################################################################################
# Trilinear interpolation

cdef inline double _interp3d_trilinear_impl(double x, double y, double z,
        double[::1] xp, double[::1] yp, double[::1] zp,
        double[:, :, :] fp) nogil:

    """
    Note on implementation: We implement this from scratch instead of using a
    composition of linear and bilinear interpolations for the same reasons as
    mention in the bilinear interpolation.
    """

    cdef long ix_lb, iy_lb, iz_lb

    # slopes in x, y and z direction
    cdef double slope_x, slope_y, slope_z
    # interpolants in x direction evaluated at (y0, z0), ..., (y1, z1)
    cdef double fx00, fx01, fx10, fx11
    # interpolants in xy direction evaluated at z0 and z1
    cdef double fxy0, fxy1
    # interpolated function value in all directions
    cdef double fxyz

    ix_lb = cy_find_lb(&(xp[0]), x, xp.shape[0])
    iy_lb = cy_find_lb(&(yp[0]), y, yp.shape[0])
    iz_lb = cy_find_lb(&(zp[0]), z, zp.shape[0])

    # interpolating in x dimension
    slope_x = (x - xp[ix_lb]) / (xp[ix_lb + 1] - xp[ix_lb])

    fx00 = (1-slope_x) * fp[ix_lb, iy_lb, iz_lb] + \
          slope_x * fp[ix_lb + 1, iy_lb, iz_lb]

    fx10 = (1-slope_x) * fp[ix_lb, iy_lb + 1, iz_lb] + \
          slope_x * fp[ix_lb + 1, iy_lb + 1, iz_lb]

    fx01 = (1-slope_x) * fp[ix_lb, iy_lb, iz_lb + 1] + \
            slope_x * fp[ix_lb + 1, iy_lb, iz_lb + 1]

    fx11 = (1-slope_x) * fp[ix_lb, iy_lb + 1, iz_lb + 1] + \
            slope_x * fp[ix_lb + 1, iy_lb + 1, iz_lb + 1]

    # interpolate in y dimension
    slope_y = (y - yp[iy_lb]) / (yp[iy_lb + 1] - yp[iy_lb])

    fxy0 = (1 - slope_y) * fx00 + slope_y * fx10
    fxy1 = (1 - slope_y) * fx01 + slope_y * fx11

    # interpolate in z dimension
    slope_z = (z - zp[iz_lb]) / (zp[iz_lb + 1] - zp[iz_lb])

    fxyz = (1 - slope_z) * fxy0 + slope_z * fxy1

    return fxyz


cdef int _interp3d_trilinear(double[:] x, double[:] y, double[:] z,
        double[::1] xp, double[::1] yp, double[::1] zp,
        double[:, :, :] fp, double[:] out) nogil:

    # Some sanity checks: make sure dimension of input and output arrays
    # match. We also require at least 2 points in each direction on the
    # domain of f().
    if xp.shape[0] != fp.shape[0] or yp.shape[0] != fp.shape[1] \
        or zp.shape[0] != fp.shape[2]:
        return -1
    if xp.shape[0] < 2 or yp.shape[0] < 2 or zp.shape[0] < 2:
        return -2
    if x.shape[0] != y.shape[0] or x.shape[0] != z.shape[0] or \
        x.shape[0] != out.shape[0]:
        return -3

    cdef unsigned long i
    for i in range(x.shape[0]):

        out[i] = _interp3d_trilinear_impl(x[i], y[i], z[i], xp, yp, zp, fp)

    return 0



################################################################################
# Python-callable convenience functions

@boundscheck(True)
def interp2d_bilinear(double[:] x0, double[:] y0, double[::1] x, double[::1] y,
                      double[:,:] fval, double[:] out=None):


    cdef unsigned long nx = x0.shape[0]
    if out is None:
        out = make_ndarray(1, nx, <double>x0[0])

    retval = _interp2d_bilinear(x0, y0, x, y, fval, out)
    if retval == -1:
        raise ValueError('Dimensions of x, y and fval not conformable')
    elif retval == -2:
        raise ValueError('Arrays x and y must contain at least 2 points!')
    elif retval == -3:
        raise ValueError('Dimensions of x0, y0 and output array not '
                         'conformable!')

    return np.asarray(out)

@boundscheck(True)
def interp3d_trilinear(double[:] x, double[:] y, double[:] z,
            double[::1] xp,  double[::1] yp, double[::1] zp,
            double[:,:,:] fp, double[:] out=None):


    cdef unsigned long nx = x.shape[0]

    if out is None:
        out = make_ndarray(1, nx, <double>x[0])

    retval = _interp3d_trilinear(x, y, z, xp, yp, zp, fp, out)
    if retval == -1:
        raise ValueError('Dimensions of xp, yp, zp and fp not conformable')
    elif retval == -2:
        raise ValueError('Arrays xp, yp and zp must contain at least 2 points!')
    elif retval == -3:
        raise ValueError('Dimensions of x, y, z and output array not '
                         'conformable!')

    return np.asarray(out)


