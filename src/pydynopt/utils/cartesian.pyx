
from cython import boundscheck

cimport numpy as cnp
import numpy as np

cpdef int _cartesian2d(num_arr2d_t a, num_arr2d_t b, num_arr2d_t out) nogil:
    """
    Create cartesian product of two arrays by repeating array `a`
    along axis=1 b.shape[1] times, and tiling array `b` along axis=1
    a.shape[1] times.
    """

    cdef bint transpose = False

    if a.shape[0] * b.shape[0] == out.shape[0] and \
                            a.shape[1] * b.shape[1] != out.shape[1]:
        transpose = True
        a, b, out = a.T, b.T, out.T

    cdef unsigned int nr_a = a.shape[0], nc_a = a.shape[1]
    cdef unsigned int nr_b = b.shape[0], nc_b = b.shape[1]
    cdef unsigned int nr_out = out.shape[0], nc_out = out.shape[1]

    if nr_out != (nr_a + nr_b) or nc_out != (nc_a * nc_b):
        return -1

    cdef unsigned int r_idx, i, j, k, stride
    # cdef numeric ak, bk
    cdef int_real_t ak, bk

    for i in range(nr_a):
        for k in range(nc_a):
            ak = a[i, k]
            stride = k * nc_b
            for j in range(nc_b):
                out[i, j + stride] = ak

    for i in range(nr_a, nr_out):
        r_idx = i - nr_a
        for k in range(nc_b):
            bk = b[r_idx, k]
            for j in range(nc_a):
                out[i, k + j * nc_b] = bk

    if transpose:
        a, b, out = a.T, b.T, out.T

    return 0


def cartesian2d(num_arr2d_t a, num_arr2d_t b, num_arr2d_t out=None):

    cdef unsigned int nr_a, nr_b, nc_a, nc_b, nr_out, nc_out
    nr_a, nc_a = a.shape[0], a.shape[1]
    nr_b, nc_b = b.shape[0], b.shape[1]

    nr_out = nr_a + nr_b
    nc_out = nc_a * nc_b

    cdef bint force_ndarray = False

    if out is None:
        out = make_ndarray(nr_out, nc_out, <int_real_t>a[0,0])
        force_ndarray = True

    retval = _cartesian2d(a, b, out)
    if retval == -1:
        raise ValueError('Incompatible input/output array dimensions!')

    if force_ndarray:
        return np.asarray(out)
    else:
        return out

@boundscheck(True)
def cartesian(num_arr1d_t a, num_arr1d_t b, num_arr2d_t out=None):

    return cartesian2d(<num_arr2d_t>a[None, :], <num_arr2d_t>b[None, :], out)

