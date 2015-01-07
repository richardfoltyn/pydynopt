from cython import boundscheck, wraparound

cimport numpy as cnp
import numpy as np
from cython cimport numeric
from ..common.ndarray_wrappers cimport make_ndarray

from ..common.types cimport numc2d_t, numc1d_t

@boundscheck(False)
@wraparound(False)
cpdef numc2d_t __cartesian2d(numc2d_t a, numc2d_t b, numc2d_t out):

    cdef unsigned int nr_out, nc_out
    nr_out = a.shape[0] + b.shape[0]
    nc_out = a.shape[1] + b.shape[1]

    cdef bint transpose = False

    if nc_out == out.shape[1]:
        transpose = True
        a, b, out = a.T, b.T, out.T

    cdef unsigned int nr_a = a.shape[0], nc_a = a.shape[1]
    cdef unsigned int nr_b = b.shape[0], nc_b = b.shape[1]

    assert out.shape[1] == nc_a * nc_b

    # For whatever reason, assigning a[:, j] to blocks in out generates a lot
    # of CPYTHON API calls, so loop through all rows manually.
    cdef unsigned int i, j, k
    for j in range(nc_a):
        for k in range(nc_b):
            for i in range(nr_a):
                out[i, j * nc_b + k] = a[i, j]

    cdef unsigned int j_start, j_end
    for k in range(nc_a):
        j_start, j_end = k * nc_b, (k + 1) * nc_b 
        out[nr_a:, j_start:j_end] = b

    if transpose:
        a, b, out = a.T, b.T, out.T

    return out


@boundscheck(False)
@wraparound(False)
def cartesian2d(numc2d_t a, numc2d_t b, numc2d_t out=None):

    cdef unsigned int nr_a, nr_b, nc_a, nc_b, nr_out, nc_out
    nr_a, nc_a = a.shape[0], a.shape[1]
    nr_b, nc_b = b.shape[0], b.shape[1]

    nr_out = nr_a + nr_b
    nc_out = nc_a * nc_b

    cdef numc2d_t _out

    if out is None:
        _out = make_ndarray(nr_out, nc_out, <numeric>a[0,0])

    return np.asarray(__cartesian2d(a, b, _out))

def cartesian(numc1d_t a, numc1d_t b, numc2d_t out=None):

    return cartesian2d(<numc2d_t>a[None, :], <numc2d_t>b[None, :], out)

