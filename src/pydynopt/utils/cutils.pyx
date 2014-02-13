cimport numpy as cnp
import numpy as np

import cython

from numpy cimport PyArray_NDIM, PyArray_DIMS, PyArray_Repeat, npy_intp, \
    PyArray_SwapAxes, PyArrayObject

from libc.math cimport floor as cfloor, fmod
from libc.stdlib cimport malloc, free

ctypedef cnp.uint64_t NP_UINT_t

def _interp_grid_prob(cnp.ndarray[cnp.float_t, ndim=1] v,
                      cnp.ndarray[cnp.float_t, ndim=1] g):

    cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] i_low, i_high
    cdef cnp.ndarray[cnp.int_t, ndim=1] fp
    cdef cnp.ndarray[cnp.float_t, ndim=1] p_low, p_high

    cdef unsigned int max_idx


    i_low = cnp.ascontiguousarray(cnp.fmax(np.searchsorted(g, v, side='right')-
                                         1, 0), dtype=np.int_)
    max_idx = g.shape[0] - 1
    fp = cnp.arange(max_idx + 1)
    p_high = cnp.interp(v, g, fp) - fp[i_low]

    assert cnp.all(p_high >= 0) and cnp.all(p_high <= 1)

    i_high = cnp.where(i_low < max_idx, i_low + 1, i_low)
    p_low = 1 - p_high

    return i_low, i_high, p_low, p_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _cartesian_cimpl_int64(a_tup, long[:,::1] out, unsigned int[:,::1] dim):

    cdef unsigned int n_arr, i, ncols
    cdef long[:,::1] c_arr
    n_arr = len(a_tup)

    cdef unsigned int *cumc, *reps

    cdef unsigned int nrepcol, dst_row, src_col, src_row, dst_col

    with nogil:
        cumc = <unsigned int*> malloc(sizeof(unsigned int) * n_arr)
        reps = <unsigned int*> malloc(sizeof(unsigned int) * n_arr)

        cumc[0] = dim[0, 1]

        for i in range(1, n_arr):
            cumc[i] = cumc[i-1] * dim[i, 1]

        for i in range(n_arr):
            reps[i] = cumc[n_arr - 1] / cumc[i]

        ncols = cumc[n_arr-1]
        # index of row in destination array, initialize to zero
        dst_row = 0

    cdef unsigned int n, strd = 0

    cdef long *out_ptr = &out[0, 0]
    cdef long *in_ptr

    for a in range(n_arr):
        c_arr = <cnp.ndarray>a_tup[a]
        in_ptr = &c_arr[0, 0]

        with nogil:
            nrepcol = dim[a, 1] * reps[a]
            # for src_row in range(dim[a, 0]):
            #     for dst_col in range(ncols):
            #         src_col = <unsigned int> cfloor(fmod(dst_col, nrepcol) / reps[a])
            #         out[dst_row, dst_col] = c_arr[src_row, src_col]
            #     dst_row += 1
            for src_row in range(dim[a, 0]):
                for src_col in range(0, dim[a, 1]):
                    while strd < ncols:
                        for n in range(0, reps[a]):
                            # out[dst_row, src_col + strd + n] = c_arr[src_row, src_col]
                            out_ptr[dst_row * ncols + src_col + strd + n] = \
                                in_ptr[src_row * dim[a, 1] + src_col]
                        strd += strd
                dst_row += 1

    with nogil:
        free(cumc)
        free(reps)

def _test(object a_tup):

    cdef long[:,::1] arr

    arr = <cnp.ndarray>a_tup[0]

    arr[0,0] = 1000

