cimport numpy as cnp
import numpy as np

import cython

from numpy cimport PyArray_NDIM, PyArray_DIMS, PyArray_Repeat, npy_intp, \
    PyArray_SwapAxes, PyArrayObject

from libc.math cimport floor as cfloor
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
def _cartesian_cimpl_int(a_tup, cnp.ndarray out, unsigned int[:,:] dim):

    cdef unsigned int n_arr, i
    cdef int[:,:] c_arr
    n_arr = len(a_tup)

    cdef unsigned int *cumc, *reps, *tiles, *crd

    with nogil:

        cumc = <unsigned int*> malloc(sizeof(unsigned int) * n_arr)
        crd = <unsigned int*> malloc(sizeof(unsigned int) * (n_arr + 1))
        tiles = <unsigned int*> malloc(sizeof(unsigned int) * n_arr)
        reps = <unsigned int*> malloc(sizeof(unsigned int) * n_arr)

        crd[0] = 0
        cumc[0] = dim[0, 1]
        tiles[0] = 1

        for i in range(1, n_arr):
            cumc[i] = cumc[i-1] * dim[i, 1]
            crd[i] = crd[i-1] + dim[i, 0]
            tiles[i] = cumc[i-1]

        crd[n_arr] = crd[n_arr - 1] + dim[n_arr - 1, 0]

        for i in range(n_arr):
            reps[i] = cumc[n_arr - 1] / cumc[i]

    cdef unsigned int nrepcol
    cdef unsigned int row_offset

    row_offset = 0

    for a in range(n_arr):
        c_arr = <cnp.ndarray>a_tup[a]
        nrepcol = dim[a, 1] * reps[a]
        for i in range(dim[a, 0]):


            row_offset += 1


    for i in range(n_arr):
        out[crd[i]:crd[i+1]] = \
            np.tile(np.repeat(a_tup[i], reps[i], axis=1), (1, tiles[i]))

    free(cumc)
    free(reps)
    free(tiles)
    free(crd)

