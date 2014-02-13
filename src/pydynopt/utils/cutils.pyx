cimport numpy as cnp
import numpy as np

import cython

from numpy cimport PyArray_NDIM, PyArray_DIMS, PyArray_Repeat, npy_intp, \
    PyArray_SwapAxes, PyArrayObject

from libc.math cimport floor, fmod, fabs, log, ceil
from libc.stdlib cimport malloc, free
# from libc.math cimport fmax, fmin


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
def _cartesian_cimpl_int64(a_tup, long[:,::1] out):

    cdef unsigned int n_arr, i, ncols
    cdef long[:,::1] c_arr
    n_arr = len(a_tup)

    cdef unsigned int *cumc, *reps
    cdef unsigned int[2] *dims

    cdef unsigned int stride, dst_row, src_col, src_row, dst_col

    with nogil:
        cumc = <unsigned int*> malloc(sizeof(unsigned int) * n_arr)
        reps = <unsigned int*> malloc(sizeof(unsigned int) * n_arr)
        dims = <unsigned int[2] *> malloc(sizeof(unsigned int) * n_arr * 2)

    for i in range(n_arr):
        c_arr = <cnp.ndarray>a_tup[i]
        for j in range(2):
            dims[i][j] = c_arr.shape[j]

    with nogil:
        cumc[0] = dims[0][1]
        for i in range(1, n_arr):
            cumc[i] = cumc[i-1] * dims[i][1]

        ncols = cumc[n_arr-1]
        for i in range(n_arr):
            reps[i] = ncols / cumc[i]

        # index of row in destination array, initialize to zero
        dst_row = 0

    cdef unsigned int n, strd, idx, cval
    cdef unsigned int acols, arows, areps

    cdef long *out_ptr, *in_ptr

    for a in range(n_arr):
        c_arr = <cnp.ndarray>a_tup[a]

        with nogil:
            in_ptr = &c_arr[0, 0]
            arows, acols = dims[a][0], dims[a][1]
            areps = reps[a]
            stride = acols * areps
            # for src_row in range(dim[a, 0]):
            #     for dst_col in range(ncols):
            #         src_col = <unsigned int> cfloor(fmod(dst_col, stride) / reps[a])
            #         out[dst_row, dst_col] = c_arr[src_row, src_col]
            #     dst_row += 1
            for src_row in range(arows):
                out_ptr = &out[dst_row, 0]
                for src_col in range(acols):
                    strd = 0
                    cval = in_ptr[0]
                    while strd < ncols:
                        for n in range(areps):
                            # out[dst_row, src_col + strd + n] = c_arr[src_row, src_col]
                            # out_ptr[dst_row * ncols + strd + n * ] = \
                            #     *(in_ptr + src_row * dim[a, 1] + src_col)
                            idx = strd + n * acols + src_col
                            out_ptr[idx] = cval
                        strd += stride
                    in_ptr += 1
                dst_row += 1

    with nogil:
        free(cumc)
        free(reps)
        free(dims)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _makegrid_mirrored_cimpl(double start, double stop,
                              double around, double [::1] out,
                              bint endpoint=True,
                              bint logs=False, double log_shift=1):

    cdef double adj_start, adj_stop, adj_around, ladj_start, ladj_stop, ladj_around
    cdef double frac
    cdef bint need_ep
    cdef unsigned long adj_num, num_long, num
    cdef unsigned int sidx, i
    cdef double step
    cdef int sgn_long

    adj_start, adj_stop = fabs(start - around), stop - around
    adj_around = 0

    need_ep = (adj_stop > 0 and fabs(adj_start) > 0) and endpoint

    num = <unsigned long> out.shape[0]
    adj_num = num if not need_ep else num - 1

    if logs:
        ladj_start, ladj_stop = log(adj_start + log_shift), log(adj_stop + log_shift)
        ladj_around = log(adj_around + log_shift)
    else:
        ladj_around, ladj_stop, ladj_around = adj_start, adj_stop, adj_around

    frm, to = ladj_around, <double> np.fmax(ladj_stop, ladj_start)

    if adj_start == 0 or adj_stop == 0:
        frac = 1
    else:
        frac = (to - frm) / (ladj_stop + ladj_start - 2 * frm)

    num_long = <unsigned long>np.fmin(ceil(frac * num), <double>adj_num)

    step = (to - frm)/num_long
    sgn_long = 1 if adj_stop > adj_start else -1

    print(frm)
    print(to)
    print(step)
    print(num)
    print(adj_num)
    print(num_long)
    print(sgn_long)


    if sgn_long > 0:
        sidx = num - num_long
        out[sidx] = frm
        for i in range(sidx + 1, num):
            out[i] = out[i-1] + step
        for i in range(1, adj_num - num_long + 1):
            out[sidx-i] = out[sidx - i + 1] - step
        if need_ep:
            out[0] = start
    else:
        sidx = num_long - 1
        out[sidx] = frm
        for i in range(1, num_long):
            out[sidx-i] = out[sidx-i+1] - step
        for i in range(sidx + 1, adj_num):
            out[i] = out[i - 1] + step
        if need_ep:
            out[num - 1] = stop

    return sidx


def _test():

    cdef int[2] *dims

    dims = <int[2] *> malloc(sizeof(int) * 4)

    for i in range(2):
        for j in range(2):
            dims[i][j] = i * 10 + j
            print(dims[i][j])


    free(dims)
