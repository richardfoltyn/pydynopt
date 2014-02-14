cimport numpy as cnp
import numpy as np

import cython

from numpy cimport PyArray_NDIM, PyArray_DIMS, PyArray_Repeat, npy_intp, \
    PyArray_SwapAxes, PyArrayObject

from libc.math cimport floor, fmod, fabs, log, ceil, exp
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def interp_grid_prob_cimpl(double[::1] v, double[::1] g,
                     long[::1] i_low, long[::1] i_high,
                     double[::1] p_low, double[::1] p_high):

    cdef ssize_t len_v, len_g
    cdef long i, j
    cdef double delta

    len_v = v.shape[0]
    len_g = g.shape[0]

    with nogil:
        for i in range(0, len_v):
            for j in range(0, len_g):
                if g[j] >= v[i] or j == (len_g - 1):
                    i_low[i] = _fmaxl(j - 1, 0)
                    i_high[i] = _fmaxl(j, i_low[i] + 1)
                    delta = g[i_high[i]] - g[i_low[i]]
                    if v[i] > g[j]:
                        p_low[i] = 0
                    elif j == 0:
                        p_low[i] = 1
                    else:
                        p_low[i] = (g[i_high[i]] - v[i]) / delta

                    p_high[i] = 1.0 - p_low[i]
                    break

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
cdef void _logshift(double[::1] arr, double shift) nogil:
    for i in range(arr.shape[0]):
        arr[i] = log(arr[i] + shift)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _logshift_inv(double[::1] arr, double shift) nogil:
    for i in range(arr.shape[0]):
        arr[i] = exp(arr[i]) - shift

# Since on Windows 64bit linking against the C library containing fmin and fmax
# does not work, we use these self-written implementations instead.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _fmin(double x, double y) nogil:
    return x if x < y else y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _fmax(double x, double y) nogil:
    return x if x > y else y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef long _fminl(long x, long y) nogil:
    return x if x < y else y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef long _fmaxl(long x, long y) nogil:
    return x if x > y else y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def makegrid_mirrored_cimpl(double start, double stop,
                              double around, double [::1] out,
                              bint endpoint=True,
                              bint logs=False, double log_shift=1):

    cdef double adj_start, adj_stop, adj_around, ladj_start, ladj_stop, ladj_around
    cdef double frac, to, frm, step
    cdef bint need_ep
    cdef unsigned long adj_num, num_long, num
    cdef unsigned int i, ar_idx
    cdef int sgn_long

    with nogil:

        adj_start, adj_stop = fabs(start - around), stop - around
        adj_around = 0

        need_ep = (adj_stop > 0 and fabs(adj_start) > 0) and endpoint

        num = <unsigned long> out.shape[0]
        adj_num = num if not need_ep else num - 1

        if logs:
            ladj_start = log(adj_start + log_shift)
            ladj_stop = log(adj_stop + log_shift)
            ladj_around = log(adj_around + log_shift)
        else:
            ladj_start, ladj_stop, ladj_around = adj_start, adj_stop, adj_around

        frm, to = ladj_around, _fmax(ladj_stop, ladj_start)

        if adj_start == 0 or adj_stop == 0:
            frac = 1.0
        else:
            frac = (to - frm) / (ladj_stop + ladj_start - 2 * frm)

        num_long = <unsigned long> _fmin(ceil(frac * num), adj_num)

        step = (to - frm)/(num_long - 1)
        sgn_long = 1 if adj_stop > adj_start else -1

        if sgn_long > 0:
            ar_idx = num - num_long
            out[ar_idx] = frm
            for i in range(ar_idx + 1, num):
                out[i] = out[i-1] + step

            if logs:
                _logshift_inv(out[ar_idx:], log_shift)

            for i in range(1, ar_idx + 1):
                out[ar_idx-i] = 2 * out[ar_idx] - out[ar_idx + i]

            for i in range(num - 1):
                out[i] = out[i] + around

            out[num-1] = stop
            if need_ep:
                out[0] = start
        else:
            ar_idx = num_long - 1
            out[ar_idx] = frm
            for i in range(1, num_long):
                out[ar_idx-i] = out[ar_idx-i+1] + step

            if logs:
                _logshift_inv(out[:ar_idx+1], log_shift)

            for i in range(1, num_long):
                out[ar_idx-i] = 2 * out[ar_idx] - out[ar_idx-i]

            for i in range(1, num - num_long + 1):
                out[ar_idx + i] = 2 * out[ar_idx] - out[ar_idx - i]

            for i in range(1, num):
                out[i] = out[i] + around

            out[0] = start
            if need_ep:
                out[num - 1] = stop

        # replace around with input value
        out[ar_idx] = around

    return ar_idx


def _test():

    cdef int[2] *dims

    dims = <int[2] *> malloc(sizeof(int) * 4)

    for i in range(2):
        for j in range(2):
            dims[i][j] = i * 10 + j
            print(dims[i][j])


    free(dims)
