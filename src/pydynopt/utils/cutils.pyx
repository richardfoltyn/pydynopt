cimport numpy as cnp
import numpy as np

from numpy cimport PyArray_NDIM, PyArray_DIMS, PyArray_Repeat, npy_intp, \
    PyArray_SwapAxes, PyArrayObject

from libc.math cimport floor as cfloor

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

cpdef _cartesian_cimpl(cnp.ndarray arr1, cnp.ndarray arr2, cnp.ndarray out, int axis):
    cdef unsigned int n_arr = 2
    cdef cnp.ndarray[cnp.uint_t, ndim=2] dim
    cdef cnp.ndarray[cnp.uint_t, ndim=1] cumc, crd, reps, tiles

    cdef cnp.ndarray arr1_2d, arr2_2d

    dim = np.empty((n_arr, 2), dtype=np.uint)

    arr1_2d = np.atleast_2d(arr1).swapaxes(axis, 0)
    arr2_2d = np.atleast_2d(arr2).swapaxes(axis, 0)

    for j in range(2):
        dim[0, j] = arr1_2d.shape[j]
        dim[0, j] = arr2_2d.shape[j]

    cumc = np.cumprod(dim[:, 1], axis=0)
    crd = np.zeros((n_arr + 1, ), dtype=np.uint32)
    crd[1:] = np.cumsum(dim[:, 0])
    reps = np.ones_like(cumc, dtype=np.uint32)
    tiles = np.ones_like(reps)
    reps = cumc[-1] / cumc
    tiles[1:] = cumc[:-1]

    out[crd[0]:crd[1]] = np.tile(np.repeat(arr1_2d, reps[0], axis=1), (1, tiles[0]))
    out[crd[1]:crd[2]] = np.tile(np.repeat(arr2_2d, reps[1], axis=1), (1, tiles[1]))

    out[...] = out.swapaxes(axis, 0)

cpdef cython_test1(cnp.ndarray arr_in, cnp.ndarray arr_out, int axis=0):
    cdef cnp.ndarray in_2d
    cdef int ndim_in, ndim_out, cols_out, nrep

    cdef npy_intp *dims_out

    ndim_in = PyArray_NDIM(arr_in)
    ndim_out = PyArray_NDIM(arr_out)
    assert ndim_in == 2 and ndim_out == 2

    dims_out = PyArray_DIMS(arr_out)
    dims_in = PyArray_DIMS(arr_in)

    if axis == 0:
        nrep = <int>cfloor(dims_out[1] / dims_in[1])
    else:
        nrep = <int>cfloor(dims_out[0] / dims_in[0])

    arr_out[...] = np.repeat(arr_in, nrep, axis=(1-axis))


cpdef cython_test2(cnp.ndarray arr1, cnp.ndarray arr2, cnp.ndarray arr_out, int axis=0):

    cdef int ndim1, ndim2
    cdef PyArrayObject *arr1_obj, *arr2_obj

    ndim1 = PyArray_NDIM(arr1)
    ndim2 = PyArray_NDIM(arr2)

    assert ndim1 == 2 and ndim2 == 2

