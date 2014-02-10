cimport numpy as np
import numpy as np

ctypedef np.uint64_t NP_UINT_t

def _interp_grid_prob(np.ndarray[np.float_t, ndim=1] v,
                      np.ndarray[np.float_t, ndim=1] g):

    cdef np.ndarray[np.int_t, ndim=1, mode='c'] i_low, i_high
    cdef np.ndarray[np.int_t, ndim=1] fp
    cdef np.ndarray[np.float_t, ndim=1] p_low, p_high

    cdef unsigned int max_idx


    i_low = np.ascontiguousarray(np.fmax(np.searchsorted(g, v, side='right')-
                                         1, 0), dtype=np.int_)
    max_idx = g.shape[0] - 1
    fp = np.arange(max_idx + 1)
    p_high = np.interp(v, g, fp) - fp[i_low]

    assert np.all(p_high >= 0) and np.all(p_high <= 1)

    i_high = np.where(i_low < max_idx, i_low + 1, i_low)
    p_low = 1 - p_high

    return i_low, i_high, p_low, p_high

def cartesian_op(a_tup, int axis=0, op=None):
    # assert axis <= 1

    cdef np.ndarray[np.uint_t, ndim=2] dim
    cdef np.ndarray[np.uint_t, ndim=1] cumc, crd, reps, tiles
    cdef np.ndarray[np.float64_t, ndim=2] out
    cdef np.ndarray[object, ndim=1] in_arrays
    cdef unsigned int na

    na = len(a_tup)
    dim = np.empty((na, 2), dtype=np.uint)
    in_arrays = np.empty((na, ), dtype=object)

    for (i, v) in enumerate(a_tup):
        in_arrays[i] = np.atleast_2d(v).swapaxes(axis, 0)
        dim[i] = np.asarray(in_arrays[i].shape, dtype=np.uint)

    cumc = np.cumprod(dim[:, 1], axis=0)
    crd = np.zeros((dim.shape[0] + 1, ), dtype=np.uint)
    crd[1:] = np.cumsum(dim[:, 0])
    reps = np.ones_like(cumc, dtype=np.uint)
    tiles = np.ones_like(reps)
    reps = cumc[-1] / cumc
    tiles[1:] = cumc[:-1]

    out = np.empty((crd[-1], cumc[-1]))

    for (i, v) in enumerate(in_arrays):
        out[crd[i]:crd[i + 1]] = np.tile(v.repeat(reps[i], axis=1), (1,
                                                                     tiles[i]))

    out = out.swapaxes(axis, 0)

    if op is not None:
        out = np.atleast_2d(op(out, axis=axis)).swapaxes(0, axis)

    return out

def _c_cartesian_int64(a_tup, int axis):
    cdef np.ndarray[np.uint_t, ndim=2] dim
    cdef np.ndarray[np.uint_t, ndim=1] cumc, crd, reps, tiles
    cdef np.ndarray[np.int64_t, ndim=2] out
    cdef np.ndarray[object, ndim=1] in_arrays
    cdef unsigned int na

    na = len(a_tup)
    dim = np.empty((na, 2), dtype=np.uint32)
    in_arrays = np.empty((na, ), dtype=object)

    for (i, v) in enumerate(a_tup):
        in_arrays[i] = np.atleast_2d(v).swapaxes(axis, 0)
        dim[i] = in_arrays[i].shape

    cumc = np.cumprod(dim[:, 1], axis=0)
    crd = np.zeros((dim.shape[0] + 1, ), dtype=np.uint32)
    crd[1:] = np.cumsum(dim[:, 0])
    reps = np.ones_like(cumc, dtype=np.uint32)
    tiles = np.ones_like(reps)
    reps = cumc[-1] / cumc
    tiles[1:] = cumc[:-1]

    out = np.empty((crd[-1], cumc[-1]), dtype=np.int64)

    for (i, v) in enumerate(in_arrays):
        out[crd[i]:crd[i+1]] = \
            np.tile(v.repeat(reps[i], axis=1), (1, tiles[i]))

    out = out.swapaxes(axis, 0)

    return out