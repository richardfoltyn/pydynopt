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
