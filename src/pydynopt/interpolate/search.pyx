
import numpy as np
from pydynopt.common.python_helpers import reshape_result

cdef struct interp_accel:
    pass

cdef inline size_t \
        c_interp_bsearch(double *xp, double x, size_t lb, size_t ub) nogil:
    cdef size_t midx

    while ub > (lb + 1):
        midx = (ub + lb) // 2
        if xp[midx] > x:
            ub = midx
        else:
            lb = midx

    return lb

cdef inline size_t \
    c_interp_find(double *xp, double x, size_t length, interp_accel *acc) nogil:

    cdef size_t index

    if acc == NULL:
        if x <= xp[0]:
            index = 0
        elif x >= xp[length - 2]:
            index = length - 2
        else:
            index = c_interp_bsearch(xp, x, 0, length - 1)
        return index
    else:
        if x <= xp[0]:
            acc.index = 0
        elif x >= xp[length - 2]:
            acc.index = length - 2
        else:
            index = acc.index
            if x < xp[index]:
                acc.index = c_interp_bsearch(xp, x, 0, index)
            elif x >= xp[index + 1]:
                acc.index = c_interp_bsearch(xp, x, index, length - 1)
        return acc.index


def interp_find(double[:] xp, x):


    cdef double *ptr_xp = &(xp[0])
    biter = np.broadcast(x, 0.0)

    cdef size_t[::1] lb = np.empty(biter.size, dtype=np.uint)

    cdef size_t i
    cdef size_t length = xp.shape[0]
    cdef double x_i
    cdef interp_accel acc

    for i, x_i in enumerate(biter):
        lb[i] = c_interp_find(ptr_xp, x_i, length, &acc)

    return reshape_result(biter, lb)