
from libc.stdlib cimport free, malloc
from cython import nonecheck, boundscheck, wraparound

import numpy as np
from numpy cimport ndarray

from .types cimport int_real_t

cdef class Finalizer:
    cdef void *data

    def __dealloc__(self):
        if self.data is not NULL:
            free(self.data)


@nonecheck(False)
@boundscheck(False)
@wraparound(False)
cdef ndarray make_ndarray(unsigned long nr, unsigned long nc,
                          int_real_t basetype, bint ravel=True):

    cdef int_real_t *arr = <int_real_t*>malloc(nr * nc * sizeof(int_real_t))
    cdef Finalizer _finalizer = Finalizer()
    _finalizer.data = <void*>arr


    cdef int_real_t[:, ::1] mv = <int_real_t[:nr, :nc]>arr
    cdef int_real_t[::1] mv1d
    cdef ndarray nparr

    if nr == 1 and ravel:
        mv1d = <int_real_t[:nc]>arr
        nparr = np.asarray(mv1d)
    else:
        nparr = np.asarray(mv)

    cnp.set_array_base(nparr, _finalizer)

    return nparr
