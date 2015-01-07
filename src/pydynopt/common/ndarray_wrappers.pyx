
from libc.stdlib cimport free, malloc

import numpy as np
from numpy cimport ndarray

from .types cimport num_arr2d_t, int_real_t

cdef class Finalizer:
    cdef void *data

    def __dealloc__(self):
        if self.data is not NULL:
            free(self.data)


cdef ndarray make_ndarray(unsigned int nr, unsigned int nc, int_real_t basetype):
    cdef int_real_t *arr = <int_real_t*>malloc(nr * nc * sizeof(int_real_t))
    cdef num_arr2d_t mv = <int_real_t[:nr, :nc]>arr
    cdef Finalizer _finalizer = Finalizer()
    _finalizer.data = <void*>arr
    cdef ndarray nparr = np.asarray(mv)
    cnp.set_array_base(nparr, _finalizer)

    return nparr