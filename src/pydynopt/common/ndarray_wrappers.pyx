
from libc.stdlib cimport free, malloc

import numpy as np
cimport numpy as cnp
from numpy cimport ndarray

from cython cimport numeric

from .types cimport numc2d_t

cdef class Finalizer:
    cdef void *data

    def __dealloc__(self):
        if self.data is not NULL:
            free(self.data)


cdef ndarray make_ndarray(unsigned int nr, unsigned int nc, numeric basetype):
    cdef numeric *arr = <numeric*>malloc(nr * nc * sizeof(numeric))
    cdef numc2d_t mv = <numeric[:nr, :nc]>arr
    cdef Finalizer _finalizer = Finalizer()
    _finalizer.data = <void*>arr
    cdef cnp.ndarray nparr = np.asarray(mv)
    cnp.set_array_base(nparr, _finalizer)

    return nparr