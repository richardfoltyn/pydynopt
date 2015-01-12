cimport numpy as cnp
from .types cimport int_real_t

cdef cnp.ndarray make_ndarray(unsigned long nr, unsigned long n,
                              int_real_t size, bint ravel=*)