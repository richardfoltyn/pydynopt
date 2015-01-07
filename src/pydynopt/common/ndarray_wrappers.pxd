cimport numpy as cnp
from .types cimport int_real_t

cdef cnp.ndarray make_ndarray(unsigned int nr, unsigned int n, int_real_t size)