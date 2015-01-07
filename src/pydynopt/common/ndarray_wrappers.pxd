cimport numpy as cnp
from cython cimport numeric

cdef cnp.ndarray make_ndarray(unsigned int nr, unsigned int n, numeric size)