
from pydynopt.common.types cimport int_real_t
from libc.math cimport fabs

cdef int_real_t cy_matrix_supnorm(int_real_t[:,:] arr1,
                                  int_real_t[:,:] arr2) nogil