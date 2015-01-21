

from pydynopt.common.ndarray_wrappers cimport make_ndarray
from pydynopt.common.types cimport num_arr2d_t, num_arr1d_t, int_real_t

cpdef int _cartesian2d(num_arr2d_t a, num_arr2d_t b, num_arr2d_t out) nogil