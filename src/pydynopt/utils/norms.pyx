

cdef int_real_t cy_matrix_supnorm(int_real_t[:,:] arr1,
                                  int_real_t[:,:] arr2) nogil:

    cdef long i, j
    cdef double norm = 0.0
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            norm = max(norm, fabs(arr1[i, j] - arr2[i, j]))

    return <int_real_t>norm


def matrix_supnorm(int_real_t[:,:] arr1, int_real_t[:,:] arr2):
    return cy_matrix_supnorm(arr1, arr2)