# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: profile = False

from libc.math cimport exp, sin, sqrt

import cython
from cython.parallel import prange, threadid

from pydynopt.common.types cimport int_real_t

cpdef int mvtest(int_real_t[:,::1] mv, int_real_t[:,::1] to) nogil:


    cdef unsigned long k, h
    cdef unsigned long m = mv.shape[0], n = mv.shape[1]
    cdef int thid

    for k in prange(m, schedule='static', num_threads=4, nogil=True):
        thid = threadid()
        for h in range(n):
            to[k, h] = worker(<int_real_t>mv[k, h], thid)

    return 1


cdef int_real_t worker(int_real_t a, int nthread) nogil:
    cdef double foo = <double>a
    cdef double res = a
    for i in range(1000):
        res += exp(res) / 10.0 + sin(res) * sqrt(res**100) + 0.5

    return <int_real_t>res

cpdef int_real_t mvtest2(int_real_t[:,:,:,::1] mv, long i, long j):

    cdef long n2 = mv.shape[2], n3 = mv.shape[3]

    cdef int_real_t[:, ::1] mv2 = mv[i, j]

    cdef int_real_t sum = 0
    cdef unsigned long k, h

    for k in range(mv2.shape[0]):
        for h in range(mv2.shape[1]):
            sum += mv2[k, h]

    return sum