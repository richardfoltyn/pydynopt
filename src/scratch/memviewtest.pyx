# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: profile = False

from libc.math cimport exp, sin, sqrt

import cython
# from cython.parallel import prange, threadid
#
# from pydynopt.common.types cimport int_real_t
#
# cpdef int mvtest(int_real_t[:,::1] mv, int_real_t[:,::1] to) nogil:
#
#
#     cdef unsigned long k, h
#     cdef unsigned long m = mv.shape[0], n = mv.shape[1]
#     cdef int thid
#
#     for k in prange(m, schedule='static', num_threads=4, nogil=True):
#         thid = threadid()
#         for h in range(n):
#             to[k, h] = worker(<int_real_t>mv[k, h], thid)
#
#     return 1
#
#
# cdef int_real_t worker(int_real_t a, int nthread) nogil:
#     cdef double foo = <double>a
#     cdef double res = a
#     for i in range(1000):
#         res += exp(res) / 10.0 + sin(res) * sqrt(res**100) + 0.5
#
#     return <int_real_t>res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mvtest2(double[:,:,:,::1] mv, unsigned long i) nogil:

    cdef double s = 0.0

    cdef double[:, ::1] mv2
    cdef unsigned long j, k, l

    for j in range(mv.shape[1]):
        mv2 = mv[i, j]

        for k in range(mv2.shape[0]):
            for l in range(mv2.shape[1]):
                s += mv2[k, l]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mvtest3(double[:,:,:,::1] mv, unsigned long i) nogil:

    cdef double s = 0.0

    cdef unsigned long j, k, l

    for j in range(mv.shape[1]):
        for k in range(mv.shape[2]):
            for l in range(mv.shape[3]):
                s += mv[i, j, k, l]

    return s

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef double mvtest1(double[:,::1] mv) nogil:
#
#     cdef double s = 0.0
#     cdef unsigned long j, k
#
#     for j in range(mv.shape[0]):
#         for k in range(mv.shape[1]):
#             s += impl(mv[j, k])
#
#     return s
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline double impl(double[:,::1] mv) nogil:
#     cdef double s = 0.0
#     cdef unsigned long k, l
#
#     for k in range(mv.shape[0]):
#         for l in range(mv.shape[1]):
#             s += mv[k, l]
#
#     return s

def test3(double[:,:,:,::1] arr, unsigned long i, unsigned int n):

    cdef double res = 0.0
    cdef unsigned int j
    for j in range(n):
        res += mvtest3(arr, i)
    return res

def test2(double[:,:,:,::1] arr, unsigned long i, unsigned int n):

    cdef double res = 0.0
    cdef unsigned int j
    for j in range(n):
        res += mvtest2(arr, i)
    return res
