__author__ = 'Richard Foltyn'


import cython
from cython.parallel import prange, threadid, parallel

from libc.stdlib cimport malloc, free

DEF NUM_THREADS=4

cdef class TestClass:
    cdef unsigned int a
    cdef double[::1] b


cdef inline double worker(TestClass cls) nogil:

    cdef double[::1] mv = cls.b
    cdef double result = 0.0
    cdef unsigned long i
    for i in range(mv.shape[0]):
        result += mv[i]

    return result * cls.a

cdef double test_func(double[:,::1] mv):

    cdef double res = 0.0
    cdef TestClass *cls = <TestClass *>malloc(sizeof(TestClass) * NUM_THREADS)
    cdef unsigned int i
    cdef unsigned int thread_id
    cdef TestClass cls_tls

    for i in range(NUM_THREADS):
        cls[i] = TestClass()


    for i in prange(mv.shape[0], schedule='static',
                    num_threads=NUM_THREADS, nogil=True):
        res += worker(cls[i])

    return res

def run_test(double[:,::1] mv):

    cdef double res
    res = test_func(mv)

    return res
