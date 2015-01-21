# cython: cdivision = True
# cython: wraparound = False
# cython: boundscheck = False

DEF ARR_SIZE = 100


from libc.math cimport ceil
from libc.stdio cimport printf

import numpy as np

cdef struct container:
    double carr[ARR_SIZE]


cdef class ContainerType:
    cdef double carr[ARR_SIZE]
    cdef double carr2[ARR_SIZE][ARR_SIZE]
    cdef double[::1] _mv
    cdef double[:, ::1] _mv2

    def __cinit__(ContainerType self):
        self._mv = self.carr
        self._mv2 = self.carr2

    property mv:
        def __get__(ContainerType self):
            return np.asarray(self._mv)

        def __set__(ContainerType self, double[::1] value):
            assert value.ndim == 1 and value.shape[0] == ARR_SIZE
            self._mv[:] = value

    property mv2:
        def __get__(ContainerType self):
            return np.asarray(self._mv2)

        def __set__(ContainerType self, double[:,::1] value):
            assert value.shape[0] == ARR_SIZE and value.shape[1] == ARR_SIZE
            self._mv2[:] = value


    def print_arr(ContainerType self):
        cdef unsigned int i, j, n = 10, m = <unsigned int>ceil(ARR_SIZE / n)
        cdef unsigned int mm, idx, nn
        for i in range(m):
            nn = min(n, ARR_SIZE  - i * n)
            for j in range(nn):
                idx = i * n + j
                print("%4.2f" % self.carr[idx])
            print('')

    def test_carr(self, unsigned int times=int(1e6)):
        cdef unsigned int i, j
        cdef double s
        for i in range(times):
            s = 0.0
            for j in range(ARR_SIZE):
                s += self.carr[j]

    def test_mv(self, unsigned int times=int(1e6)):
        cdef unsigned int i, j
        cdef double s
        for i in range(times):
            s = 0.0
            for j in range(ARR_SIZE):
                s += self._mv[j]
