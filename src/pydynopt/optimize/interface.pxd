
from pydynopt.common.types cimport uint32_t

cdef class OptResult:
    cdef public double fx_opt
    cdef public double x_opt
    cdef public uint32_t flags

ctypedef double (*objective_t)(double) nogil

cdef class Optimizer:
    cdef double objective(Optimizer self, double val) nogil

cdef class OptimizerWrapper(Optimizer):
    cdef objective_t objective_ptr
