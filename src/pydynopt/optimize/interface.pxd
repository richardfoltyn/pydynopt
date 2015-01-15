

cdef class OptResult:
    cdef public double fx_opt
    cdef public double x_opt
    cdef int flag

ctypedef double (*objective_t)(double) nogil

cdef class Optimizer:
    cdef double objective(Optimizer self, double val) nogil

cdef class OptimizerWrapper(Optimizer):
    cdef objective_t objective_ptr
