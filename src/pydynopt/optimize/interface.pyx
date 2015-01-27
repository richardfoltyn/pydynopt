


cdef class Optimizer:
    cdef double objective(Optimizer self, double val) nogil:
        pass

cdef class OptResult:

    def __cinit__(self):
        # initialize to zero, the default flag when no errors are encountered.
        self.flags = 0x0

cdef class OptimizerWrapper(Optimizer):

    cdef double objective(OptimizerWrapper self, double val) nogil:
        if self.objective_ptr is not NULL:
            return self.objective_ptr(val)