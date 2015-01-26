


cdef class Optimizer:
    cdef double objective(Optimizer self, double val) nogil:
        pass

cdef class OptResult:
    """
    Include empty definition, otherwise this extension type is not accessible
    from Python.
    """
    pass

cdef class OptimizerWrapper(Optimizer):

    cdef double objective(OptimizerWrapper self, double val) nogil:
        if self.objective_ptr is not NULL:
            return self.objective_ptr(val)