

from pydynopt.optimize.interface cimport Optimizer, OptResult

cpdef double fminbound(Optimizer opt, double x1, double x2, OptResult res,
                       double xatol=*, unsigned int maxiter=*) nogil