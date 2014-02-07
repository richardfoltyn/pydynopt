__author__ = 'richard'


class ConvergenceError(BaseException):

    def __init__(self, iterations, tol):
        self._iters = iterations
        self._tol = tol

    @property
    def iterations(self):
        return self._iters

    @property
    def tol(self):
        return self._tol