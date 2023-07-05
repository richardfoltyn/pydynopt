"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""


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