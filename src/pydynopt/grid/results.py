from __future__ import absolute_import, print_function, division


class DynoptResult(object):

    def __init__(self, ps, v, opt_choice, iters, tol, idx_to=None):
        self._ps = ps
        self._v = v
        self._opt_choice = opt_choice
        self._iters, self._tol = iters, tol

    @property
    def vfun(self):
        return self._v

    @property
    def opt_choice(self):
        return self._opt_choice

    @property
    def iterations(self):
        return self._iters

    @property
    def tol(self):
        return self._tol

    @property
    def problem_spec(self):
        return self._ps