from __future__ import absolute_import, print_function, division


class DynoptResult(object):

    def __init__(self, ps, v, opt_choice, iters, tol, transitions=None):
        self._ps = ps
        self._v = v.reshape(ps.grid_shape)
        # Policy functions might contain more than one choice variable,
        # which will be the last dimension of this thing. Reshape accordingly.
        # Squeeze the policy function dimension if there is only one choice
        # variable.
        if len(opt_choice.shape) > 2 and opt_choice.shape[2] > 1:
            shp = ps.grid_shape + (-1,)
        else:
            shp = ps.grid_shape
        self._opt_choice = opt_choice.reshape(shp)
        self._iters, self._tol = iters, tol
        self._trans = transitions

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

    @property
    def transition_matrix(self):
        return self._trans