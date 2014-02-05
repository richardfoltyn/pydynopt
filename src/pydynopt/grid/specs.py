__author__ = 'richard'

import numpy as np


class ProblemSpec(object):
    def __init__(self, grid_shape, discount):
        self._grid_shape = grid_shape
        self._discount = discount

        self._ndim = len(grid_shape)
        self._nstates = np.prod(grid_shape)

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def discount(self):
        return self._discount

    @property
    def nstates(self):
        return self._nstates

    @property
    def ndim(self):
        return self._ndim

    def actions(self, i_state, ix_state=0):
        pass

    def transitions(self, actions, i_state, ix_state=0):
        pass

    def util(self, actions, i_state, ix_state=0):
        pass


class ProblemSpecExogenous(ProblemSpec):
    def __init__(self, grid_shape_end, grid_shape_exo, discount):
        self._grid_shape_end = grid_shape_end
        self._grid_shape_exo = grid_shape_exo

        self._ndim_end = len(grid_shape_end)
        self._ndim_exo = len(grid_shape_exo)
        self._nstates_end = np.prod(grid_shape_end)
        self._nstates_exo = np.prod(grid_shape_exo)

        super(ProblemSpecExogenous, self).__init__(
            grid_shape_end + grid_shape_exo, discount)

    @property
    def grid_shape_end(self):
        return self._grid_shape_end

    @property
    def grid_shape_exo(self):
        return self._grid_shape_exo

    @property
    def ndim_end(self):
        return self._ndim_end

    @property
    def ndim_exo(self):
        return self._ndim_exo

    @property
    def nstates_end(self):
        return self._nstates_end

    @property
    def nstates_exo(self):
        return self._nstates_exo

    def transitions_exo(self, ix):
        pass
