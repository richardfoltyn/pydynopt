__author__ = 'richard'

import numpy as np


class ProblemSpec(object):
    def __init__(self, grid_shape, discount):
        self._grid_shape = grid_shape
        self._discount = discount

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def discount(self):
        return self._discount

    @property
    def nstates(self):
        return np.prod(self._grid_shape)

    @property
    def ndim(self):
        return len(self._grid_shape)

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
        return len(self._grid_shape_end)

    @property
    def ndim_exo(self):
        return len(self._grid_shape_exo)

    @property
    def nstates_end(self):
        return np.prod(self._grid_shape_end)

    @property
    def nstates_exo(self):
        return np.prod(self._grid_shape_exo)

    def transitions_exo(self, ix):
        pass
