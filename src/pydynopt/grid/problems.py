from __future__ import division, print_function, absolute_import

import numpy as np


class GridContainer(object):
    """
    Container class to be used as mapping between grid dimension names and
    their integer indices.
    """
    pass


class ProblemSpecExogenous(object):
    def __init__(self, par, discount):
        self._discount = discount

        self._grid_shape = tuple()
        self._grid_shape_exo = tuple()
        self._grid_shape_end = tuple()

        self._ndim = self._ndim_end = self._ndim_exo = 0
        self._nstates = self._nstates_exo = self._nstates_end = 0

        self._grid = list()
        self._transm = list()
        self._stationary_dist = list()

        self._idx = GridContainer()

        self._par = par

    @property
    def par(self):
        return self._par

    @property
    def idx(self):
        return self._idx

    @property
    def grid(self):
        return self._grid

    @property
    def transm(self):
        return self._transm

    @property
    def stationary_dist(self):
        return self._stationary_dist

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
        return self._nstates_exo

    @property
    def nstates_end(self):
        return self._nstates_end

    @property
    def nstates_exo(self):
        return self._nstates_exo

    def actions(self, i_state, ix_state=0):
        pass

    def transitions(self, actions, i_state, ix_state=0):
        pass

    def transitions_exo(self, ix):
        pass

    def add_grid_dim(self, idx_name, grid_vals, exogenous=False, transm=None,
                     stationary_dist=None):
        if idx_name in self._idx.__dict__.keys():
            raise ValueError('Grid index name already exists')

        self._idx.__dict__[idx_name] = len(self._grid)

        grid_vals = np.atleast_1d(grid_vals)
        self._grid.append(grid_vals)
        grid_len = len(grid_vals)

        self._transm.append(transm)
        if stationary_dist is not None:
            self._stationary_dist.append(np.atleast_1d(stationary_dist))
        else:
            self._stationary_dist.append(None)

        # recompute grid shapes, etc.
        if exogenous:
            self._grid_shape_exo = self._grid_shape_exo + (grid_len, )
        else:
            self._grid_shape_end = self._grid_shape_end + (grid_len, )

        self._grid_shape = self._grid_shape_end + self._grid_shape_exo
        self._update_grid_props()

    def _update_grid_props(self):
        self._nstates = int(np.prod(self._grid_shape))
        self._nstates_end = int(np.prod(self._grid_shape_end))
        self._nstates_exo = int(np.prod(self._grid_shape_exo))

        self._ndim = len(self._grid_shape)
        self._ndim_end = len(self._grid_shape_end)
        self._ndim_exo = len(self._grid_shape_exo)

