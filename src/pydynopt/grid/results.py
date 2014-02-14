from __future__ import absolute_import, print_function, division


class DynoptResult(object):

    def __init__(self, ps, v, opt_choice, iters, tol, transm=None):
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
        self._transm = transm

        # Copy all these attributes separately instead of storing the entire
        # problem specification object, as that one might be quite large due to
        # precomputed cached data.
        self._grid = ps.grid
        self._idx = ps.idx
        self._grid_shape = ps.grid_shape
        self._grid_shape_end = ps.grid_shape_end
        self._grid_shape_exo = ps.grid_shape_exo
        self._nstates, self._nstates_end, self._nstates_exo = \
            ps.nstates, ps.nstates_end, ps.nstates_exo
        self._ndim, self._ndim_end, self._ndim_exo = \
            ps.ndim, ps.ndim_end, ps.ndim_exo

    @property
    def grid(self):
        return self._grid

    @property
    def idx(self):
        return self.idx

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def grid_shape_end(self):
        return self._grid_shape_end

    @property
    def grid_shape_exo(self):
        return self._grid_shape_exo

    @property
    def ndim(self):
        return self._ndim

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
    def nstates(self):
        return self._nstates

    @property
    def nstates_exo(self):
        return self._nstates_exo

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
    def transm(self):
        return self._transm