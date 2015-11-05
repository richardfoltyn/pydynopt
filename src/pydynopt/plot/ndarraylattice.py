from __future__ import print_function, division, absolute_import

from pydynopt.plot.plotmap import PlotDimension, PlotMap, loc_kwargs, plot_pm

__author__ = 'Richard Foltyn'

import numpy as np
from copy import copy

for i, (valign, y) in enumerate(zip(('bottom', 'center', 'top'),
                                    (0.05, 0.5, 0.95))):
    for j, (halign, x) in enumerate(zip(('left', 'center', 'right'),
                                        (0.05, 0.5, 0.95))):
        loc_kwargs[i, j] = {'verticalalignment': valign,
                            'horizontalalignment': halign,
                            'x': x, 'y': y}


class NDArrayLattice(PlotMap):

    def set_fixed_dims(self, dim, at_idx):
        """
        Fix data array dimensions to specific indexes. This is useful for
        high-dimensional arrays that can or should not be mapped to
        rows/columns/layers. The specified dimensions are fixed across all
        plot on grid.

        Parameters
        ----------
        dim : array_like
            Data array dimension to fix

        at_idx : array_like
            Indexes at which dimensions given in `dim` should be held fixed,
            one for each element given in `dim`

        Returns
        -------

        Nothing

        """

        self.add_fixed(dim, at_idx)


    def reset_fixed_dims(self):
        self.fixed = None

    def reset_layers(self):
        self.layers = None

    def reset_rows(self):
        self.rows = None

    def reset_cols(self):
        self.cols = None

    def reset_xaxis(self):
        self.xaxis = None

    @property
    def ndim(self):
        ndim = 0
        for z in (self.rows, self.cols, self.xaxis, self.layers):
            if z and z.dim is not None:
                ndim = max(ndim, z.dim)

        if self.fixed_dims is not None:
            ndim = max(np.max(self.fixed_dims), ndim)

        ndim += 1
        return ndim

    def get_plot_map(self, data):

        template = [0] * max(1, data.ndim)
        if self.fixed_dims is not None:
            for dim, idx in zip(self.fixed_dims, self.fixed_idx):
                template[dim] = idx

        xaxis = self.xaxis if self.xaxis is not None else PlotDimension()
        rows = self.rows if self.rows is not None else PlotDimension()
        cols = self.cols if self.cols is not None else PlotDimension()
        layers = self.layers if self.layers is not None else PlotDimension()

        if rows.dim is not None:
            if rows.index is None:
                rows = copy(rows)
                rows.update(at_idx=np.arange(data.shape[rows.dim]))

        if cols.dim is not None:
            if cols.index is None:
                cols = copy(cols)
                cols.update(at_idx=np.arange(data.shape[cols.dim]))

        if layers.dim is not None:
            if layers.index is None:
                layers = copy(layers)
                layers.update(at_idx=np.arange(data.shape[layers.dim]))

        pm = PlotMap(xaxis=xaxis, rows=rows, cols=cols, layers=layers,
                     template=template)
        return pm

    @staticmethod
    def plot_array(*args, **kwargs):
        plot_pm(*args, **kwargs)


