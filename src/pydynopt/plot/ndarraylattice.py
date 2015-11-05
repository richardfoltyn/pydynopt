from __future__ import print_function, division, absolute_import

from pydynopt.plot.plotmap import PlotMap, plot_pm

__author__ = 'Richard Foltyn'


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

    def reset_layers(self):
        if self.layers is not None:
            del self.mapped[self.layers.dim]
        self.layers = None

    def reset_rows(self):
        if self.rows is not None:
            del self.mapped[self.rows.dim]
        self.rows = None

    def reset_cols(self):
        if self.cols is not None:
            del self.mapped[self.cols.dim]
        self.cols = None

    def reset_xaxis(self):
        if self.xaxis is not None:
            del self.xaxis[self.xaxis.dim]
        self.xaxis = None

    @property
    def ndim(self):
        raise NotImplementedError()

    @staticmethod
    def plot_array(*args, **kwargs):
        plot_pm(*args, **kwargs)


