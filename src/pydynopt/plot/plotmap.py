from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'
import numpy as np


class PlotDimension(object):

    def __init__(self, idx_dim=None, idx_arr=None, values=None, labelfmt=None):
        self.idx_dim = idx_dim
        self.labelfmt = labelfmt

        if idx_arr is None:
            idx_arr = (None, )

        if isinstance(idx_arr, slice):
            self.idx_arr = idx_arr
        else:
            self.idx_arr = tuple(np.atleast_1d(idx_arr))

        if values is None:
            values = (None, )

        self.values = np.atleast_1d(values)

    def __getitem__(self, key):
        return self.values[key]

    def __len__(self):
        return len(self.idx_arr)


class PlotMap(object):

    IDX_XAXIS = 0
    IDX_ROW = 1
    IDX_COL = 2
    IDX_LAYER = 3

    def __init__(self, shape, xaxis, rows=PlotDimension(), cols=PlotDimension(),
                 layers=PlotDimension, template=None):

        for arg in (xaxis, rows, cols, layers):
            if not isinstance(arg, PlotDimension):
                raise ValueError('Argument must be of type PlotDimension')

        shape = tuple(np.atleast_1d(shape))

        if template is not None:
            template = list(np.atleast_1d(template))
        else:
            template = list(shape)

        if len(shape) != len(template):
            raise ValueError('shape and template arguments must be of equal '
                             'length')

        template[xaxis.idx_dim] = xaxis.idx_arr

        self.rows, self.cols, self.layers, self.xaxis = rows, cols, layers, xaxis
        nrow, ncol, nlyr = len(rows), len(cols), len(layers)

        arr = np.ndarray((nrow, ncol, nlyr), dtype=object)
        arr_idx = np.ndarray((nrow, ncol, nlyr), dtype=object)
        for i in range(nrow):
            ri = self.rows.idx_arr[i]
            if ri is not None:
                template[rows.idx_dim] = ri
            for j in range(ncol):
                cj = self.cols.idx_arr[j]
                if cj is not None:
                    template[cols.idx_dim] = cj
                for k in range(nlyr):
                    pk = self.layers.idx_arr[k]
                    if pk is not None:
                        template[layers.idx_dim] = pk

                    arr[i, j, k] = tuple(template)
                    arr_idx[i, j, k] = (ri, cj, pk)

        self.slices = arr
        self.arr_idx = arr_idx

    @property
    def shape(self):
        return self.slices.shape

    @property
    def nrow(self):
        return self.slices.shape[0]

    @property
    def ncol(self):
        return self.slices.shape[1]

    @property
    def nlayer(self):
        return self.slices.shape[2]

    def __getitem__(self, idx):
        idx = tuple(np.atleast_1d(idx))
        sl = self.slices[idx]
        return sl

    def indices(self, idx):
        if len(idx) != 3:
            raise IndexError('Need 3-tuple index')
        idx = tuple(np.atleast_1d(idx))

        ai = (self.xaxis.idx_arr, ) + self.arr_idx[idx]
        return ai

    def values(self, idx):
        if len(idx) != 3:
            raise IndexError('Need 3-tuple index')
        idx = tuple(idx)

        sl = self.slices[idx]

        vals = [self.xaxis[sl[self.xaxis.idx_dim]]]
        for dim in (self.rows, self.cols, self.layers):
            if dim.idx_dim is not None:
                vals.append(dim[sl[dim.idx_dim]])
            else:
                vals.append(None)

        return tuple(vals)



