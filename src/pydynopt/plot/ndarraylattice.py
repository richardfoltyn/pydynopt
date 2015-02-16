from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from .styles import DefaultStyle
import numpy as np

import collections
import itertools as it

from .baseplots import plot_grid


class PlotDimension(object):

    def __init__(self, dim=None, at_idx=None, at_val=None, values=None,
                 label_fmt=None, label=None, label_loc='lower right',
                 label_fun=None):

        self.dim = dim
        self.label_fmt = label_fmt
        self.label_loc = label_loc
        self.label = label
        self.label_fun = label_fun

        if at_val is not None and values is None:
            raise ValueError('Argument values must not be None if at_val '
                             'specified')

        if dim is not None:
            if values is not None:
                values = np.sort(np.atleast_1d(values))
            if at_idx is not None:
                if isinstance(at_idx, slice):
                    step = at_idx.step
                    at_idx = np.arange(at_idx.start, at_idx.stop, step)
                else:
                    at_idx = np.atleast_1d(at_idx)
            if at_val is not None:
                at_val = np.atleast_1d(at_val)

            if values is not None:
                if at_val is not None:
                    at_idx = np.searchsorted(values, at_val)
                elif at_idx is not None:
                    at_val = values[at_idx]
                else:
                    at_idx = np.arange(len(values))
                    at_val = values

            if at_idx is None:
                raise ValueError('Insufficient information to obtain array '
                                 'indices for dimension {:d}'.format(dim))
        else:
            at_idx = (None, )
            at_val = (None, )
            values = (None, )

        self.at_idx = at_idx
        self.at_val = at_val
        self.values = values

    def __len__(self):
        return len(self.at_idx)

    def __iter__(self):
        return zip(self.at_idx, self.at_val)


class PlotLayer(PlotDimension):

    def __init__(self, dim=None, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None, label_loc=None, label_fun=None,
                 **kwargs):

        super(PlotLayer, self).__init__(dim=dim, at_idx=at_idx, at_val=at_val,
                                        values=values, label=label,
                                        label_fmt=label_fmt,
                                        label_loc=label_loc,
                                        label_fun=label_fun)
        self.plot_kw = kwargs


class PlotMap(object):

    IDX_XAXIS = 0
    IDX_ROW = 1
    IDX_COL = 2
    IDX_LAYER = 3

    def __init__(self, xaxis, rows=None, cols=None, layers=None):

        rows = rows if rows is not None else PlotDimension()
        cols = cols if cols is not None else PlotDimension()
        layers = layers if layers is not None else (PlotLayer(), )

        for arg in (xaxis, rows, cols):
            if not isinstance(arg, PlotDimension):
                raise ValueError('Argument must be of type PlotDimension')

        if isinstance(layers, PlotLayer):
            layers = (layers, )
        else:
            layers = tuple(layers)

        self.rows, self.cols, self.layers, self.xaxis = rows, cols, layers, xaxis
        nrow, ncol, nlyr = len(rows), len(cols), len(layers)

        arr_slices = np.ndarray((nrow, ncol, nlyr), dtype=object)
        arr_values = np.ndarray((nrow, ncol, nlyr), dtype=object)

        ndim = 0
        for z in (self.rows, self.cols, self.xaxis):
            if z.dim is not None:
                ndim = max(ndim, z.dim)
        for z in self.layers:
            if z.dim_max is not None:
                ndim = max(ndim, z.dim_max)
        ndim += 1

        for i, (ridx, rval) in enumerate(self.rows):
            for j, (cidx, cval) in enumerate(self.cols):
                for k, layer in enumerate(self.layers):
                    lst = [0] * ndim
                    if ridx is not None:
                        lst[self.rows.dim] = ridx
                    if cidx is not None:
                        lst[self.cols.dim] = cidx

                    for ldim, lidx, _ in layer:
                        lst[ldim] = lidx

                    if self.xaxis.dim is not None:
                        lst[self.xaxis.dim] = self.xaxis.at

                    arr_slices[i, j, k] = tuple(lst)
                    arr_values[i, j, k] = rval, cval, layer.values

        self.slices = arr_slices
        self.values = arr_values

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

    @property
    def xvalues(self):
        return self.xaxis.values

    def values(self, idx):
        if len(idx) != 3:
            raise IndexError('Need 3-tuple index')
        idx = tuple(idx)

        return self.values[idx]


loc_map = {'upper': 2, 'lower': 0, 'left': 0, 'right': 2, 'center': 1}


def loc_text_to_tuple(text):
    tok = text.split()

    if len(tok) == 1:
        vidx = 1
        hidx = loc_map[tok[0]]
    else:
        vidx = loc_map[tok[0]]
        hidx = loc_map[tok[1]]

    return vidx, hidx

loc_kwargs = np.ndarray((3, 3), dtype=object)
for i, (valign, y) in enumerate(zip(('bottom', 'center', 'top'),
                                    (0.05, 0.5, 0.95))):
    for j, (halign, x) in enumerate(zip(('left', 'center', 'right'),
                                        (0.05, 0.5, 0.95))):
        loc_kwargs[i, j] = {'verticalalignment': valign,
                            'horizontalalignment': halign,
                            'x': x, 'y': y}


def normalize_plot_args(args):
    res = args.copy()
    if 'color' in res:
        res['c'] = res['color']
        del res['color']
    if 'linewidth' in res:
        res['lw'] = res['linewidth']
        del res['lw']
    if 'linestyle' in res:
        res['ls'] = res['linestyle']
        del res['ls']
    return res


def bnd_extend(arr, by=0.025):
    diff = arr[-1] - arr[0]
    return arr[0] - by * diff, arr[1] + by * diff


def plot_identity(ax, xmin, xmax, extend=1):
    xx = [xmin - extend*(xmax - xmin),
          xmax + extend*(xmax - xmin)]
    ax.plot(xx, xx, ls=':', color='black', alpha=0.6, lw=1, zorder=-500)


class NDArrayLattice(object):

    def __init__(self):
        self.rows = PlotDimension()
        self.cols = PlotDimension()
        self.xaxis = PlotDimension()
        self.layers = PlotLayer()

    def map_rows(self, dim, at=None, values=None, label=None, label_fmt=None,
                 label_loc='lower left'):
        self.rows = PlotDimension(dim=dim, at=at, label=label, values=values,
                                  label_fmt=label_fmt, label_loc=label_loc)

    def map_columns(self, dim, at=None, values=None, label=None, label_fmt=None,
                 label_loc='lower left'):
        self.cols = PlotDimension(dim=dim, at=at, label=label, values=values,
                                  label_fmt=label_fmt, label_loc=label_loc)

    def map_layers(self, dim, at=None, values=None, label=None,
                   label_fmt=None, **kwargs):
        self.layers.append(PlotLayer(dim=dim, at=at, values=values,
                                     label=label, label_fmt=label_fmt,
                                     **kwargs))

    def reset_layers(self):
        self.layers = list()

    def reset_rows(self):
        self.rows = PlotDimension()

    def reset_cols(self):
        self.cols = PlotDimension()

    def set_xaxis(self, dim=None, at=None, values=None):
        self.xaxis = PlotDimension(dim=dim, at=at, values=values)

    def get_plot_map(self):
        layers = self.layers if self.layers else PlotLayer()

        pm = PlotMap(xaxis=self.xaxis, rows=self.rows, cols=self.cols,
                     layers=layers)
        return pm

    def plot(self, data, style=DefaultStyle, trim_iqr=2, ylim=None,
             xlim=None, extend_by=0, identity=False, **kwargs):
        NDArrayLattice._plot(self, data, style, trim_iqr, ylim, xlim,
                             extend_by, identity=identity, **kwargs)



    @staticmethod
    def _plot(nda, data, style=DefaultStyle, trim_iqr=2, ylim=None,
              xlim=None, extend_by=0, identity=False,
              **kwargs):

        if not isinstance(data, (tuple, list)):
            data = (data, )

        ndat = len(data)
        nda = tuple(np.atleast_1d(nda))
        pmaps = tuple(z.get_plot_map() for z in nda)
        if ndat > 1:
            if len(pmaps) == 1:
                pmaps = tuple(it.repeat(pmaps[0], ndat))
            elif len(pmaps) != ndat:
                raise ValueError('Plot map / data lengths not compatible!')

        nrow = max(pm.nrow for pm in pmaps)
        ncol = max(pm.ncol for pm in pmaps)
        nlayer = max(pm.nlayer for pm in pmaps)

        xmin, xmax = np.inf, -np.inf

        yy = []
        trim_iqr = float(trim_iqr)

        stl = style()
        colors = stl.color_seq(nlayer)
        lstyle = stl.lstyle_seq(nlayer)
        alphas = stl.alpha_seq(nlayer)
        lwidth = stl.lwidth_seq(nlayer)

        def func(ax, idx):
            i, j = idx

            nonlocal yy, xmin, xmax

            annotations = np.ndarray((3, 3), dtype=object)

            for idx_dat, (dat, pm) in enumerate(zip(data, pmaps)):
                if i < pm.nrow and j < pm.ncol:
                    for k, sl_k in enumerate(pm[i, j]):
                        idx_plt = idx + (k, )
                        # ix, irow, icol, iplt = sl.indices(plt_idx)
                        vals = pm.values[idx_plt]
                        layer = pm.layers[k]

                        y = dat[sl_k].squeeze()
                        if ylim is None or trim_iqr is not None:
                            yy.append(y)
                        xvals = pm.xvalues
                        if xvals is None or len(xvals) == 0:
                            xvals = np.arange(len(y))

                        if xlim is None:
                            xmin = min(np.min(xvals), xmin)
                            xmax = max(np.max(xvals), xmax)

                        if layer.label_fmt:
                            lbl = layer.label_fmt.format(row=i, col=j,
                                                         values=vals)
                        else:
                            lbl = layer.label if layer.label else None

                        plot_kw = {'ls': lstyle[k], 'lw': lwidth[k],
                                   'c': colors[k], 'alpha': alphas[k]}
                        if layer.plot_kw:
                            plot_kw.update(normalize_plot_args(layer.plot_kw))

                        ax.plot(xvals, y, label=lbl, **plot_kw)

                        if k == 0:
                            if pm.rows.label_fmt:
                                rtxt = pm.rows.label_fmt.format(row=i, col=j,
                                                                values=vals)
                            else:
                                rtxt = pm.rows.label if pm.rows.label else ''

                            if pm.cols.label_fmt:
                                ctxt = pm.cols.label_fmt.format(row=i, col=j,
                                                               values=vals)
                            else:
                                ctxt = pm.cols.label if pm.cols.label else ''

                            lst = []
                            if rtxt or ctxt:
                                if pm.rows.label_loc == pm.cols.label_loc:
                                    lidx = loc_text_to_tuple(pm.rows.label_loc)
                                    txt = ', '.join((rtxt, ctxt))
                                    lst.append((lidx, txt))
                                else:
                                    if rtxt:
                                        lidx = loc_text_to_tuple(pm.rows.label_loc)
                                        lst.append((lidx, rtxt))
                                    if ctxt:
                                        lidx = loc_text_to_tuple(pm.cols.label_loc)
                                        lst.append((lidx, ctxt))

                                for lidx, txt in lst:
                                    if not annotations[lidx]:
                                        annotations[lidx] = list()
                                    annotations[lidx].append(txt)

            for ii in range(3):
                for jj in range(3):
                    lst = annotations[ii, jj]
                    if lst:
                        txt_kwargs = loc_kwargs[ii, jj].copy()
                        y = txt_kwargs['y']
                        dy = 1 if y < 0.5 else -1
                        txt_kwargs['transform'] = ax.transAxes
                        for itxt, txt in enumerate(annotations[ii, jj]):
                            yi = y + dy * itxt * 0.075
                            txt_kwargs.update({'s': txt, 'y': yi})
                            ax.text(**txt_kwargs)

            if identity:
                if xlim and len(xlim) == 2:
                    xxmin, xxmax = xlim
                else:
                    xxmin, xxmax = xmin, xmax
                plot_identity(ax, xxmin, xxmax)

            ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 3))

            if i == (nrow - 1) and j == (ncol - 1):
                if xlim is not None and len(xlim) == 2:
                    ax.set_xlim(xlim)
                else:
                    xmin, xmax = bnd_extend((xmin, xmax), by=extend_by)
                    ax.set_xlim(xmin, xmax)

                if ylim is not None and len(ylim) == 2:
                    ax.set_ylim(ylim)
                else:
                    yy = np.hstack(tuple(yy))
                    ymin, ymax = bnd_extend((np.min(yy), np.max(yy)), by=extend_by)
                    if trim_iqr is not None:
                        q25, q50, q75 = np.percentile(yy, (25, 50, 100))
                        ymin = max(ymin, q50 - trim_iqr * (q75-q25))
                        ymax = min(ymax, q50 + trim_iqr * (q75-q25))

                    ax.set_ylim(ymin, ymax)

        plot_grid(func, nrow=nrow, ncol=ncol, **kwargs)