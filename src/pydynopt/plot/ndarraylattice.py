from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from .styles import DefaultStyle
import numpy as np

from collections import Callable
import itertools as it

from .baseplots import plot_grid


class PlotDimension(object):

    def __init__(self, dim=None, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None, label_fun=None,
                 label_loc='lower right'):

        self.dim = dim
        self.label_fmt = label_fmt
        self.label_loc = label_loc
        self.label = label

        if label_fun is not None and not isinstance(label_fun, Callable):
            raise ValueError('Argument label_fun not callable')
        self.label_fun = label_fun

        if at_val is not None and values is None:
            raise ValueError('Argument values must not be None if at_val '
                             'specified')

        if at_val is not None and at_idx is not None and values is not None:
            try:
                values = np.atleast_1d(values)
                at_val2 = values[np.atleast_1d(at_idx)]
                assert np.all(np.atleast_1d(at_val) == at_val2)
            except:
                raise ValueError('Arguments at_val and at_idx not compatible')

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

    def get_label(self, largs):
        if self.label_fun is not None:
            lbl = self.label_fun(largs)
        elif self.label_fmt:
            lbl = self.label_fmt.format(largs)
        elif self.label:
            lbl = self.label
        else:
            lbl = None

        return lbl


class PlotLayer(PlotDimension):

    def __init__(self, dim=None, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None, label_fun=None,
                 **kwargs):

        super(PlotLayer, self).__init__(dim=dim, at_idx=at_idx, at_val=at_val,
                                        values=values, label=label,
                                        label_fmt=label_fmt,
                                        label_fun=label_fun)
        self.plot_kw = kwargs


class PlotMap(object):

    IDX_ROW = 0
    IDX_COL = 1
    IDX_LAYER = 2

    def __init__(self, xaxis, rows=None, cols=None, layers=None, template=None):

        rows = rows if rows is not None else PlotDimension()
        cols = cols if cols is not None else PlotDimension()
        layers = layers if layers is not None else PlotLayer()

        for arg in (xaxis, rows, cols):
            if not isinstance(arg, PlotDimension):
                raise ValueError('Argument must be of type PlotDimension')

        self.rows, self.cols, self.layers, self.xaxis = rows, cols, layers, xaxis
        nrow, ncol, nlyr = len(rows), len(cols), len(layers)

        slices = np.ndarray((nrow, ncol, nlyr), dtype=object)
        values = np.ndarray((nrow, ncol, nlyr), dtype=object)
        indexes = np.ndarray((nrow, ncol, nlyr), dtype=object)

        if template is None:
            ndim = 0
            for z in (self.rows, self.cols, self.xaxis, self.layers):
                if z.dim is not None:
                    ndim = max(ndim, z.dim)
            ndim += 1
            template = [0] * ndim
        else:
            template = list(template)
            ndim = len(template)

        for i, (ridx, rval) in enumerate(self.rows):
            for j, (cidx, cval) in enumerate(self.cols):
                for k, (lidx, lval) in enumerate(self.layers):
                    lst = template.copy()
                    if ridx is not None:
                        lst[self.rows.dim] = ridx
                    if cidx is not None:
                        lst[self.cols.dim] = cidx
                    if lidx is not None:
                        lst[self.layers.dim] = lidx
                    if self.xaxis.dim is not None:
                        lst[self.xaxis.dim] = self.xaxis.at

                    indexes[i, j, k] = ridx, cidx, lidx
                    slices[i, j, k] = tuple(lst)
                    values[i, j, k] = rval, cval, lval

        self.slices = slices
        self.indexes = indexes
        self.values = values
        self.ndim = ndim

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


class LabelArgs:
    def __init__(self, row, col, layer, idx=None, val=None):
        self.row = row
        self.col = col
        self.layer = layer
        self.idx = idx
        self.val = val


class NDArrayLattice(object):

    def __init__(self):
        self.rows = PlotDimension()
        self.cols = PlotDimension()
        self.xaxis = PlotDimension()
        self.layers = PlotLayer()
        self.fixed_dims = None
        self.fixed_idx = None

    def map_rows(self, dim, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None, label_fun=None,
                 label_loc='lower left'):

        pd = PlotDimension(dim, at_idx, at_val, values, label, label_fmt,
                           label_fun, label_loc)

        self.rows = pd

    def map_columns(self, dim, at_idx=None, at_val=None, values=None,
                    label=None, label_fmt=None, label_fun=None,
                    label_loc='lower left'):

        pd = PlotDimension(dim, at_idx, at_val, values, label, label_fmt,
                           label_fun, label_loc)
        self.cols = pd

    def map_layers(self, dim, at_idx=None, at_val=None, values=None,
                   label=None, label_fmt=None, label_fun=None, **kwargs):

        pl = PlotLayer(dim, at_idx, at_val, values, label, label_fmt,
                       label_fun, **kwargs)

        self.layers = pl

    def set_fixed_dims(self, dim, at_idx):
        self.fixed_dims = np.atleast_1d(dim)
        self.fixed_idx = np.atleast_1d(at_idx)

    def reset_fixed_dims(self):
        self.fixed_dims = None

    def reset_layers(self):
        self.layers = PlotLayer()

    def reset_rows(self):
        self.rows = PlotDimension()

    def reset_cols(self):
        self.cols = PlotDimension()

    def set_xaxis(self, dim=None, at_idx=None, at_val=None, values=None):
        self.xaxis = PlotDimension(dim=dim, at_idx=at_idx, at_val=at_val,
                                   values=values)

    @property
    def ndim(self):
        ndim = 0
        for z in (self.rows, self.cols, self.xaxis, self.layers):
            if z.dim is not None:
                ndim = max(ndim, z.dim)

        if self.fixed_dims is not None:
            ndim = max(np.max(self.fixed_dims), ndim)

        ndim += 1
        return ndim

    def get_plot_map(self):

        template = None
        if self.fixed_dims is not None:
            template = [0] * self.ndim
            for dim, idx in zip(self.fixed_dims, self.fixed_idx):
                template[dim] = idx

        pm = PlotMap(xaxis=self.xaxis, rows=self.rows, cols=self.cols,
                     layers=self.layers, template=template)
        return pm

    def plot(self, data, style=None, trim_iqr=2, ylim=None,
             xlim=None, extend_by=0, identity=False, **kwargs):

        NDArrayLattice._plot(self, data, style, trim_iqr, ylim, xlim,
                             extend_by, identity=identity, **kwargs)



    @staticmethod
    def _plot(nda, data, style=None, trim_iqr=2, ylim=None,
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

        if style is None:
            style = DefaultStyle()
        colors = style.color_seq(nlayer)
        lstyle = style.lstyle_seq(nlayer)
        alphas = style.alpha_seq(nlayer)
        lwidth = style.lwidth_seq(nlayer)

        def func(ax, idx):
            i, j = idx

            nonlocal yy, xmin, xmax

            annotations = np.ndarray((3, 3), dtype=object)

            for idx_dat, (dat, pm) in enumerate(zip(data, pmaps)):
                if i < pm.nrow and j < pm.ncol:
                    for k, sl_k in enumerate(pm.slices[i, j]):
                        idx_plt = idx + (k, )
                        vals = pm.values[idx_plt]
                        aidx = pm.indexes[idx_plt]
                        layer = pm.layers

                        y = dat[sl_k].squeeze()
                        if ylim is None or trim_iqr is not None:
                            yy.append(y)
                        xvals = pm.xvalues
                        if xvals is None or len(xvals) == 0:
                            xvals = np.arange(len(y))

                        if xlim is None:
                            xmin = min(np.min(xvals), xmin)
                            xmax = max(np.max(xvals), xmax)

                        larg = LabelArgs(i, j, k, aidx[PlotMap.IDX_LAYER],
                                         vals[PlotMap.IDX_LAYER])
                        lbl = pm.layers.get_label(larg)

                        plot_kw = {'ls': lstyle[k], 'lw': lwidth[k],
                                   'c': colors[k], 'alpha': alphas[k]}
                        if layer.plot_kw:
                            plot_kw.update(normalize_plot_args(layer.plot_kw))

                        ax.plot(xvals, y, label=lbl, **plot_kw)

                        if k == 0:
                            larg = LabelArgs(i, j, k, aidx[PlotMap.IDX_ROW],
                                             vals[PlotMap.IDX_ROW])
                            rtxt = pm.rows.get_label(larg)

                            larg.val = vals[PlotMap.IDX_COL]
                            larg.idx = aidx[PlotMap.IDX_COL]
                            ctxt = pm.cols.get_label(larg)

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
                        txt_kwargs.update(style.text)
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

        plot_grid(func, nrow=nrow, ncol=ncol, style=style, **kwargs)