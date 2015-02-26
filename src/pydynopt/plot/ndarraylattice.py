from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from .styles import DefaultStyle
import numpy as np

from collections import Callable
import itertools as it
from copy import copy

from .baseplots import plot_grid


class PlotAxis(object):

    def __init__(self, dim=None, at_idx=slice(None, None, 1), at_val=None,
                 values=None):

        self.indexes = None
        self.values = None
        self.dim = dim

        self.update(at_idx, at_val, values)

    def update(self, at_idx=slice(None, None, 1), at_val=None, values=None):
        """

        Initialization procedure:

        The same as for PlotDimension, except that we do not expand at_idx if it
        is a slice instance.
        """

        if at_val is not None:
            at_val = np.atleast_1d(at_val)
        if values is not None:
            values = np.atleast_1d(values)
        if at_idx is not None:
            if not isinstance(at_idx, slice):
                at_idx = np.atleast_1d(at_idx)

        if at_val is not None:
            if at_idx is not None:
                if not isinstance(at_idx, slice) and \
                        (at_val.shape != at_idx.shape):
                    raise ValueError('Arguments at_val and at_idx not '
                                     'compatible')
            elif values is not None:
                at_idx = np.searchsorted(values, at_val)
            else:
                at_idx = np.arange(len(at_val))
        elif at_idx is not None:
            if values is not None:
                at_val = values[at_idx]
        elif values is not None:
            at_val = values
            at_idx = np.arange(len(values))

        if self.dim is not None and at_idx is None:
            raise ValueError('Insufficient information to obtain array '
                             'indices for dimension {:d}'.format(self.dim))

        self.indexes = at_idx
        self.values = at_val


class PlotDimension(object):

    def __init__(self, dim=None, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None, label_fun=None,
                 label_loc='lower right'):

        self.label_fmt = label_fmt
        self.label = label
        self.label_loc = label_loc if label or label_fmt or label_fun else None

        if label_fun is not None and not isinstance(label_fun, Callable):
            raise ValueError('Argument label_fun not callable')
        self.label_fun = label_fun

        self.dim = dim
        self.values = None
        self.indexes = None

        self.update(at_idx, at_val, values)

    def update(self, at_idx=None, at_val=None, values=None):
        """

        Initialization procedure:

        1) at_val takes highest precedence. If specified:
            - `values` argument is ignored
            - if at_idx given -> check for consistency
            - if at_idx not given but values given -> impute from values
            - if neither at_idx nor values given -> at_idx is sequence along
                    at_val
        2) at_idx specified, at_val not specified:
            - if values not given -> ignore
            - if values given: let at_val = values[at_idx]
        3) `values` specified, at_idx and at_val not specified:
            - set at_val = values and at_idx = range(len(values))
        """

        if at_val is not None:
            at_val = np.atleast_1d(at_val)
        if values is not None:
            values = np.atleast_1d(values)
        if at_idx is not None:
            if isinstance(at_idx, slice):
                if at_idx.start is None or at_idx.stop is None:
                    raise ValueError('at_idx does not support slices without '
                                     'start/stop values!')
                step = at_idx.step
                at_idx = np.arange(at_idx.start, at_idx.stop, step)
            else:
                at_idx = np.atleast_1d(at_idx)

        if at_val is not None:
            if at_idx is not None:
                if at_val.shape != at_idx.shape:
                    raise ValueError('Arguments at_val and at_idx not '
                                     'compatible')
            elif values is not None:
                at_idx = np.searchsorted(values, at_val)
            else:
                at_idx = np.arange(len(at_val))
        elif at_idx is not None:
            if values is not None:
                at_val = values[at_idx]
            else:
                at_val = tuple([None] * len(at_idx))
        elif values is not None:
            at_val = values
            at_idx = np.arange(len(values))

        self.indexes = at_idx
        self.values = at_val

    def __len__(self):
        if self.indexes is None:
            # Length is as least implicitly 1, even if nothing is specified,
            # since each plot must have at least one row, one column, one layer
            # and a point on the x-axis.
            return 1
        else:
            return len(self.indexes)

    def __iter__(self):
        if self.indexes is None:
            return zip((None, ), (None, ))
        else:
            return zip(self.indexes, self.values)

    def get_label(self, largs, idx):
        lbl = None
        if self.label_fun is not None:
            lbl = self.label_fun(largs)
        elif self.label_fmt:
            lbl = self.label_fmt.format(largs)
        elif self.label:
            if isinstance(self.label, str):
                lbl = self.label
            else:
                try:
                    lbl = self.label[idx]
                except (TypeError, IndexError):
                    pass

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

        for arg in (rows, cols):
            if not isinstance(arg, PlotDimension):
                raise ValueError('Argument must be of type PlotDimension')
        if not isinstance(layers, PlotLayer):
            raise ValueError('\'layer\' argument must be of type PlotLayer')
        if not isinstance(xaxis, PlotAxis):
            raise ValueError('\'xaxis\' argument must be of type PlotAxis')

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
                    slice_ijk = template.copy()
                    if ridx is not None:
                        slice_ijk[self.rows.dim] = ridx
                    if cidx is not None:
                        slice_ijk[self.cols.dim] = cidx
                    if lidx is not None:
                        slice_ijk[self.layers.dim] = lidx
                    if self.xaxis.dim is not None:
                        slice_ijk[self.xaxis.dim] = self.xaxis.indexes

                    if ndim != 1:
                        slice_ijk = tuple(slice_ijk)
                    else:
                        slice_ijk = self.xaxis.indexes

                    indexes[i, j, k] = ridx, cidx, lidx
                    slices[i, j, k] = slice_ijk
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
    """
    Class to hold arguments passed to labor formatting functions.

    The following attributes can be used either in a str.format() string,
    or in a custom callback function which will be called with an instance of
    LabelArgs as the only argument:

    Attributes
    ----------

    row : int
        Current row's index on plot grid

    column : int
        Current column's index on plot grid

    layer : int
        Current layer's index on plot grid

    index : int
        Current data array index along dimension `dim` that is mapped to row
        / column / layer. (None if not mapped)

    value : float
        Current value corresponding to `index` (None if not mapped)

    """
    def __init__(self, row, column, layer, index=None, value=None):
        self.row = row
        self.column = column
        self.layer = layer
        self.index = index
        self.value = value


class NDArrayLattice(object):
    """
    Mapping from N-dimensional array to 3-dimensional grid of plots (
    rows/columns/layers).
    """

    def __init__(self):
        self.rows = None
        self.cols = None
        self.xaxis = None
        self.layers = None
        self.fixed_dims = None
        self.fixed_idx = None

    def map_rows(self, dim, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None, label_fun=None,
                 label_loc='lower left'):
        """
        Map data array dimension `dim` to rows on plot grid. Rows will be
        plotted at array indexes given by `at_idx' along dimension `dim`.

        Alternatively, indexes to be plotted can be selected by specifying the
        desired values using `at_val`. The corresponding indexes will be
        imputed from `values'.

        Either `at_idx` or both `at_val` and `values` must be specified.

        Parameters
        ----------

        dim : int
            Data array dimension to be mapped to rows.

        at_idx: array_like
            Indexes along `dim' to be plotted row-wise. (optional)

        at_val: array_like
            Values corresponding to indexes along `dim' to be plotted
            row-wise. (optional)

        values: array_like
            Values corresponding to indexes on `dim' (optional)

        label: str
            String specifying a static label (optional)

        label_fmt: str
            String format. If not None, str.format(larg) will be applied,
            passing LabelArgs as unnamed argument. (optional)

        label_fun: callable
            Callback function that returns string to be used as label. Will
            be passed an instance of LabelArgs. (optional)

        label_loc: str
            Location on subplot where label will be placed. Valid values are
            the same as for Matplotlib legend's `loc` argument.

        Returns
        -------

        Nothing

        """

        pd = PlotDimension(dim, at_idx, at_val, values, label, label_fmt,
                           label_fun, label_loc)

        self.rows = pd

    def map_columns(self, dim, at_idx=None, at_val=None, values=None,
                    label=None, label_fmt=None, label_fun=None,
                    label_loc='lower left'):

        pd = PlotDimension(dim, at_idx, at_val, values, label, label_fmt,
                           label_fun, label_loc)
        self.cols = pd

    def map_layers(self, dim=None, at_idx=None, at_val=None, values=None,
                   label=None, label_fmt=None, label_fun=None, **kwargs):

        pl = PlotLayer(dim, at_idx, at_val, values, label, label_fmt,
                       label_fun, **kwargs)

        self.layers = pl

    def map_xaxis(self, dim=None, at_idx=slice(None, None, 1), at_val=None,
                  values=None):
        """
        Map data array dimension `dim` to x-axis, using `values` as labels.
        Indexes to be shown on x-axis can optionally be restricted using
        `at_idx` or `at_val`.

        Parameters
        ----------

        dim : int
            Data array dimension to be mapped to rows.

        at_idx: array_like
            Indexes along `dim` to be plotted on x-axis. (optional)

        at_val: array_like
            Values corresponding to indexes along `dim' to be plotted
            on x-axis. (optional)

        values: array_like
            Values corresponding to indexes on `dim' (optional)

        Returns
        -------

        Nothing

        """
        self.xaxis = PlotAxis(dim=dim, at_idx=at_idx, at_val=at_val,
                              values=values)

    def set_fixed_dims(self, dim, at_idx):
        """
        Fix data array dimensions to specific indexes. This is useful for
        high-dimensional arrays that can or should not be mapped to
        rows/columns/layers. The specified dimensions are fixed across all
        plots on grid.

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
        dim = np.atleast_1d(dim)
        at_idx = np.atleast_1d(at_idx)

        if dim.shape != at_idx.shape:
            raise ValueError('Arguments dim and at_idx must be of equal shape')

        self.fixed_dims = dim
        self.fixed_idx = at_idx

    def reset_fixed_dims(self):
        self.fixed_dims = None
        self.fixed_idx = None

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

        xaxis = self.xaxis if self.xaxis is not None else PlotAxis()
        rows = self.rows if self.rows is not None else PlotDimension()
        cols = self.cols if self.cols is not None else PlotDimension()
        layers = self.layers if self.layers is not None else PlotLayer()

        if rows.dim is not None:
            if rows.indexes is None:
                rows = copy(rows)
                rows.update(at_idx=np.arange(data.shape[rows.dim]))

        if cols.dim is not None:
            if cols.indexes is None:
                cols = copy(cols)
                cols.update(at_idx=np.arange(data.shape[cols.dim]))

        if layers.dim is not None:
            if layers.indexes is None:
                layers = copy(layers)
                layers.update(at_idx=np.arange(data.shape[layers.dim]))

        pm = PlotMap(xaxis=xaxis, rows=rows, cols=cols, layers=layers,
                     template=template)
        return pm

    def plot(self, data, style=None, trim_iqr=2.0, ylim=None,
             xlim=None, extend_by=0.0, identity=False, **kwargs):
        """
        Plot data array using mappings specified prior to calling this method.

        Parameters
        ----------
        data: ndarray
            Data array

        style: pydynopt.plot.styles.AbstractStyle
            Style to be applied to plot (optional)

        trim_iqr: float
            If not None, all observations outside of the interval
            defined by the median +/- `trim_iqr` * IQR will not be plotted. (optional)

        ylim: tuple
            Plot limits along y-axis (optional)

        xlim: tuple
            Plot limits along x-axis (optional)

        extend_by: float
            if not None, limits of x- and y-axis will be extended by a fraction
            `extend_by' of the limits obtained from the data array. (optional)

        identity: bool
            Add identity function (45° line) to plot background. (optional)

        kwargs: dict
            Parameters passed to plot_grid.

        Returns
        -------

        Nothing

        """

        # construct the minimum required shape of array that supports the
        # specified mapping.

        min_shape = np.zeros((self.ndim, ), dtype=np.int)

        for pd in (self.rows, self.cols, self.layers):
            if pd and pd.dim is not None and pd.indexes is not None:
                min_shape[pd.dim] = np.max(pd.indexes) + 1
        x = self.xaxis
        if x and x.dim is not None:
            if isinstance(x.indexes, slice):
                if x.indexes.stop is not None:
                    min_shape[x.dim] = x.indexes.stop
            else:
                min_shape[x.dim] = np.max(x.indexes)
        if self.fixed_dims is not None:
            for dim, idx in zip(self.fixed_dims, self.fixed_idx):
                min_shape[dim] = idx + 1

        if not np.all(min_shape <= data.shape):
            raise ValueError('Expected minimum array shape {}, '
                             'got {}'.format(tuple(min_shape), data.shape))

        NDArrayLattice.plot_arrays(self, data, style, trim_iqr, ylim, xlim,
                             extend_by, identity=identity, **kwargs)


    @staticmethod
    def plot_arrays(nda, data, style=None, trim_iqr=2, ylim=None,
                    xlim=None, extend_by=0, identity=False,
                    **kwargs):

        if not isinstance(data, (tuple, list)):
            data = (data, )

        ndat = len(data)
        nda = tuple(np.atleast_1d(nda))
        pmaps = tuple(z.get_plot_map(d) for z, d in zip(nda, data))
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
                        lbl = pm.layers.get_label(larg, k)

                        plot_kw = {'ls': lstyle[k], 'lw': lwidth[k],
                                   'c': colors[k], 'alpha': alphas[k]}
                        if layer.plot_kw:
                            plot_kw.update(normalize_plot_args(layer.plot_kw))

                        ax.plot(xvals, y, label=lbl, **plot_kw)

                        if k == 0:
                            larg = LabelArgs(i, j, k, aidx[PlotMap.IDX_ROW],
                                             vals[PlotMap.IDX_ROW])
                            rtxt = pm.rows.get_label(larg, i)

                            larg.value = vals[PlotMap.IDX_COL]
                            larg.index = aidx[PlotMap.IDX_COL]
                            ctxt = pm.cols.get_label(larg, j)

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
                    ymin, ymax = bnd_extend((np.min(yy), np.max(yy)),
                                            by=extend_by)
                    if trim_iqr is not None:
                        q25, q50, q75 = np.percentile(yy, (25, 50, 100))
                        ymin = max(ymin, q50 - trim_iqr * (q75-q25))
                        ymax = min(ymax, q50 + trim_iqr * (q75-q25))

                    ax.set_ylim(ymin, ymax)

        plot_grid(func, nrow=nrow, ncol=ncol, style=style, **kwargs)