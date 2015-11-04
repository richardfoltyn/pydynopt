from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

from .styles import DefaultStyle
import numpy as np

from collections import Callable
import itertools as it
from copy import copy, deepcopy

from .baseplots import plot_grid
import sys


class PlotDimension(object):

    def __init__(self, dim=None, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None, label_loc='lower right',
                 fixed=False, **kwargs):
        """
        Keep **kwargs for compatibility since older code might pass label_fun
        argument.
        """

        self.label_fmt = label_fmt
        self.label = label
        self.label_loc = label_loc if label or label_fmt else None

        self.dim = int(dim) if dim is not None else None
        self.values = None
        self.index = None
        self.fixed = fixed

        self.update(at_idx, at_val, values)

    def update(self, at_idx=None, at_val=None, values=None):
        """

        Initialization procedure:

        1)  Nothing was specified: assume everything is to be plotted along this
            dimension. Axis labels will not be available since values is
            missing.
        2)  `values` specified: Same as (1), but axis labels available
        3)  `values` and `at_idx` given: Same as (2), but restrict to indices
            that correspond to values in `at_val`
        4)  `at_idx` specified: Plot at requested indices, no values for
            xaxis available
        5)  `at_idx` and `values` specified: same as (4), but with axis labels
        6)  `at_idx` and `at_val` given: same as (5), but values are provided
            directly
        """

        if at_idx is None:
            if at_val is not None and values is not None:
                at_idx = tuple(np.searchsorted(values, at_val))
            else:
                # Plot everything if nothing else was specified
                at_idx = slice(None)
        elif not isinstance(at_idx, slice):
            at_idx = tuple(np.atleast_1d(at_idx))

        # ignore values if at_val given
        if at_val is None and values is not None:
            at_val = values
            print('Ignoring \'values\' argument as \'at_val\' was given',
                  file=sys.stderr)

        # At this point we have a non-missing at_idx and a missing or
        # non-missing at_val
        if at_val is not None:
            at_val = np.atleast_1d(at_val)

        if at_val is not None and not isinstance(at_idx, slice):
            if len(at_idx) != len(at_val):
                raise ValueError('Non-conformable index / value arrays!')

        # if index is a scalar, create a slice as otherwise subscripting
        # would drop the dimension from the resulting mapped data array
        if not self.fixed and not isinstance(at_idx, slice) and len(at_idx) == 1:
            at_idx = slice(at_idx[0], at_idx[0] + 1)

        self.index = at_idx
        self.values = at_val

    def __iter__(self):
        if self.index is None:
            return zip((None, ), (None, ))
        else:
            return zip(self.index, self.values)

    def get_label(self, largs, idx):
        lbl = None
        if isinstance(self.label, Callable):
            lbl = self.label(largs)
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


class PlotMap(object):

    def __init__(self, xaxis=PlotDimension(dim=0, at_idx=slice(None)),
                 rows=None, cols=None, layers=None,
                 fixed=None):

        for arg in (rows, cols, layers, xaxis):
            if arg and not isinstance(arg, PlotDimension):
                raise ValueError('Argument must be of type PlotDimension')

        self.rows, self.cols, self.layers, self.xaxis = rows, cols, layers, xaxis

        self.mapped = dict()
        for d in (rows, cols, layers, xaxis):
            if d and d.dim is not None:
                if d.dim in self.mapped:
                    raise ValueError("Duplicate mapping for single dimension")
                self.mapped[d.dim] = d

        self.fixed = dict()
        if fixed is not None:
            fixed = tuple(np.array(fixed, dtype=object))
            assert all(isinstance(x, PlotDimension) for x in fixed)

            for d in fixed:
                if d.dim is None:
                    raise ValueError('Missing dim attribute for fixed '
                                     'dimension')
                if d.index is None:
                    raise ValueError('Missing index for fixed dimension')

                if d.dim in self.mapped:
                    raise ValueError("Duplicate mapping for single dimension")
                self.fixed[d.dim] = d

        # To be filled from specific data array
        self.nrow = None
        self.ncol = None
        self.nlayer = None
        self.ndim = None

    @property
    def xvalues(self):
        return self.xaxis.values

    def apply(self, data):
        """
        Apply mapping to data array, resulting in a 4-dimensional array
        (rows x cols x layers x xaxis). Some dimensions can be of length 1 if you
        mapping was specified for that particular dimension.
        """

        all_dims = self.mapped.copy()
        all_dims.update(self.fixed)

        dims = np.sort(list(all_dims))

        data = np.asarray(data)
        if (data.ndim - 1) < dims[-1]:
            raise ValueError('Plot map dimension exceeds data dimension')

        fixed = dict()
        if len(dims) < data.ndim:
            # There are holes in dimension map; this is fine if the missing
            # dimension are length 1, but were not specified using `fixed`
            # argument
            for i in range(data.ndim):
                if i not in all_dims and data.shape[i] == 1:
                    fixed[i] = PlotDimension(dim=i, at_idx=0, fixed=True)
                else:
                    raise ValueError('No mapping for data dimension {:d} '
                                     'given'.format(i))

        idx = [0] * data.ndim
        for key, val in all_dims.items():
            idx[key] = val.at_idx
        # Add newly found fixed dimensions for particular data array
        for key, val in fixed.items():
            idx[key] = val.at_idx

        idx = tuple(idx)

        reduced_data = data[idx]

        # create a concrete plot mapping for particular data array
        pm = deepcopy(self)
        pm.fixed_dims.update(fixed)

        for d in (pm.rows, pm.cols, pm.layers, pm.xaxis):
            if d is not None:
                if isinstance(d.index, slice):
                    # expand to true index values if index was defined as slice
                    d.index = np.arange(data.shape[d.dim])[d.index]
                if d.values is None:
                    d.values = d.index

        return reduced_data, pm

    def add_fixed(self, dim, at_idx):
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

        for d, i in zip(dim, at_idx):
            if d in self.fixed:
                raise ValueError('Fixed dimension {:d} exists!'.format(d))
            self.fixed[d] = PlotDimension(dim=d, at_idx=i, fixed=True)

    def map_rows(self, dim, *args, **kwargs):
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

        if self.rows is not None:
            del self.mapped[self.rows.dim]

        self.rows = PlotDimension(dim, *args, **kwargs)
        self.mapped[dim] = self.rows

    def map_columns(self, dim, *args, **kwargs):

        if self.cols is not None:
            del self.mapped[self.cols.dim]

        self.cols = PlotDimension(dim, *args, **kwargs)
        self.mapped[dim] = self.cols

    def map_layers(self, dim, *args, **kwargs):

        if self.layers is not None:
            del self.mapped[self.layers.dim]

        self.layers = PlotDimension(dim, *args, **kwargs)

    def map_xaxis(self, dim=0, at_idx=slice(None), *args, **kwargs):
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

        del self.mapped[self.xaxis.dim]
        self.xaxis = PlotDimension(dim, at_idx, *args, **kwargs)

    def plot(self, data, style=None, trim_iqr=2.0, ylim=None,
             xlim=None, extendy=0.01, extendx=0.01, identity=False, **kwargs):
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

        extendy: float
            if not None, limits of x- and y-axis will be extended by a fraction
            `extendy' of the limits obtained from the data array. (optional)

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
                                   extendy=extendy, extendx=extendx,
                                   identity=identity,
                                   **kwargs)


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


def bnd_extend(arr, by=0.025):
    diff = arr[-1] - arr[0]
    return arr[0] - by * diff, arr[1] + by * diff


def adj_bounds(arr, extend_by, trim_iqr):
    ymin, ymax = np.min(arr), np.max(arr)

    if trim_iqr is not None:
        q25, q50, q75 = np.percentile(arr, (25, 50, 100))
        ymin = max(ymin, q50 - trim_iqr * (q75 - q25))
        ymax = min(ymax, q50 + trim_iqr * (q75 - q25))

    if extend_by and extend_by > 0:
        ymin, ymax = bnd_extend((ymin, ymax), by=extend_by)

    return ymin, ymax


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


class NDArrayLattice(PlotMap):

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

        xaxis = self.xaxis if self.xaxis is not None else PlotAxis()
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


def plot_pm(pm, data, style=None, trim_iqr=2, ylim=None,
                xlim=None, extendy=0.01, extendx=0.01,
                identity=False, sharey=True,
                **kwargs):

    if not isinstance(data, (tuple, list)):
        data = (data, )

    ndat = len(data)
    pm = tuple(np.atleast_1d(pm))
    pmaps = tuple(z.get_plot_map(d) for z, d in zip(nda, data))
    if ndat > 1:
        if len(pmaps) == 1:
            pmaps = tuple(it.repeat(pmaps[0], ndat))
        elif len(pmaps) != ndat:
            raise ValueError('Plot map / data lengths not compatible!')

    nrow = max(pm.nrow for pm in pmaps)
    ncol = max(pm.ncol for pm in pmaps)
    nlayer = sum(pm.nlayer for pm in pmaps)

    xmin, xmax = np.inf, -np.inf

    yy = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            yy[i, j] = list()

    trim_iqr = float(trim_iqr) if trim_iqr is not None else trim_iqr

    if style is None:
        style = (DefaultStyle(), )
    else:
        style = tuple(np.atleast_1d(style))

    if len(style) != ndat:
        if len(style) == 1:
            style *= ndat
        else:
            raise ValueError('style parameter cannot be broadcast to '
                             'match length of data arrays.')

    def func(ax, idx):
        i, j = idx

        nonlocal yy, xmin, xmax

        annotations = np.ndarray((3, 3), dtype=object)

        for idx_dat, (dat, pm, st) in enumerate(zip(data, pmaps, style)):
            if i < pm.nrow and j < pm.ncol:
                for k, sl_k in enumerate(pm.slices[i, j]):
                    idx_plt = idx + (k, )
                    vals = pm.values[idx_plt]
                    aidx = pm.indexes[idx_plt]
                    layer = pm.layers

                    y = dat[sl_k].squeeze()
                    if ylim is None:
                        yy[i, j].append(y)

                    xvals = pm.xvalues
                    if xvals is None or len(xvals) == 0:
                        xvals = np.arange(len(y))

                    if xlim is None:
                        xmin = min(np.min(xvals), xmin)
                        xmax = max(np.max(xvals), xmax)

                    larg = LabelArgs(i, j, k, aidx[PlotMap.IDX_LAYER],
                                     vals[PlotMap.IDX_LAYER])
                    lbl = pm.layers.get_label(larg, k)

                    plot_kw = {'ls': st.linestyle[k],
                               'lw': st.linewidth[k],
                               'c': st.color[k],
                               'alpha': st.alpha[k]}
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
                    txt_kwargs.update(style[0].text)
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

        if not sharey and ylim is None:
            arr = np.hstack(yy[i, j])
            ymin, ymax = adj_bounds(arr, extendy, trim_iqr)

            ax.set_ylim(ymin, ymax)

        if i == (nrow - 1) and j == (ncol - 1):
            if xlim is not None and len(xlim) == 2:
                ax.set_xlim(xlim)
            else:
                xmin, xmax = bnd_extend((xmin, xmax), by=extendx)
                ax.set_xlim(xmin, xmax)

            if ylim is not None and len(ylim) == 2:
                ax.set_ylim(ylim)
            elif sharey:
                arr = np.hstack([np.hstack(yy_i) for yy_i in yy.ravel()])
                ymin, ymax = adj_bounds(arr, extendy, trim_iqr)

                ax.set_ylim(ymin, ymax)

    plot_grid(func, nrow=nrow, ncol=ncol, style=style[0],
              sharey=sharey, **kwargs)