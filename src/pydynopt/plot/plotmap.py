"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from __future__ import print_function, division, absolute_import

import itertools as it
import re
from collections.abc import Callable
from copy import deepcopy, copy

import numpy as np

from pydynopt.plot.styles import DefaultStyle
from pydynopt.plot.baseplots import plot_grid


class PlotDimension(object):

    DEFAULT_LABEL_LOC = 'lower right'

    def __init__(self, dim, at_idx=None, at_val=None, values=None, fixed=False,
                 label=None, label_fmt=None, label_loc=DEFAULT_LABEL_LOC,
                 **kwargs):
        """
        Keep **kwargs for compatibility since older code might pass label_fun
        argument.
        """

        self.label_fmt = label_fmt

        # Backward compatibility: use label_fun argument if no label is present
        kwargs = kwargs.copy()
        if 'label_fun' in kwargs and label is None:
            label = kwargs['label_fun']
            del kwargs['label_fun']

        self.label = label
        self.label_loc = label_loc

        self._dim = None
        self.dim = dim
        self._values = None
        self._at_val = None
        self._index = None
        self.fixed = fixed
        # save remaining kwargs to be used when calling axis.plot()
        self.plot_kwargs = kwargs

        self.__repr__cached = None

        self.update(at_idx, at_val, values)

    def update(self, at_idx=None, at_val=None, values=None):
        """

        Initialization procedure:

        1)  Nothing was specified: assume everything is to be plotted along this
            dimension. Axis labels will not be available since values is
            missing.
        2)  `values` specified: Same as (1), but axis labels available
        3)  `values` and `at_idx` given: Same as (2), but restrict to indices
            that correspond to values in `values`
        4)  `at_idx` specified: Plot at requested indices, no values for
            xaxis available
        5)  `at_idx` and `values` specified: same as (4), but with axis labels
        6)  `at_idx` and `at_val` given: same as (5), but values are provided
            directly
        """

        if at_idx is None:
            if at_val is not None and values is not None:
                at_idx = np.searchsorted(values, at_val).flatten()
            else:
                # Plot everything if nothing else was specified
                at_idx = slice(None)
        elif not isinstance(at_idx, slice):
            at_idx = np.array(at_idx, dtype=np.int).flatten()

        if at_val is not None:
            at_val = np.array(at_val).flatten()
        if values is not None:
            values = np.array(values).flatten()

        if at_val is not None and not isinstance(at_idx, slice):
            if len(at_idx) != len(at_val):
                raise ValueError('Non-conformable index / value arrays!')

        # if index is a effectively a scalar and this is not a fixed
        # mapping, create a slice as otherwise subscripting
        # would drop the dimension from the resulting mapped data array.
        # Conversely, for fixed mappings use a scalar to eliminate the
        # dimension from data array.
        if self.fixed:
            try:
                at_idx = int(at_idx[0])
            except TypeError:
                m = 'Dim {:d}: at_idx must be integer type for fixed dimensions'
                raise TypeError(m.format(self.dim))
        else:
            if not isinstance(at_idx, slice) and len(at_idx) == 1:
                at_idx = slice(at_idx[0], at_idx[0] + 1)

        self._index = at_idx
        self._values = values
        self._at_val = at_val

        self.__repr__cached = None

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = int(value)
        self.__repr__cached = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        self.__repr__cached = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        self._values = value
        self.__repr__cached = None

    @property
    def at_val(self):
        return self._at_val

    @at_val.setter
    def at_val(self, value):
        self._at_val = value
        self.__repr__cached = None

    def __iter__(self):
        if self.index is None:
            return zip((None, ), (None, ))
        else:
            return zip(self.index, self.values)

    def get_label(self, largs=None, idx=0):
        lbl = None
        if isinstance(self.label, Callable):
            lbl = self.label(largs)
        elif self.label_fmt is not None:
            # For compatibility with old format strings, detect deprecated
            # format string which contain {.attribute} with a leading dot. If
            # there is no leading dots, pass a dictionary instead.
            regex = r'(.*[^{])?[{][^{}]*\.((index)|(value)|(row)|(column)|(layer))'
            if re.match(regex, self.label_fmt):
                lbl = self.label_fmt.format(largs)
            else:
                lbl = self.label_fmt.format(**largs.to_dict())
        elif self.label is not None:
            if isinstance(self.label, str):
                lbl = self.label
            else:
                try:
                    lbl = self.label[idx]
                except (TypeError, IndexError):
                    pass

        return lbl

    def __repr__(self):
        if self.__repr__cached is None:
            tokens = ['dim={:d}'.format(self.dim)]
            if isinstance(self.index, slice):
                index = self.index
                sidx = "{!r}".format(index)
                tokens.append('index={:s}'.format(sidx))
            else:
                index = np.atleast_1d(self.index)
                if len(index) > 3:
                    sidx = '...'.join(str(x) for x in index[[0, -1]])
                else:
                    sidx = ','.join('{:d}'.format(int(x)) for x in index)

                fmt = 'index=[{:s}]' if len(index) > 1 else 'index={:s}'
                tokens.append(fmt.format(sidx))

            if self.at_val is None and self.values is not None:
                values = np.atleast_1d(self.values)[index]
            elif self.at_val is not None:
                values = self.at_val
            else:
                values = None

            if values is not None:
                if len(values) > 3:
                    sval = '...'.join(str(x) for x in values[[0, -1]])
                else:
                    sval = ','.join("{:.3f}".format(x) for x in values)

                fmt = 'values=[{:s}]' if len(values) > 1 else 'values={:s}'
                tokens.append(fmt.format(sval))

            if self.fixed:
                tokens.append('fixed=True')

            s = 'PlotDim({:s})'.format(', '.join(tokens))
            self.__repr__cached = s

        return self.__repr__cached


class PlotMap(object):

    def __init__(self, xaxis=None, rows=None, cols=None, layers=None,
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

        if fixed is not None:
            fixed = tuple(np.atleast_1d(fixed))
            assert all(isinstance(x, PlotDimension) for x in fixed)

            for d in fixed:
                if d.dim in self.mapped:
                    raise ValueError("Duplicate mapping for single dimension")
                self.mapped[d.dim] = d

    @property
    def xvalues(self):
        return self.xaxis.values

    @property
    def nrow(self):
        n = 1
        if self.rows is not None:
            try:
                n = len(self.rows.index)
            except TypeError:
                m = 'Number of rows not yet determined: need to call apply() first!'
                raise RuntimeError(m)
        return n

    @property
    def ncol(self):
        n = 1
        if self.cols is not None:
            try:
                n = len(self.cols.index)
            except TypeError:
                m = 'Number of columns not yet determined: need to call ' \
                    'apply() first!'
                raise RuntimeError(m)
        return n

    @property
    def nlayer(self):
        n = 1
        if self.layers is not None:
            try:
                n = len(self.layers.index)
            except TypeError:
                m = 'Number of layers not yet determined: need to call ' \
                    'apply() first!'
                raise RuntimeError(m)
        return n

    def __repr__(self):
        tokens = list()
        if self.xaxis is not None:
            tokens.append('{:d}=>x-axis'.format(self.xaxis.dim))
        if self.rows is not None:
            tokens.append('{:d}=>Rows'.format(self.rows.dim))
        if self.cols is not None:
            tokens.append('{:d}=>Cols'.format(self.cols.dim))
        if self.layers is not None:
            tokens.append('{:d}=>Layers'.format(self.layers.dim))
        if any(x.fixed for x in self.mapped.values()):
            fixed = list()
            for k, v in self.mapped.items():
                if v.fixed:
                    fixed.append('{:d}'.format(v.dim))
            tokens.append('Fixed: {:s}'.format(' '.join(fixed)))

        s = 'PlotMap({:s})'.format(', '.join(tokens))
        return s

    def apply(self, data):
        """
        Apply mapping to data array, resulting in a 4-dimensional array
        (rows x cols x layers x xaxis). Some dimensions can be of length 1 if you
        mapping was specified for that particular dimension.

        Parameters
        ----------

        data : array_like
            Data array to which PlotMap should be applied

        """

        # create a concrete plot mapping for aligned data array
        pm = deepcopy(self)

        # Create default x-axis mapping if nothing else was specified
        if pm.xaxis is None:
            if len(pm.mapped) != 0:
                raise RuntimeError("No map for x-axis specified!")
            else:
                pm.map_xaxis(dim=0)

        dmap = pm.mapped
        ndims = max(dmap.keys())

        data = np.asarray(data)
        if (data.ndim - 1) < ndims:
            raise ValueError('Plot map dimension exceeds data dimension')

        # Step 1: squeeze out length-1 dimensions that are not explicitly
        # mapped. Do not iterate over data.shape directly, as we are
        # modifying it in place in the loop.
        i = 0
        while i < data.ndim - 1:
            if i not in dmap:
                if data.shape[i] == 1:
                    data = data.squeeze(axis=i)
                    # adjust dimension index for mapped dimensions
                    key = np.sort(list(dmap.keys()))
                    update = dict()
                    for k in key[key > i]:
                        d = dmap.pop(k)
                        d.dim -= 1
                        update[d.dim] = d
                    # Add back updated plot dimensions at updated key values
                    dmap.update(update)
                else:
                    raise RuntimeError("Unmapped non-degenerate dimension "
                                       "encountered")
                # Note: do not increment i, as we dropped axis i and hence
                # next dimension will be at position i again!
            else:
                # move on to next dimension
                i += 1

        dlist = [dmap[k] for k in sorted(dmap.keys())]

        # Step 2: remove all fixed mappings from data and plot map. To apply
        # dimension indices we first need to make sure that these are not
        # slice() objects and they are broadcastable, otherwise numpy
        # advanced indexing might fail. For example, if for the row dimension
        # i = (1,2,3) and for the col axis j = (1,2), data[i, j, :] will fail
        # because i and j cannot be broadcast against each other.

        # Replace all non-slice tuples of non-fixed indices with
        # appropriately shaped arrays. First we need to collect all indices
        # that need to be replaced to be able to determine the broadcast
        # dimensions.

        idx = [d.index for d in dlist]
        ix_args = list()
        ix_dims = list()
        for k, index in enumerate(idx):
            # should not need to checked for non-fixed dim. as we check in
            # PlotDimensions that at_idx is int for those.
            if not isinstance(index, slice) and not dlist[k].fixed:
                ix_args.append(index)
                ix_dims.append(k)
            elif isinstance(index, slice) and not dlist[k].fixed:
                # Convert any slice indices to arrays
                ix_args.append(np.arange(data.shape[k])[index])
                ix_dims.append(k)

        if len(ix_args) > 0:
            ix_res = np.ix_(*tuple(ix_args))
            # store back results
            for k, index in zip(ix_dims, ix_res):
                idx[k] = index
                # write back expanded slices into list of PlotDimensions too;
                # flatten array since we do not want the broadcast version
                # that is used for finding the values in data. We need to do
                # this before transforming the data, later there is no way to
                #  find out what these indices actually were!
                dlist[k].index = index.flatten()

        # Reduce array to data that will be plotted
        data = data[tuple(idx)]

        # Remove fixed mappings too, keep only explicit row/col/layer/xaxis
        i = 0
        while i < len(dlist):
            if dlist[i].fixed:
                del dlist[i]
                for j in range(i, len(dlist)):
                    dlist[j].dim -= 1
            else:
                # Increment only if nothing was deleted!
                i += 1

        # Step 3: swap and insert axis as necessary to obtain an array with
        # shape (nrows, ncols, nlayers, nxvals). If no mapping is provided
        # for any of {rows, cols, layers}, then a length-1 axis is inserted.
        for i, x in enumerate((pm.rows, pm.cols, pm.layers, pm.xaxis)):
            if x is not None:
                if x.dim != i:
                    # need to swap axis and adjust dimension mapping
                    data = data.swapaxes(i, x.dim)

                    dim1 = dlist[i]
                    dim2 = dlist[x.dim]
                    # Update dimension index in PlotDimension object
                    dim1.dim = x.dim
                    dim2.dim = i
                    # Update mapping in dictionary
                    dlist[dim2.dim] = dim2
                    dlist[dim1.dim] = dim1
            else:
                data = np.expand_dims(data, i)
                # increase dimension index for all dims >= i by 1
                for d in [x for x in dlist if x.dim >= i]:
                    d.dim += 1

                dlist.insert(i, PlotDimension(dim=i, at_idx=slice(0, 1)))

        # reset mapping and refill with specific stuff
        pm.mapped = dict()

        for d in (pm.rows, pm.cols, pm.layers, pm.xaxis):
            if d is not None:
                if d.at_val is not None:
                    if len(d.index) != len(d.at_val):
                        m = "Dim {:d}: Index and value length differ"
                        raise RuntimeError(m.format(d.dim))
                elif d.values is not None:
                    try:
                        d.at_val = d.values[d.index]
                    except IndexError:
                        m = 'Dim {:d}: Non-conformable index and values arrays'
                        raise RuntimeError(m.format(d.dim))
                else:
                    d.at_val = d.index
                pm.mapped[d.dim] = d

        return data, pm

    def add_fixed(self, dim, at_idx, replace=True):
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
            if d in self.mapped:
                if not (self.mapped[d].fixed and replace):
                    m = 'Mapping for dimension {:d} already exists!'.format(d)
                    raise ValueError(m)
            self.mapped[d] = PlotDimension(dim=d, at_idx=i, fixed=True)

    def map_rows(self, dim, at_idx=None, at_val=None, values=None,
                 label=None, label_fmt=None,
                 label_loc=PlotDimension.DEFAULT_LABEL_LOC):
        """
        Map data array dimension `dim` to rows on plot grid. Rows will be
        plotted at array indexes given by `at_idx' along dimension `dim`.

        Alternatively, indexes to be plotted can be selected by specifying the
        desired values using `at_val`. The corresponding indexes will be
        imputed from `values'.

        Parameters
        ----------
        dim : int
            Data array dimension to be mapped to rows.

        at_idx : array_like
            Indexes along `dim' to be plotted row-wise. (optional)

        at_val : array_like
            Values corresponding to indexes along `dim' to be plotted
            row-wise. (optional)

        values : array_like
            Values corresponding to indexes on `dim' (optional)

        label : str or callable
            If string, specifies (static) label that does not change across
            rows / columns. If callable, will be invoked for each subplot to
            obtain subplot-specific label.

        label_fmt : str
            String format. If not None, str.format(larg) will be applied,
            passing LabelArgs as unnamed argument. (optional)

        label_loc : str
            Location on subplot where label will be placed. Valid values are
            the same as for Matplotlib legend's `loc` argument.

        Returns
        -------
        Nothing

        """

        self.map_generic('rows', dim, at_idx, at_val, values,
                         label=label, label_fmt=label_fmt, label_loc=label_loc)

    def map_columns(self, dim, at_idx=None, at_val=None, values=None,
                    label=None, label_fmt=None,
                    label_loc=PlotDimension.DEFAULT_LABEL_LOC):

        self.map_generic('cols', dim, at_idx, at_val, values,
                         label=label, label_fmt=label_fmt, label_loc=label_loc)

    def map_layers(self, dim, at_idx=None, at_val=None, values=None,
                    label=None, label_fmt=None,
                    label_loc=PlotDimension.DEFAULT_LABEL_LOC):

        self.map_generic('layers', dim, at_idx, at_val, values,
                         label=label, label_fmt=label_fmt, label_loc=label_loc)

    def map_xaxis(self, dim, at_idx=None, at_val=None, values=None,
                  label=None):
        """
        Map data array dimension `dim` to x-axis, using `values` as labels.
        Indexes to be shown on x-axis can optionally be restricted using
        `at_idx` or `at_val`.

        Parameters
        ----------

        dim : int
            Data array dimension to be mapped to rows.

        at_idx: array_like or slice
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

        self.map_generic('xaxis', dim, at_idx, at_val, values, label=label)

    def map_generic(self, kind, dim, *args, **kwargs):

        if dim in self.mapped:
            raise ValueError('Dimension {:d} already mapped!'.format(dim))

        pd = getattr(self, kind)
        if pd is not None:
            del self.mapped[pd.dim]

        pd = PlotDimension(dim, *args, **kwargs)
        setattr(self, kind, pd)
        self.mapped[pd.dim] = pd

    def plot(self, data, style=DefaultStyle(), trim_iqr=2.0, xlim=None,
             ylim=None, extendy=0.01, extendx=0.01, identity=False, sharey=True,
             legend=True, xlabel=None, ylabel=None, callback=None, **kwargs):

        """
        Plot data array using mappings specified prior to calling this method.

        Parameters
        ----------
        data : array_like
            Data to be plotted. Can be specified as list of several Numpy arrays.
        style : array_like or pydynopt.plot.styles.AbstractStyle, optional
            List of style to be applied to plots.
        trim_iqr : float, optional
            If not None, all observations outside of the interval
            defined by the median +/- `trim_iqr` * IQR will not be plotted.
        xlim : tuple
            Plot limits along x-axis
        ylim : tuple, optional
            Plot limits along y-axis
        extendx : float, optional
            If not None, limits of x-axis will be extended by a fraction
            `extendx' of the limits obtained from the data array.
        extendy : float
            If not None, limits of y-axis will be extended by a fraction
            `extendy' of the limits obtained from the data array.
        identity : bool
            Add identity function (45° line) to plot background. (optional)
        sharey : bool, optional
            If true, share common plot range on y-axis.
        legend : bool, optional
            It true, plot legend.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        callback : callable, optional
            If not None, this function will be invoked for each panel within
            the graph, proving access to the axis object, allowing for further
            customization.
        kwargs:
            Parameters passed to plot_grid.

        """

        plot_pm(self, data, style, trim_iqr, xlim, ylim, extendx, extendy,
                identity, sharey, legend, xlabel, ylabel, callback, **kwargs)


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
        # Set integer attributes via properties to ensure that these are
        # actually Python ints and not some degenerate numpy arrays

        self._row, self._column, self._layer, self._index = \
            None, None, None, None

        self.row = row
        self.column = column
        self.layer = layer
        self.index = index
        self.value = value

    @property
    def row(self):
        return self._row

    @row.setter
    def row(self, value):
        self._row = int(value)

    @property
    def column(self):
        return self._column

    @column.setter
    def column(self, value):
        self._column = int(value)

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, value):
        self._layer = int(value) if value is not None else None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = int(value) if value is not None else None

    def to_dict(self):

        d = {'row': self.row, 'column': self.column}
        if self.index is not None:
            d['index'] = self.index
        if self.value is not None:
            d['value'] = self.value
        if self.layer is not None:
            d['layer'] = self.layer

        return d


def loc_text_to_tuple(text):
    loc_map = {'upper': 0, 'lower': 2, 'left': 0, 'right': 2, 'center': 1}
    tok = text.split()

    if len(tok) == 1:
        vidx = 1
        hidx = loc_map[tok[0]]
    else:
        vidx = loc_map[tok[0]]
        hidx = loc_map[tok[1]]

    return vidx, hidx


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


def row_col_labels(nrow, ncol, maps, styles, first_only=True):

    # Compute row / column labels for each cell on grid; each cell contains
    # list of locations and labels
    labels = np.ndarray((nrow, ncol, 3, 3), dtype=object)
    text_kwargs = np.empty_like(labels)
    it = np.ix_(*tuple(range(x) for x in labels.shape))
    # initialize with empty lists so we do no need to check for None all the
    # time
    for i, j, iv, ih in np.broadcast(*it):
        labels[i, j, iv, ih] = list()
        text_kwargs[i, j, iv, ih] = list()

    # Map positions without subplot to positional kwargs
    valign = ['top', 'center', 'bottom']
    halign = ['left', 'center', 'right']
    init_y = [0.95, 0.50, 0.05]
    init_x = [0.05, 0.50, 0.95]
    # move vertical into this direction when other labels are present,
    # depending on where label is positioned within subplot
    direction_y = [-1, -1, 1]

    for i, j in np.broadcast(*np.ix_(range(nrow), range(ncol))):
        largs = LabelArgs(i, j, None)
        labels_ij = list()

        # for each subplot, collect all the labels that should be plotted
        for k, pm in enumerate(maps):
            r, c = pm.rows, pm.cols

            rtxt, ctxt, rloc, cloc = None, None, None, None

            if r and i < len(r.index):
                largs.index = r.index[i]
                largs.value = r.at_val[i]
                rtxt = r.get_label(largs, i)
                if rtxt:
                    rloc = loc_text_to_tuple(r.label_loc)

            if c and j < len(c.index):
                largs.index = c.index[j]
                largs.value = c.at_val[j]
                ctxt = c.get_label(largs, j)
                if ctxt:
                    cloc = loc_text_to_tuple(c.label_loc)

            if rtxt or ctxt:
                # labels show up in the same spot, merge them into one
                if rloc and cloc and all(np.array(rloc) == np.array(cloc)):
                    txt = ', '.join((rtxt, ctxt))
                    labels_ij.append((rloc, txt))
                else:
                    if rtxt:
                        labels_ij.append((rloc, rtxt))
                    if ctxt:
                        labels_ij.append((cloc, ctxt))

            if first_only:
                break

        # store all accumulated labels in row/col/label loc-specific list
        for k, (loc, txt) in enumerate(labels_ij):
            iv, ih = loc

            labels[i, j, iv, ih].append(txt)

            # Append kwargs that will be passed to ax.text()
            kwargs = {'verticalalignment': valign[iv],
                      'horizontalalignment': halign[ih],
                      'y': init_y[iv], 'x': init_x[ih],
                      's': txt}

            if len(text_kwargs[i, j, iv, ih]) > 0:
                y_prev = text_kwargs[i, j, iv, ih][-1]['y']
                # move up or down by 7.5% or axes area from previously
                # positioned label
                kwargs['y'] = y_prev + direction_y[iv] * 0.075

            # Add text style definitions
            kwargs.update(**styles[0].text)

            text_kwargs[i, j, iv, ih].append(kwargs)

    return labels, text_kwargs


def get_finite_ylim(values):
    """
    Compute joint ylim from list of arrays in values, taking into
    consideration only finite elements.

    Parameters
    ----------
    values : list
        List of arrays to process

    Returns
    -------

    """
    min_v = [np.amin(v[np.isfinite(v)]) for v in values]
    max_v = [np.amax(v[np.isfinite(v)]) for v in values]

    if len(min_v) > 0 and len(max_v) > 0:
        ymin = np.amin(min_v)
        ymax = np.amax(max_v)
    else:
        msg = 'Array(s) contain any finite values'
        raise ValueError(msg)

    return ymin, ymax


def get_ylim(nrow, ncol, ylim, extendy, sharey, maps, values):
    # Determine plot limits for y-axis
    if ylim is None:
        if sharey:
            # properly handle NaNs and Infs
            vv = [v for v in values if np.any(np.isfinite(v))]
            ylim = get_finite_ylim(vv)

            if extendy:
                ylim = bnd_extend(ylim, extendy)
            # repeat across all rows / cols
            ylim = np.tile(ylim, reps=(nrow, ncol, 1))

        else:
            # Compute ylims for each subplot; for this we need to find all
            # PMs that are to be shown in subplot (i, j), and find the
            # min. and max of their y-values
            ylim = np.ndarray((nrow, ncol, 2), dtype=float)
            rr, cc = np.ix_(range(nrow), range(ncol))
            for i, j in np.broadcast(rr, cc):
                # find all those pmaps which should be plotted in i, j
                r = np.array([i < p.nrow for p in maps])
                c = np.array([j < p.ncol for p in maps])
                use = np.logical_and(r, c)
                # properly handle NaNs and Infs
                vv = [v[i, j] for v in values[use]
                      if np.any(np.isfinite(v[i, j]))]
                ylim[i, j] = get_finite_ylim(vv)
    else:
        # User-provided ylim: if it's a simple tuple and sharey, broadcast
        # across all subplots
        if sharey:
            if len(ylim) != 2:
                raise ValueError('Argument ylim must be a length-2 tuple!')
            ylim = np.tile(ylim, reps=(nrow, ncol, 1))
        else:
            # Tile ylim as needed to obtain array dimension (nrow, ncol, 2)
            ylim = np.atleast_1d(ylim)
            if not (1 <= ylim.ndim <= 3):
                raise ValueError('ylim dimension must be between 1 and 3!')

            if ylim.ndim == 1:
                ylim = ylim[np.newaxis, np.newaxis]
            elif ylim.ndim == 2:
                # Insert column dimension
                ylim = ylim[:, np.newaxis]

            if ylim.shape[0] not in [1, nrow]:
                raise ValueError('Non-conformable argument ylim!')
            if ylim.shape[1] not in [1, ncol]:
                raise ValueError('Non-conformable argument ylim!')

            reps = [1, 1, 1]
            if ylim.shape[0] != nrow:
                reps[0] = nrow
            if ylim.shape[1] != ncol:
                reps[1] = ncol
            ylim = np.tile(ylim, reps)

    return ylim


def axis_plot_args(maps, styles):

    plot_kwargs = list()

    # normalized key names to be used as kwargs to plot()
    norm_keys = {'c': 'color', 'linewidth': 'lw', 'linestyle': 'ls',
                 'markersize': 'ms'}

    for p, s in zip(maps, styles):
        kwargs_i = list()
        overrides = dict()
        if p.layers and p.layers.plot_kwargs:
            # names that do not need to be normalized
            ov1 = {k: v for k, v in p.layers.plot_kwargs.items() if
                   k not in norm_keys}
            # normalize names
            ov2 = {norm_keys[k]: v for k, v in p.layers.plot_kwargs.items()
                   if k in norm_keys}
            overrides.update(ov1)
            overrides.update(ov2)

        for k in range(p.nlayer):
            defaults = {'ls': s.linestyle[k], 'lw': s.linewidth[k],
                        'color': s.color[k], 'alpha': s.alpha[k],
                        'marker': s.marker[k], 'ms': s.markersize[k],
                        'mec': s.mec[k]}
            defaults.update(overrides)
            kwargs_i.append(defaults)

        plot_kwargs.append(kwargs_i)

    return plot_kwargs


def plot_pm(pm, data, style=DefaultStyle(), trim_iqr=2.0, xlim=None, ylim=None,
            extendx=0.01, extendy=0.01, identity=False, sharey=True,
            legend=True, xlabel=None, ylabel=None, callback=None,
            label_first_only=True, **kwargs):
    """
    Plot multiple sets of data using their corrsponding plot maps and styles.

    Parameters
    ----------
    pm : array_like or PlotMap
        Mapping from data to plots. Accepts list of multile mappings.
    data : array_like
        Data to be plotted. Can be specified as list of several Numpy arrays.
    style : array_like or pydynopt.plot.styles.AbstractStyle, optional
        List of style to be applied to plots.
    trim_iqr : float, optional
        If not None, all observations outside of the interval
        defined by the median +/- `trim_iqr` * IQR will not be plotted.
    xlim : tuple
        Plot limits along x-axis
    ylim : tuple, optional
        Plot limits along y-axis
    extendx : float, optional
        If not None, limits of x-axis will be extended by a fraction
        `extendx' of the limits obtained from the data array.
    extendy : float
        If not None, limits of y-axis will be extended by a fraction
        `extendy' of the limits obtained from the data array.
    identity : bool
        Add identity function (45° line) to plot background. (optional)
    sharey : bool, optional
        If true, share common plot range on y-axis.
    legend : bool, optional
        It true, plot legend.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    callback : callable, optional
        If not None, this function will be invoked for each panel within
        the graph, proving access to the axis object, allowing for further
        customization.
    label_first_only : bool, optional
        If true, print annotations only for the first data set if multiple
        data sets are given.
    kwargs:
        Parameters passed to plot_grid.

    """

    # Convert input arguments to sequences.
    if not isinstance(data, (list, tuple)):
        tmp = (data, )
    else:
        tmp = tuple(data)

    # Manually copy this into an object-type ndarray
    data = np.ndarray((len(tmp), ), dtype=object)
    for i, d in enumerate(tmp):
        data[i] = d
    pm = np.atleast_1d(pm)
    style = np.atleast_1d(style)

    # Broadcast PlotMaps and data against each to support mapping multiple
    # arrays using the same PlotMap, or applying different PlotMaps to a
    # single array.
    bcast = np.broadcast(pm, data, style)
    maps = np.empty((bcast.size, ), dtype=object)
    data = np.empty_like(maps)
    styles = np.empty_like(maps)
    for i, (p, v, s) in enumerate(bcast):
        # extract plot data for given plot map to create specific values array
        # and PlotMap objects
        v_s, p_s = p.apply(v)
        maps[i] = p_s
        data[i] = v_s
        styles[i] = s

    # compute number of rows, cols and layers
    nrow = max(x.nrow for x in maps)
    ncol = max(x.ncol for x in maps)
    nlayer = max(x.nlayer for x in maps)

    # Generate labels for all subplots, store them on array with shape
    # (nrow, ncol, 3, 3) where the last 2 axis specify the label location
    # within each subplot
    labels, label_kwargs = row_col_labels(nrow, ncol, maps, styles,
                                          first_only=label_first_only)

    # Inter-quartile range
    trim_iqr = float(trim_iqr) if trim_iqr is not None else trim_iqr

    # Determine plot limits for x-axis
    if xlim is None:
        xmin = min(x.xaxis.at_val[0] for x in maps)
        xmax = max(x.xaxis.at_val[-1] for x in maps)
        xlim = (xmin, xmax)
    if extendx:
        xlim = bnd_extend(xlim, extendx)

    # Determine plot limits for y-axis
    ylim = get_ylim(nrow, ncol, ylim, extendy, sharey, maps, data)

    # Construct kwargs to be passed to plot() invocation for each plot map
    # and layer
    plot_kwargs = axis_plot_args(maps, styles)

    # label for x-axis; find label attached to some xaxis object if xlabel
    # was not passed as an argument.
    if xlabel is None:
        for pm in maps:
            xlabel = pm.xaxis.get_label()
            # Exit as soon as we found something that is not None, empty etc.
            if xlabel:
                break

    do_callback = callback is not None and isinstance(callback, Callable)

    def subplot(ax, idx):
        i, j = idx

        # plot all plot objects sequentially
        for i_m, (p, v, s) in enumerate(zip(maps, data, styles)):
            # True if this is first obj. in sequence, or action should not be
            # limited to first-only
            first_or_all = not label_first_only or i_m == 0

            for k, yy in enumerate(v[i, j]):
                if legend and first_or_all and (p.layers and k < p.nlayer):
                    lidx = p.layers.index[k]
                    lval = p.layers.at_val[k]
                    # construct label argument for layer k
                    largs = LabelArgs(i, j, k, lidx, lval)
                    txt = p.layers.get_label(largs, k)
                else:
                    txt = None

                # kwargs such as color, line style, etc.
                plot_kw = plot_kwargs[i_m][k]

                ax.plot(p.xaxis.at_val, yy, label=txt, **plot_kw)

                if do_callback:
                    callback(ax, (i, j, k, i_m), p.xaxis.at_val, yy, p, s)

        # iterate over all potential label locations in subplot,
        # check whether something should be plotted there, at plot it.
        # There is a possibly empty list in each array cell.
        for iv, ih in np.broadcast(np.arange(3)[:, None], np.arange(3)[None]):
            for lbl in label_kwargs[i, j, iv, ih]:
                # Add this otherwise position relative to current
                # axes will not work
                lbl['transform'] = ax.transAxes
                ax.text(**lbl)

        # Plot 45° line if requested
        if identity:
            plot_identity(ax, xlim[0], xlim[1])

        ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 3))

    plot_grid(subplot, nrow=nrow, ncol=ncol, style=style[0], sharey=sharey,
              legend=legend, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
              **kwargs)
