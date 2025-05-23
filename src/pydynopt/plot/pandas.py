"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import collections.abc
from collections.abc import Sequence, Mapping, Iterable, Callable
import math
from typing import Union, Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .styles import DefaultStyle, AbstractStyle
from .baseplots import plot_grid

from ..utils import anything_to_list, anything_to_tuple

__all__ = ['plot_dataframe']


def _text_loc_to_kwargs(loc):
    """
    Map location text to corresponding arguments to MPL's text() method.

    Parameters
    ----------
    loc : str

    Returns
    -------
    dict
    """

    map_vert = {
        'upper': {'y': 0.95, 'va': 'top'},
        'top': {'y': 0.95, 'va': 'top'},
        'center': {'y': 0.5, 'va': 'center'},
        'lower': {'y': 0.05, 'va': 'bottom'},
        'bottom': {'y': 0.05, 'va': 'bottom'},
    }

    map_hor = {
        'left': {'x': 0.05, 'ha': 'left'},
        'center': {'x': 0.5, 'ha': 'center'},
        'right': {'x': 0.95, 'ha': 'right'},
    }

    vert, hor = loc.lower().split()

    kwargs = map_hor[hor]
    kwargs.update(map_vert[vert])

    return kwargs


def _get_yerr(data, moment_name, yvalues=None):
    """
    Return y-data for errors bars in a form that can be passed as yerr
    argument to MPL's errorbar().

    Parameters
    ----------
    data : pd.DataFrame
    moment_name : str
        Name of point estimate (moment) for which SEs/CIs
        should be returned.
    yvalues : array_like, optional
        Optional actual y-values. This is needed to compute the correct
        values for yerr if sampling variance is stored as CI in `data`
        as opposed to standard errors.

    Returns
    -------
    tuple of np.ndarray

    """

    columns = data.columns.get_level_values(0)
    yerr = None

    if not moment_name:
        return yerr

    se_name = [n for n in columns if n.lower() == f'{moment_name.lower()}_se']
    if se_name:
        se = data[se_name[0]].to_numpy()
        yerr_lb = 1.96 * se
        yerr_ub = 1.96 * se
        yerr = (yerr_lb, yerr_ub)

    ci_lb_name = [n for n in columns if n.lower() == f'{moment_name.lower()}_ci_lb']
    ci_ub_name = [n for n in columns if n.lower() == f'{moment_name.lower()}_ci_ub']

    if ci_lb_name and ci_ub_name:
        # Caller expects values to be centered around y-values
        # Lower bound: drawn as yvalues - yerr_lb, so we need to return
        # CI_lb = yvalues - yerr_lb => yerr_lb = yvalues - CI_lb
        yerr_lb = yvalues - data[ci_lb_name[0]].to_numpy()
        # Upper bound: drawn as yvalues + yerr_ub, so we need to return
        # CI_ub = yvalues + yerr_ub => yerr_ub = CI_ub - yvalues
        yerr_ub = data[ci_ub_name[0]].to_numpy() - yvalues
        if np.any(np.isfinite(yerr_lb)) and np.any(np.isfinite(yerr_ub)):
            yerr = np.stack((yerr_lb, yerr_ub))

    return yerr


def _find_name(df, fmt='__{:06d}'):
    """
    Return a variable name that can be used as a column or index name
    for the given DataFrame without clashing with any of the existing
    index or top-level column names.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
    fmt : str, optional
        Format used to generate a new name. Must accept a single integer
        argument.

    Returns
    -------
    str
    """

    names_present = list(df.index.names)

    if isinstance(df, pd.DataFrame):
        columns = list(df.columns.get_level_values(0))
        names_present += columns

    counter = 0
    while True:
        name = fmt.format(counter)
        if name not in names_present:
            return name
        counter += 1


def _process_slice(df, varlist=None, labels=None, order=None):
    """

    Parameters
    ----------
    varlist : str or Iterable of str, optional
    labels
    order

    Returns
    -------
    df : pd.DataFrame
    varname : str
    labels : dict
    order : np.ndarray
    """

    varlist = anything_to_list(varlist, force=True)

    # Original index names
    index_orig = list(df.index.names)

    if len(varlist) == 0:
        # No variable given, just create a degenerate variable and add it to
        # the index
        varname = _find_name(df)
        labels = dict()
        value = 0
        order = np.array([value])
        # Insert degenerate name into index
        df = pd.concat((df,), axis=0, names=[varname], keys=[value])

    elif len(varlist) == 1:
        # Single variable, no need to create new one
        varname = varlist[0]

        if varname not in index_orig:
            df = df.set_index(varname, append=True)

        if order is None:
            values = df.index.get_level_values(varname)
            order = values.drop_duplicates(keep='first').to_numpy()

        if labels is None:
            labels = dict()
        elif isinstance(labels, str):
            labels = {v: labels.format(v) for v in order}
        elif isinstance(labels, collections.abc.Mapping):
            pass
        elif isinstance(labels, Sequence):
            labels = {k: labels[i] for i, k in enumerate(order)}
        elif callable(labels):
            labels = {v: labels(**{varname: v}) for v in order}
    else:
        # Multiple variables given, we need to consolidate them into a single
        # one respecting any sort order, etc.
        varname = _find_name(df)

        df_values = df.reset_index()[varlist].copy()
        # Drop any hierarchical column index
        df_values.columns = varlist
        # Drop duplicates, this should preserve sort order
        df_values_uniq = df_values.drop_duplicates(keep='first').copy()
        # Sort in given variable order
        df_values_uniq = df_values_uniq.sort_values(varlist)

        if order is None:
            df_values_uniq[varname] = np.arange(df_values_uniq.shape[0])
            df_values = df_values.merge(df_values_uniq, on=varlist, how='left')

            df[varname] = df_values[varname].to_numpy()

            order = np.arange(df_values_uniq.shape[0])
        else:
            # Caller imposed an order on (some) of the variables in varlist
            if isinstance(order, collections.abc.Mapping):
                # Already in desired format
                pass
            elif isinstance(order, collections.abc.Sequence):
                # Needs to be a sequence of iterable items
                if len(order) != len(varlist):
                    msg = 'order and variable names must be of equal length!'
                    raise ValueError(msg)
                order = {name: v for name, v in zip(varlist, order)}
            else:
                raise ValueError('order format not understood!')

            df_tmp = None
            for var in varlist:
                if var in order:
                    df_new = pd.DataFrame({var: np.atleast_1d(order[var])})
                else:
                    values = df_values[[var]].drop_duplicates(keep='first')
                    df_new = pd.DataFrame({var: values})

                if df_tmp is None:
                    df_tmp = df_new
                else:
                    df_tmp = pd.merge(df_tmp, df_new, how='cross')

            ivalues = np.arange(df_tmp.shape[0])
            df_tmp[varname] = ivalues

            df_values = pd.merge(df_values, df_tmp, on=varlist, how='left')
            # "merge" on observation order
            df[varname] = df_values[varname].to_numpy()
            # Drop any rows that were not selected by order argument
            df = df.loc[df[varname].notna()].copy()
            df[varname] = df[varname].astype(int)

            order = ivalues

        df = df.set_index(varname, append=True)

        df_values_uniq = df_values.drop_duplicates(keep='first')
        if varname in df_values_uniq.columns:
            df_values_uniq = df_values_uniq.set_index(varname)

        if isinstance(labels, str):
            lbl = {}
            for row in df_values_uniq.itertuples():
                dct = row._asdict()
                # Index attribute of names tuple is called 'Index'
                i = dct.pop('Index')
                lbl[i] = labels.format(**dct)
            labels = lbl
        elif callable(labels):
            lbl = {}
            for row in df_values_uniq.itertuples():
                dct = row._asdict()
                # Index attribute of names tuple is called 'Index'
                i = dct.pop('Index')
                lbl[i] = labels(**dct)
            labels = lbl
        elif isinstance(labels, Mapping):
            lbl = {}
            for row in df_values_uniq.itertuples():
                dct = row._asdict()
                # Index attribute of names tuple is called 'Index'
                i = dct.pop('Index')
                lbl[i] = ', '.join(labels[k][v] for k, v in dct.items())
            labels = lbl
        elif labels is None:
            labels = {}
        else:
            raise ValueError('Unsupported labels format')

    return df, varname, labels, order


def _process_dep_vars(df, yvar=None, moment=None):
    df = df.copy()

    yvars = anything_to_list(yvar)

    level_names = ['Variable', 'Moment']

    if isinstance(df, pd.Series):
        yvar = yvar if yvar is not None else _find_name(df)
        df = df.to_frame(yvar)
        df.columns.set_names([level_names[0]], inplace=True)

    if df.columns.nlevels == 1:
        if yvar and moment:
            msg = 'Both yvar and moment given, but column index is non-hierarchical'
            raise ValueError(msg)

        varlist = df.columns.get_level_values(0).unique()

        lname = None
        if df.columns.name:
            lname = anything_to_list(df.columns.name)[0]

        if lname and lname.lower().startswith('variable'):
            if yvars:
                for yvar in yvars:
                    if yvar not in varlist:
                        raise ValueError(f'{yvar} not in DataFrame columns')
            else:
                yvars = varlist
            moment = '__mom'
            midx = pd.MultiIndex.from_product((varlist, (moment,)), names=level_names)
            df.columns = midx
        elif lname and lname.lower().startswith('moment'):
            if moment is not None and moment not in varlist:
                raise ValueError(f'{moment} not in DataFrame columns')
            yvar = _find_name(df)
            yvars = [yvar]
            midx = pd.MultiIndex.from_product((yvars, varlist), names=level_names)
            df.columns = midx
        else:
            if yvars is not None and all(yvar in varlist for yvar in yvars):
                moment = '__mom'
                midx = pd.MultiIndex.from_product(
                    (varlist, (moment,)), names=level_names
                )
                df.columns = midx
            elif moment in varlist:
                yvar = _find_name(df)
                yvars = [yvar]
                midx = pd.MultiIndex.from_product((yvars, varlist), names=level_names)
                df.columns = midx
            elif yvar or moment:
                msg = 'yvar/moment arguments not compatible with DataFrame'
                raise ValueError(msg)
            else:
                # Nothing specified, assume that columns are variables
                yvars = varlist
                moment = '__mom'
                midx = pd.MultiIndex.from_product((yvars, (moment,)), names=level_names)
                df.columns = midx

    elif df.columns.nlevels == 2:
        names = {}
        for name in df.columns.names:
            if name and name.lower().startswith('variable'):
                names[name] = level_names[0]
            elif name and name.lower().startswith('moment'):
                names[name] = level_names[1]

        if len(names) == 0:
            # Note of the names match, force rename
            df.columns.names = level_names
        else:
            names_upd = [names.get(x, x) for x in df.columns.names]
            df.columns.names = names_upd

        for i, name in enumerate(level_names):
            if df.columns.names[i] != level_names[i]:
                raise ValueError('DataFrame column index not understood')
    else:
        raise ValueError('DataFrame column index not understood')

    # Sanity check
    columns = list(df.columns.get_level_values(0).unique())
    if isinstance(yvars, collections.abc.Iterable):
        for yvar in yvars:
            if yvar not in columns:
                raise ValueError(f'{yvar} not in DataFrame columns')
    else:
        yvars = columns

    return df, yvars, moment


def _find_moment_name(df):
    """
    Guess the name of the moment to be plotted for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
    """

    columns = df.columns.get_level_values(0).unique()
    columns_dct = {name.lower(): name for name in columns}

    cand = [
        name
        for name in columns_dct.keys()
        if name and not any(name.endswith(f'_{s}') for s in ('se', 'ci_lb', 'ci_ub'))
    ]

    if len(cand) == 1:
        mname = columns_dct[cand[0]]
    else:
        # Just take the first column
        mname = columns[0]

    return mname


def _get_scatter_size(scatter_size, yvar, data, default):
    """
    Return the marker size for scatter plots, either as a uniform constant,
    or as values of a given column from a DataFrame.

    Parameters
    ----------
    scatter_size : str, optional
    data : pd.DataFrame
    default : float

    Returns
    -------
    float or np.ndarray
    """

    size = default
    if scatter_size is None:
        return size

    if scatter_size in data[yvar].columns.get_level_values(0):
        size = data[(yvar, scatter_size)].to_numpy().flatten()
        # Prevent non-finite sizes due to NaN data as this will break legend
        size[~np.isfinite(size)] = 0.0
    elif scatter_size in data.columns.get_level_values(0):
        size = data[scatter_size].to_numpy().flatten()
        # Prevent non-finite sizes due to NaN data as this will break legend
        size[~np.isfinite(size)] = 0.0
    else:
        try:
            size = float(scatter_size)
        except:
            pass

    return size


def plot_dataframe(
    df: pd.DataFrame,
    xvar: str = None,
    yvar: Optional[Union[str, Sequence[str]]] = None,
    yvar_labels: Union[Sequence[str], Mapping[str, str]] = None,
    moment: Optional[str] = None,
    by: Optional[str | Sequence[str]] = None,
    by_labels: Optional[Union[Sequence[str], Mapping, Callable]] = None,
    by_order: Optional[Sequence] = None,
    over: Optional[Union[str, Sequence[str]]] = None,
    over_order: Optional[Sequence] = None,
    over_labels: Optional[Union[Sequence[str], Mapping, Callable]] = None,
    over_label_pos: Optional[str | Sequence[str]] = None,
    ncol: Optional[int] = None,
    jitter: Optional[float] = None,
    plot_type: Optional[Union[str, Mapping[str, str], Sequence[str]]] = None,
    callback: Optional[Callable] = None,
    callback_args: tuple = (),
    scatter_size: Union[str, float] = 'size',
    style: Union[
        Sequence[AbstractStyle], Mapping[str, AbstractStyle], AbstractStyle
    ] = DefaultStyle(),
    **kwargs,
) -> np.ndarray[Axes]:
    """
    Plot selected variables in DataFrame, optionally disaggregating by groups.

    Parameters
    ----------
    df : pd.DataFrame
    xvar : str, optional
        Variable or index name storing x-values.
    yvar : str or Iterable of str, optional
        Column names storing y-values to be plotted.
    yvar_labels : str or Iterable of str or Mapping, optional
        Variable labels
    moment : str, optional
        Name of moment to be plotted
    by : str or Iterable of str, optional
        Variable or index name by which to disaggregate within individual
        plot panels.
    by_labels : Mapping or Sequence or str or callable, optional
        Pretty labels to be used in legend.
    by_order : Sequence or str, optional
        Values of categorical variable `by` which specify
        plotting order (useful to harmonize legend and plot order)
    over : str or Iterable or str, optional
        Variable or index name by which to disaggregate data into separate
        panels
    over_order : Sequence or dict, optional
        Values of categorical variable `over` which specify
        plotting order.
    over_labels : Mapping or Sequence or str or callable, optional
        Mapping of values of `over` variable to pretty labels.
    over_label_pos : str, optional
        Position of annotation text containing the `over` value for current
        panel.
    ncol : int, optional
        Number of columns used to arrange plot panels (ignored unless `over`
        is given)
    jitter : float, optional
        Perturb x-location by given fraction of x-range (ignored unless `by`
        is given)
    plot_type : str or Mapping or Iterable of str, optional
        Plot type ('bar', 'area' or None, the default)
    callback : callable, optional
        If not None, will be called at the end of plotting code executed
        for each panel with arguments callback(ax, idx).
    callback_args : tuple, optional
        If not None, tuple will be expanded and passed as additional positional
        arguments to `callback()`
    scatter_size : str or float, optional
        If string, it is interpreted as a column name in `df`
        with values to be interpreted as marker sizes. If float,
        the value is used as a uniform marker size.
    style : pydynopt.plot.styles.AbstractStyle or Iterable or Mapping, optional
    kwargs :
        Keyword arguments passed to plot_grid()

    """

    jitter = float(jitter) if jitter is not None else None

    df = df.copy()

    df, by_var, by_labels, by_order = _process_slice(df, by, by_labels, by_order)
    df, over_var, over_labels, over_order = _process_slice(
        df, over, over_labels, over_order
    )

    df, yvars, moment_name = _process_dep_vars(df, yvar, moment)

    if xvar is None:
        if df.index.nlevels > 1:
            raise ValueError("Cannot determine x-variable")
        elif not df.index.name:
            xvar = '_xvalues'
            df.index.set_names([xvar], inplace=True)
        else:
            xvar = df.index.name
    else:
        if xvar in df.columns and xvar not in df.index:
            # Append xvar to index where it's expected by plotting functions.
            # Perform this indirectly so that MultiIndex columns work as well
            midx = df.index.to_frame(index=False)
            midx[xvar] = df[xvar].to_numpy()
            midx = pd.MultiIndex.from_frame(midx)
            df.index = midx
            del df[xvar]

    varlist = [over_var, by_var, xvar]
    index_other = [name for name in df.index.names if name not in varlist]

    # Reorder index levels (specific order is required below), push
    # user-given index levels that are not required to the end.
    df = df.reorder_levels(varlist + index_other)

    # --- Fix plot types, labels and styles for each variable ---

    # Plot types
    if plot_type is None:
        plot_type = {v: '' for v in yvars}
    elif isinstance(plot_type, str):
        # Same plot type for all variables
        plot_type = {v: plot_type for v in yvars}
    elif isinstance(plot_type, collections.abc.Mapping):
        # Expected data type
        pass
    elif isinstance(plot_type, collections.abc.Iterable):
        # Plot types passed as list, assumed in same order as variables
        plot_type = {v: t for v, t in zip(yvars, plot_type)}
    else:
        raise ValueError('Unsupported plot_type value')

    # Process variable labels
    if isinstance(yvar_labels, str):
        yvar_labels = {yvars[0]: yvar_labels}
    elif isinstance(yvar_labels, collections.abc.Mapping):
        # Expected data type
        pass
    elif isinstance(yvar_labels, collections.abc.Iterable):
        # Convert from list to dict
        yvar_labels = {v: lbl for v, lbl in zip(yvars, yvar_labels)}
    elif yvar_labels is None and not by_labels:
        yvar_labels = {v: v for v in yvars}
    elif yvar_labels is None:
        pass
    else:
        raise ValueError('Unsupported yvar_labels value')

    # Replicate style for all variables, if needed
    if isinstance(style, AbstractStyle):
        # Will be propagated / converted to dict below
        style = [style]
    if isinstance(style, collections.abc.Mapping):
        # Expected data type
        styles = style
    elif isinstance(style, collections.abc.Sequence):
        styles = anything_to_list(style)
        if len(styles) == 1 and len(yvars) > 1:
            styles = styles * len(yvars)
        # Convert to dict
        styles = {v: s for v, s in zip(yvars, styles)}
    else:
        raise ValueError('Unsupported style vale')

    # Determine number of rows and columns from number of vars to be plotted.
    ncol = len(over_order) if not ncol else ncol
    nrow = int(math.ceil(len(over_order) / ncol))
    npanels = len(over_order)

    def fplot(ax, idx):
        i, j = idx

        ipanel = i * ncol + j

        if ipanel >= npanels:
            # Skip any residual panels that are not needed
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(bottom=False, left=False)
            ax.set_frame_on(False)
            ax.grid(None)
            return

        # Restrict to data plotted in particular panel
        df_panel = df.xs(over_order[ipanel], level=over_var, axis=0)

        for ivar, yvar in enumerate(yvars):
            # Variable-specific style
            style = styles[yvar]
            data = df_panel[yvar]

            if moment_name:
                mname = moment_name
            else:
                mname = _find_moment_name(data)

            df_moment = data[mname]

            barwidth = 1.0

            for k, by_value in enumerate(by_order):
                mask = df_moment.index.isin([by_value], level=by_var)
                if not mask.any():
                    # For this panel there are no observations for the given layer
                    continue
                yvalues = df_moment[mask].to_numpy()
                xvalues = df_moment[mask].index.get_level_values(xvar).to_numpy()

                # Legend labels. by-labels take precedence due to backwards
                # compatibility!
                leglbl = None
                if by_labels and yvar_labels:
                    bylbl = by_labels.get(by_value, by_value)
                    vlbl = yvar_labels.get(yvar, yvar)
                    leglbl = f'{vlbl}: {bylbl}'
                elif by_labels:
                    leglbl = by_labels.get(by_value, by_value)
                elif yvar_labels:
                    leglbl = yvar_labels.get(yvar, yvar)
                else:
                    # Fallback: use default string representation of -by- value
                    leglbl = f'{by_value}'

                if not np.any(np.isfinite(xvalues) & np.isfinite(yvalues)):
                    # Disable artificial legend labels when nothing is displayed
                    leglbl = None

                if plot_type[yvar] == 'bar':
                    if xvalues.size > 1:
                        dx = np.amin(xvalues[1:] - xvalues[:-1]) * 0.8
                        barwidth = dx / len(by_order)
                        if len(by_order) % 2 == 0:
                            left = barwidth * (len(by_order) - 1) / 2
                        else:
                            left = barwidth * (len(by_order) // 2)

                        xvalues = xvalues - left + barwidth * k
                elif jitter:
                    dx = xvalues[-1] - xvalues[0]
                    if len(by_order) % 2 == 0:
                        left = dx * jitter * (len(by_order) + 1) / 2
                    else:
                        left = dx * jitter * (len(by_order) // 2)

                    offset = dx * jitter * k
                    xvalues = xvalues - left + offset

                yerr = _get_yerr(data.loc[by_value], mname, yvalues)

                if plot_type[yvar] == 'bar':
                    kw = style.bar_kwargs[k]

                    bw = barwidth * (1.0 - 2.0 * style.barmargin)

                    ax.bar(xvalues, yvalues, width=bw, yerr=yerr, label=leglbl, **kw)

                elif plot_type[yvar] == 'area' and yerr is not None:
                    ylb = yvalues - yerr[0]
                    yub = yvalues + yerr[1]
                    isfin = any(np.isfinite(ylb) & np.isfinite(yub))
                    if isfin:
                        kw = style.fill_between_face_kwargs[k]
                        ax.fill_between(xvalues, ylb, yub, **kw)

                        # Create lower and upper boundaries manually
                        kw = style.fill_between_edge_kwargs[k].copy()
                        kw['zorder'] += 10
                        ax.plot(xvalues, ylb, **kw)
                        ax.plot(xvalues, yub, **kw)

                    kw = style.plot_kwargs[k].copy()
                    kw['zorder'] += 20
                    ax.plot(xvalues, yvalues, label=leglbl, **kw)

                elif plot_type[yvar] == 'scatter':
                    size = _get_scatter_size(
                        scatter_size, yvar, df_panel.loc[by_value], style.markersize[k]
                    )

                    if style.split_scatter:
                        # Plot face component of scatter
                        kw = style.scatter_face_kwargs[k]
                        ax.scatter(xvalues, yvalues, s=size, **kw)

                        # Plot edge component of scatter
                        kw = style.scatter_edge_kwargs[k]
                        kw['zorder'] += 1
                        ax.scatter(xvalues, yvalues, s=size, label=leglbl, **kw)
                    else:
                        # Default: plot edges and faces in single call
                        kw = style.scatter_kwargs[k]
                        ax.scatter(xvalues, yvalues, s=size, label=leglbl, **kw)

                else:
                    kw = style.errorbar_kwargs[k]
                    # Check whether style includes a marker
                    marker = kw.get('marker', None)
                    has_marker = bool(marker)
                    if marker:
                        marker = marker.lower().strip()
                        has_marker = marker != '' and marker != 'none'

                    # Split between line/marker components only if marker is
                    # present
                    if style.split_scatter and has_marker:
                        kw = style.errorbar_no_marker_kwargs[k]
                        ax.errorbar(xvalues, yvalues, yerr=yerr, label=leglbl, **kw)

                        kw = style.marker_no_line_kwargs[k].copy()
                        if 'zorder' in kw:
                            kw['zorder'] += 1
                        else:
                            kw['zorder'] = 1
                        ax.plot(xvalues, yvalues, **kw)
                    else:
                        kw = style.errorbar_kwargs[k]
                        ax.errorbar(xvalues, yvalues, yerr=yerr, label=leglbl, **kw)

        # --- Label over group ---

        lbl = over_labels.get(over_order[ipanel], None)
        positions = anything_to_tuple(over_label_pos, force=True)
        labels = anything_to_tuple(lbl, force=True)
        for lbl, pos in zip(labels, positions):
            _style: AbstractStyle = styles[yvars[0]]
            if pos.lower() == 'title':
                ax.set_title(lbl, **_style.title)
            else:
                kw = _style.text.copy()
                kw.update(_text_loc_to_kwargs(pos))
                kw['s'] = lbl
                kw['transform'] = ax.transAxes

                ax.text(**kw)

        # --- Call any user-provided callback function ---

        if callable(callback):
            callback(ax, idx, df_panel, styles[yvars[0]], *callback_args)

    kwargs_default = {'style': styles[yvars[0]]}

    kwargs_default.update(kwargs)
    kwargs = kwargs_default

    return plot_grid(fplot, nrow, ncol, **kwargs)
