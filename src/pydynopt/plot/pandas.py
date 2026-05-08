"""
Utilities to plot pandas objects with grouped panels and shared style logic.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import collections.abc
from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
import logging
import math
from typing import Any

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from ..utils import anything_to_list, anything_to_tuple
from .baseplots import plot_grid
from .styles import AbstractStyle, DefaultStyle

__all__ = ['plot_dataframe']

type LabelMap = Mapping[object, Any]
type LabelFormatter = Callable[..., Any]
type LabelSpec = str | Sequence[str] | LabelMap | LabelFormatter | None
type OrderSpec = Sequence[object] | Mapping[str, Sequence[object]] | None
type StyleSpec = (
    AbstractStyle | Sequence[AbstractStyle] | Mapping[str, AbstractStyle] | None
)


def _text_loc_to_kwargs(loc: str) -> dict[str, float | str]:
    """
    Map location text to corresponding arguments to MPL's text() method.

    Parameters
    ----------
    loc
        Location string.

    Returns
    -------
    Arguments to MPL's text() method.
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

    return map_hor[hor] | map_vert[vert]


def _get_yerr(
    data: pd.DataFrame, moment_name: str, yvalues: np.ndarray | None = None
) -> np.ndarray | None:
    """
    Return y-data for error bars in a form that can be passed to MPL's errorbar().

    Parameters
    ----------
    data
        Input DataFrame.
    moment_name
        Name of point estimate (moment) for which SEs/CIs should be returned.
    yvalues
        Optional actual y-values. This is needed to compute the correct
        values for yerr if sampling variance is stored as CI in `data`
        as opposed to standard errors.

    Returns
    -------
    Error bar values with shape ``(2, n)`` or ``None`` if no uncertainty data are available.
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
        yerr = np.stack((yerr_lb, yerr_ub))

    ci_lb_name = [n for n in columns if n.lower() == f'{moment_name.lower()}_ci_lb']
    ci_ub_name = [n for n in columns if n.lower() == f'{moment_name.lower()}_ci_ub']

    if ci_lb_name and ci_ub_name and yvalues is not None:
        # Caller expects values to be centered around y-values
        # Lower bound: drawn as yvalues - yerr_lb, so we need to return
        # CI_lb = yvalues - yerr_lb => yerr_lb = yvalues - CI_lb
        yerr_lb = yvalues - data[ci_lb_name[0]].to_numpy()
        # Upper bound: drawn as yvalues + yerr_ub, so we need to return
        # CI_ub = yvalues + yerr_ub => yerr_ub = CI_ub - yvalues
        yerr_ub = data[ci_ub_name[0]].to_numpy() - yvalues
        if np.any(np.isfinite(yerr_lb)) and np.any(np.isfinite(yerr_ub)):
            yerr = np.stack((yerr_lb, yerr_ub))

            if np.any(yerr < 0.0):
                logger = logging.getLogger('pydynopt.plot')
                logger.warning('Clipping negative yerr values to 0.0')
                yerr = np.clip(yerr, 0.0, np.inf)

    return yerr


def _find_name(df: pd.DataFrame | pd.Series, fmt: str = '__{:06d}') -> str:
    """
    Return a variable name that can be used as a column or index name.

    The name is guaranteed not to clash with any of the existing
    index or top-level column names.

    Parameters
    ----------
    df
        Input DataFrame or Series.
    fmt
        Format used to generate a new name. Must accept a single integer
        argument.

    Returns
    -------
    Generated unique variable name.
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


def _process_slice(
    df: pd.DataFrame,
    varlist: str | Iterable[str] | None = None,
    labels: LabelSpec = None,
    order: OrderSpec = None,
) -> tuple[pd.DataFrame, str, Mapping[object, object], np.ndarray]:
    """
    Process and normalize slice variables, labels, and ordering.

    Parameters
    ----------
    df
        Input DataFrame.
    varlist
        Slice variables.
    labels
        Label specification for resulting grouped values.
    order
        Ordering specification for grouped values.

    Returns
    -------
    df
        Processed DataFrame.
    varname
        Name of the grouped/slice variable.
    labels
        Mapping of grouped values to pretty labels.
    order
        Ordering of grouped values.
    """
    varlist = anything_to_list(varlist, force=True)

    # Original index names
    index_orig = list(df.index.names)

    labels_val: Any = labels
    order_val: Any = order

    if len(varlist) == 0:
        # No variable given, just create a degenerate variable and add it to
        # the index
        varname = _find_name(df)
        labels_val = {}
        value = 0
        order_val = np.array([value])
        # Insert degenerate name into index
        df = pd.concat((df,), axis=0, names=[varname], keys=[value])

    elif len(varlist) == 1:
        # Single variable, no need to create new one
        varname = varlist[0]

        if varname not in index_orig:
            df = df.set_index(varname, append=True)

        if order_val is None:
            values = df.index.get_level_values(varname)
            order_val = values.drop_duplicates(keep='first').to_numpy()

        if labels_val is None:
            labels_val = {}
        elif isinstance(labels_val, str):
            labels_val = {v: labels_val.format(v) for v in order_val}
        elif isinstance(labels_val, Mapping):
            pass
        elif isinstance(labels_val, Sequence):
            labels_val = {k: labels_val[i] for i, k in enumerate(order_val)}
        elif callable(labels_val):
            labels_val = {v: labels_val(**{varname: v}) for v in order_val}
        else:
            raise ValueError('Unsupported labels format')
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

        if order_val is None:
            df_values_uniq[varname] = np.arange(df_values_uniq.shape[0])
            df_values = df_values.merge(df_values_uniq, on=varlist, how='left')

            df[varname] = df_values[varname].to_numpy()

            order_val = np.arange(df_values_uniq.shape[0])
        else:
            # Caller imposed an order on (some) of the variables in varlist
            if isinstance(order_val, collections.abc.Mapping):
                # Already in desired format
                pass
            elif isinstance(order_val, collections.abc.Sequence):
                # Needs to be a sequence of iterable items
                if len(order_val) != len(varlist):
                    msg = 'order and variable names must be of equal length!'
                    raise ValueError(msg)
                order_val = dict(zip(varlist, order_val, strict=False))
            else:
                raise ValueError('order format not understood!')

            df_tmp: Any = None
            for var in varlist:
                if var in order_val:
                    val: Any = order_val[var]
                    df_new = pd.DataFrame({var: np.atleast_1d(val)})
                else:
                    df_new = df_values[[var]].drop_duplicates(keep='first')

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

            order_val = ivalues

        df = df.set_index(varname, append=True)

        df_values_uniq = df_values.drop_duplicates(keep='first')
        if varname in df_values_uniq.columns:
            df_values_uniq = df_values_uniq.set_index(varname)

        if isinstance(labels_val, str):
            lbl = {}
            for row in df_values_uniq.itertuples():
                row_any: Any = row
                dct = row_any._asdict()
                # Index attribute of names tuple is called 'Index'
                i = dct.pop('Index')
                lbl[i] = labels_val.format(**dct)
            labels_val = lbl
        elif callable(labels_val):
            lbl = {}
            labels_func: Any = labels_val
            for row in df_values_uniq.itertuples():
                row_any: Any = row
                dct = row_any._asdict()
                # Index attribute of names tuple is called 'Index'
                i = dct.pop('Index')
                lbl[i] = labels_func(**dct)
            labels_val = lbl
        elif isinstance(labels_val, Mapping):
            lbl = {}
            for row in df_values_uniq.itertuples():
                row_any: Any = row
                dct = row_any._asdict()
                # Index attribute of names tuple is called 'Index'
                i = dct.pop('Index')
                lbl[i] = ', '.join(labels_val[k][v] for k, v in dct.items())
            labels_val = lbl
        elif labels_val is None:
            labels_val = {}
        else:
            raise ValueError('Unsupported labels format')

    ret_labels: Mapping[object, object] = labels_val
    ret_order: np.ndarray = order_val
    return df, varname, ret_labels, ret_order


def _process_dep_vars(
    df: pd.DataFrame | pd.Series,
    yvar: str | Sequence[str] | None = None,
    moment: str | None = None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """
    Normalize dependent-variable columns to a two-level MultiIndex.

    Parameters
    ----------
    df
        Input data with variables and moments either in columns or index.
    yvar
        Variables to select.
    moment
        Moment name to select.

    Returns
    -------
    df
        Normalized DataFrame.
    yvars
        Selected variable names.
    moment
        Selected moment name.
    """
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
            lname_list = anything_to_list(df.columns.name)
            lname = lname_list[0] if lname_list else None

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
            name = str(name)
            if name and name.lower().startswith('variable'):
                names[name] = level_names[0]
            elif name and name.lower().startswith('moment'):
                names[name] = level_names[1]

        if len(names) == 0:
            # Note of the names match, force rename
            df.columns.names = level_names
        else:
            names_upd = [names.get(str(x), str(x)) for x in df.columns.names]
            df.columns.names = names_upd

        for i, _ in enumerate(level_names):
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

    yvars_list: list[str] = list(yvars)
    return df, yvars_list, moment


def _find_moment_name(df: pd.DataFrame) -> str:
    """
    Guess the name of the moment to be plotted for the given DataFrame.

    Parameters
    ----------
    df
        Input DataFrame.

    Returns
    -------
    Guessed name of the moment.
    """
    columns = df.columns.get_level_values(0).unique()
    columns_dct = {name.lower(): name for name in columns}

    cand = [
        name
        for name in columns_dct
        if name and not any(name.endswith(f'_{s}') for s in ('se', 'ci_lb', 'ci_ub'))
    ]

    return columns_dct[cand[0]] if len(cand) == 1 else columns[0]


def _get_scatter_size(
    scatter_size: str | float | None,
    yvar: str,
    data: pd.DataFrame,
    default: float,
) -> float | np.ndarray:
    """
    Return the marker size for scatter plots.

    The size can be either a uniform constant, or values of a given column
    from a DataFrame.

    Parameters
    ----------
    scatter_size
        Name of the column containing marker sizes, or a uniform size.
    yvar
        Name of the dependent variable.
    data
        Input DataFrame.
    default
        Default marker size.

    Returns
    -------
    Marker size as a uniform float or an array of sizes.
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
        with contextlib.suppress(TypeError, ValueError):
            size = float(scatter_size)

    return size


def plot_dataframe(
    df: pd.DataFrame,
    xvar: str | None = None,
    yvar: str | Sequence[str] | None = None,
    yvar_labels: str | Sequence[str] | Mapping[str, str] | None = None,
    moment: str | None = None,
    by: str | Sequence[str] | None = None,
    by_labels: LabelSpec = None,
    by_order: OrderSpec = None,
    over: str | Sequence[str] | None = None,
    over_order: OrderSpec = None,
    over_labels: LabelSpec = None,
    over_label_pos: str | Sequence[str] | None = None,
    ncol: int | None = None,
    jitter: float | None = None,
    plot_type: str | Mapping[str, str] | Sequence[str] | None = None,
    callback: Callable[..., None] | None = None,
    callback_args: tuple[Any, ...] = (),
    scatter_size: str | float | None = 'size',
    style: StyleSpec = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Plot selected variables in DataFrame, optionally disaggregating by groups.

    Parameters
    ----------
    df
        Input DataFrame.
    xvar
        Variable or index name storing x-values.
    yvar
        Column names storing y-values to be plotted.
    yvar_labels
        Variable labels.
    moment
        Name of moment to be plotted.
    by
        Variable or index name by which to disaggregate within individual
        plot panels.
    by_labels
        Pretty labels to be used in legend.
    by_order
        Values of categorical variable `by` which specify plotting order
        (useful to harmonize legend and plot order).
    over
        Variable or index name by which to disaggregate data into separate
        panels.
    over_order
        Values of categorical variable `over` which specify plotting order.
    over_labels
        Mapping of values of `over` variable to pretty labels.
    over_label_pos
        Position of annotation text containing the `over` value for current
        panel.
    ncol
        Number of columns used to arrange plot panels (ignored unless `over`
        is given).
    jitter
        Perturb x-location by given fraction of x-range (ignored unless `by`
        is given).
    plot_type
        Plot type ('bar', 'area', 'scatter', 'errorbar', etc.).
    callback
        If not None, will be called at the end of plotting code executed
        for each panel with arguments callback(ax, idx, df_panel, style, *callback_args).
    callback_args
        Tuple that will be expanded and passed as additional positional
        arguments to `callback()`.
    scatter_size
        If string, it is interpreted as a column name in `df`
        with values to be interpreted as marker sizes. If float,
        the value is used as a uniform marker size.
    style
        Plot style specification.
    **kwargs
        Keyword arguments passed to plot_grid().

    Returns
    -------
    Array of matplotlib axes objects.
    """
    jitter = float(jitter) if jitter is not None else None
    style = DefaultStyle() if style is None else style

    df = df.copy()

    df, by_var, by_labels_dict, by_order_arr = _process_slice(
        df, by, by_labels, by_order
    )
    df, over_var, over_labels_dict, over_order_arr = _process_slice(
        df, over, over_labels, over_order
    )

    df, yvars, moment_name = _process_dep_vars(df, yvar, moment)

    if xvar is None:
        if df.index.nlevels > 1:
            raise ValueError('Cannot determine x-variable')
        elif not df.index.name:
            xvar_ = '_xvalues'
            df.index.set_names([xvar_], inplace=True)
        else:
            xvar_ = str(df.index.name)
    else:
        xvar_ = str(xvar)
        if xvar_ in df.columns and xvar_ not in df.index:
            # Append xvar to index where it's expected by plotting functions.
            # Perform this indirectly so that MultiIndex columns work as well
            midx = df.index.to_frame(index=False)
            midx[xvar_] = df[xvar_].to_numpy()
            midx = pd.MultiIndex.from_frame(midx)
            df.index = midx
            del df[xvar_]

    varlist: list[str] = [over_var, by_var, xvar_]
    index_other = [str(name) for name in df.index.names if name not in varlist]

    # Reorder index levels (specific order is required below), push
    # user-given index levels that are not required to the end.
    df = df.reorder_levels(varlist + index_other)

    # --- Fix plot types, labels and styles for each variable ---

    # Plot types
    plot_type_dict: Mapping[str, str]
    if plot_type is None:
        plot_type_dict = dict.fromkeys(yvars, '')
    elif isinstance(plot_type, str):
        # Same plot type for all variables
        plot_type_dict = dict.fromkeys(yvars, plot_type)
    elif isinstance(plot_type, Mapping):
        # Expected data type
        plot_type_map: Any = plot_type
        plot_type_dict = {str(k): str(v) for k, v in plot_type_map.items()}
    elif isinstance(plot_type, Iterable):
        # Plot types passed as list, assumed in same order as variables
        plot_type_dict = dict(zip(yvars, plot_type, strict=False))
    else:
        raise ValueError('Unsupported plot_type value')

    # Process variable labels
    yvar_labels_dict: Mapping[str, str] | None = None
    if isinstance(yvar_labels, str):
        yvar_labels_dict = {yvars[0]: yvar_labels}
    elif isinstance(yvar_labels, Mapping):
        # Expected data type
        yvar_labels_map: Any = yvar_labels
        yvar_labels_dict = {str(k): str(v) for k, v in yvar_labels_map.items()}
    elif isinstance(yvar_labels, Iterable):
        # Convert from list to dict
        yvar_labels_dict = dict(zip(yvars, yvar_labels, strict=False))
    elif yvar_labels is None and not by_labels:
        yvar_labels_dict = {v: v for v in yvars}
    elif yvar_labels is None:
        pass
    else:
        raise ValueError('Unsupported yvar_labels value')

    # Replicate style for all variables, if needed
    styles: Mapping[str, AbstractStyle]
    if isinstance(style, AbstractStyle):
        # Will be propagated / converted to dict below
        style = [style]
    if isinstance(style, Mapping):
        # Expected data type
        styles_map: Any = style
        styles = {str(k): v for k, v in styles_map.items()}
    elif isinstance(style, Sequence):
        styles_list = list(style)
        if len(styles_list) == 1 and len(yvars) > 1:
            styles_list = styles_list * len(yvars)
        # Convert to dict
        styles = dict(zip(yvars, styles_list, strict=False))
    else:
        raise ValueError('Unsupported style vale')

    # Determine number of rows and columns from number of vars to be plotted.
    ncol = ncol if ncol else len(over_order_arr)
    nrow = math.ceil(len(over_order_arr) / ncol)
    npanels = len(over_order_arr)

    def fplot(ax: Axes, idx: tuple[int, int], **kwargs: Any) -> None:
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
        df_panel = df.xs(
            over_order_arr[ipanel], level=over_var, axis=0, drop_level=False
        )

        for _ivar, yvar in enumerate(yvars):
            # Variable-specific style
            style = styles[yvar]
            data: pd.DataFrame = df_panel[yvar]  # type: ignore

            mname = moment_name or _find_moment_name(data)

            df_moment = data[mname]

            barwidth = 1.0

            for k, by_value in enumerate(by_order_arr):
                mask = df_moment.index.isin([by_value], level=by_var)
                if not mask.any():
                    # For this panel there are no observations for the given layer
                    continue
                yvalues = df_moment[mask].to_numpy()
                xvalues = df_moment[mask].index.get_level_values(xvar_).to_numpy()

                # Legend labels. by-labels take precedence due to backwards
                # compatibility!
                leglbl = None
                if by_labels_dict and yvar_labels_dict:
                    bylbl = by_labels_dict.get(by_value, by_value)
                    vlbl = yvar_labels_dict.get(yvar, yvar)
                    leglbl = f'{vlbl}: {bylbl}'
                elif by_labels_dict:
                    leglbl = by_labels_dict.get(by_value, by_value)
                elif yvar_labels_dict:
                    leglbl = yvar_labels_dict.get(yvar, yvar)
                else:
                    # Fallback: use default string representation of -by- value
                    leglbl = f'{by_value}'

                if not np.any(np.isfinite(xvalues) & np.isfinite(yvalues)):
                    # Disable artificial legend labels when nothing is displayed
                    leglbl = None

                if plot_type_dict[yvar] == 'bar':
                    if xvalues.size > 1:
                        dx = np.amin(xvalues[1:] - xvalues[:-1]) * 0.8
                        barwidth = dx / len(by_order_arr)
                        if len(by_order_arr) % 2 == 0:
                            left = barwidth * (len(by_order_arr) - 1) / 2
                        else:
                            left = barwidth * (len(by_order_arr) // 2)

                        xvalues = xvalues - left + barwidth * k
                elif jitter:
                    dx = xvalues[-1] - xvalues[0]
                    if len(by_order_arr) % 2 == 0:
                        left = dx * jitter * (len(by_order_arr) + 1) / 2
                    else:
                        left = dx * jitter * (len(by_order_arr) // 2)

                    offset = dx * jitter * k
                    xvalues = xvalues - left + offset

                df_by: pd.DataFrame = data.xs(by_value, level=by_var)  # type: ignore
                yerr = _get_yerr(df_by, mname, yvalues)

                if plot_type_dict[yvar] == 'bar':
                    bw = barwidth * (1.0 - 2.0 * style.barmargin)

                    ax.bar(
                        xvalues,
                        yvalues,
                        width=bw,
                        yerr=yerr,
                        label=leglbl,
                        **style.bar_kwargs[k],
                    )

                elif plot_type_dict[yvar] == 'area' and yerr is not None:
                    ylb = yvalues - yerr[0]
                    yub = yvalues + yerr[1]
                    isfin = any(np.isfinite(ylb) & np.isfinite(yub))
                    if isfin:
                        ax.fill_between(
                            xvalues, ylb, yub, **style.fill_between_face_kwargs[k]
                        )

                        # Create lower and upper boundaries manually
                        kw = style.fill_between_edge_kwargs[k].copy()
                        kw['zorder'] += 10
                        ax.plot(xvalues, ylb, **kw)
                        ax.plot(xvalues, yub, **kw)

                    kw = style.plot_kwargs[k].copy()
                    kw['zorder'] += 20
                    ax.plot(xvalues, yvalues, label=leglbl, **kw)

                elif plot_type_dict[yvar] == 'scatter':
                    df_by: pd.DataFrame = df_panel.xs(by_value, level=by_var)  # type: ignore
                    size = _get_scatter_size(
                        scatter_size, yvar, df_by, style.markersize[k]
                    )

                    if style.split_scatter:
                        # Plot face component of scatter
                        ax.scatter(
                            xvalues, yvalues, s=size, **style.scatter_face_kwargs[k]
                        )

                        # Plot edge component of scatter
                        kw = style.scatter_edge_kwargs[k].copy()
                        kw['zorder'] += 1
                        ax.scatter(xvalues, yvalues, s=size, label=leglbl, **kw)
                    else:
                        # Default: plot edges and faces in single call
                        ax.scatter(
                            xvalues,
                            yvalues,
                            s=size,
                            label=leglbl,
                            **style.scatter_kwargs[k],
                        )

                else:
                    # Check whether style includes a marker
                    marker = style.errorbar_kwargs[k].get('marker')
                    has_marker = bool(marker)
                    if marker:
                        marker = marker.lower().strip()
                        has_marker = marker != '' and marker != 'none'

                    # Split between line/marker components only if marker is
                    # present
                    if style.split_scatter and has_marker:
                        ax.errorbar(
                            xvalues,
                            yvalues,
                            yerr=yerr,
                            label=leglbl,
                            **style.errorbar_no_marker_kwargs[k],
                        )

                        kw = style.marker_no_line_kwargs[k].copy()
                        if 'zorder' in kw:
                            kw['zorder'] += 1
                        else:
                            kw['zorder'] = 1
                        ax.plot(xvalues, yvalues, **kw)
                    else:
                        ax.errorbar(
                            xvalues,
                            yvalues,
                            yerr=yerr,
                            label=leglbl,
                            **style.errorbar_kwargs[k],
                        )

        # --- Label over group ---

        lbl = over_labels_dict.get(over_order_arr[ipanel], None)
        positions = anything_to_tuple(over_label_pos, force=True)
        labels = anything_to_tuple(lbl, force=True)
        if labels and positions:
            for lbl, pos in zip(labels, positions, strict=True):
                _style: AbstractStyle = styles[yvars[0]]
                if pos.lower() == 'title':
                    ax.set_title(lbl, **_style.title)
                else:
                    # Use title variant of text attributes
                    kw = _style.text_title.copy()
                    kw.update(_text_loc_to_kwargs(pos))
                    kw['s'] = lbl
                    kw['transform'] = ax.transAxes

                    ax.text(**kw)

        # --- Call any user-provided callback function ---

        if callable(callback):
            callback(ax, idx, df_panel, styles[yvars[0]], *callback_args)

    kwargs_default: dict[str, Any] = {'style': styles[yvars[0]]}

    kwargs_default.update(kwargs)
    kwargs = kwargs_default

    return plot_grid(fplot, nrow, ncol, **kwargs)
