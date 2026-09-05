"""
Helpers for creating and styling rectangular Matplotlib subplot grids.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import collections.abc
from collections.abc import Mapping, Sequence
import copy
from pathlib import Path
from typing import Any, Literal, Protocol

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter, Locator
import numpy as np

from ..utils import anything_to_dict
from .styles import AbstractStyle, DefaultStyle

__all__ = ['hide_subplot', 'plot_grid']


type ShareMode = bool | Literal['all', 'none', 'row', 'col']
type LegendAtArg = str | tuple[int, int] | Sequence[tuple[int, int]] | np.ndarray | None
type BboxAnchor = tuple[float, float] | tuple[float, float, float, float]
type SharedAxisGroup = tuple[str, int] | tuple[str, int, int]
type GuideLinesArg = (
    Mapping[float, Mapping[str, Any] | None]
    | Sequence[float]
    | np.ndarray
    | float
    | None
)


def _shared_axis_group(mode: ShareMode, i: int, j: int) -> SharedAxisGroup:
    """Return an identifier for the axis-sharing group of subplot (i, j)."""
    if mode is True or mode == 'all':
        return 'all', 0
    if mode == 'row':
        return 'row', i
    if mode == 'col':
        return 'col', j
    return 'panel', i, j


def _handle_titles(
    column_title: Sequence[str] | str | None,
    title: str | Sequence[str] | np.ndarray | None,
    nrow: int,
    ncol: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Handle and broadcast subplot titles and column titles.

    Parameters
    ----------
    column_title
        List of column titles. Will be ignored if `title` is given.
    title
        Titles for each subplot. Will be broadcast to match the dimensions
        of the subplot grid.
    nrow
        Number of rows.
    ncol
        Number of columns.

    Returns
    -------
    column_title_
        Processed and broadcasted column titles.
    title_
        Processed and broadcasted subplot titles.
    """
    if column_title is None:
        column_title_ = np.zeros((ncol,), dtype=object)
    else:
        column_title_ = np.atleast_1d(column_title)

    if title is None:
        title_ = np.zeros((nrow, ncol), dtype=object)
    else:
        # Disable column titles, axes titles take precedence
        column_title_ = np.zeros(ncol, dtype=object)
        title_tmp = np.atleast_2d(title)
        title_ = np.broadcast_to(title_tmp, (nrow, ncol))

    return column_title_, title_


def _handle_legend_at(legend_at: LegendAtArg) -> str | np.ndarray | None:
    """
    Validate and format the legend_at specification.

    Parameters
    ----------
    legend_at
        Location of the subplot(s) or figure where the legend should be placed.

    Returns
    -------
    Processed legend position (either string representation or 2D array of coordinates).
    """
    if legend_at is None:
        return None

    if isinstance(legend_at, str):
        if legend_at.lower() != 'figure':
            msg = f'Invalid string value for legend_at: {legend_at}'
            raise ValueError(msg)
        return legend_at

    legend_at_arr = np.array(legend_at, dtype=int)
    if not (1 <= legend_at_arr.ndim <= 2):
        raise ValueError('legend_at must be 1D or 2D.')
    if legend_at_arr.shape[-1] != 2:
        raise ValueError('legend_at entries must be (row, col) pairs.')
    return legend_at_arr.reshape((-1, 2))


def _handle_axis_label(
    label: str | Sequence[str] | None,
    expected_size: int,
    label_name: str,
) -> np.ndarray | None:
    """
    Validate, broadcast and format xlabel or ylabel sequences.

    Parameters
    ----------
    label
        Input labels to process.
    expected_size
        The expected length of the broadcasted labels (ncol for xlabel, nrow for ylabel).
    label_name
        The type of label ('xlabel' or 'ylabel') used for ValueError messages.

    Returns
    -------
    An array of labels conformable to expected_size, or None.
    """
    if label is None:
        return None

    label_arr = np.atleast_1d(label)
    if len(label_arr) != expected_size:
        if len(label_arr) != 1:
            raise ValueError(f'Non-conformable number of {label_name}s passed')
        label_arr = np.repeat(label_arr, expected_size)
    return label_arr


class PlotGridFunc(Protocol):
    """Callback protocol for plot_grid()."""

    def __call__(self, ax: Axes, idx: tuple[int, int], **kwargs: Any) -> Any: ...


def plot_grid(
    fun: PlotGridFunc,
    nrow: int = 1,
    ncol: int = 1,
    *,
    column_title: Sequence[str] | str | None = None,
    title: str | Sequence[str] | np.ndarray | None = None,
    suptitle: str | None = None,
    figure_kw: Mapping[str, Any] | None = None,
    subplot_kw: Mapping[str, Any] | None = None,
    sharex: ShareMode = True,
    sharey: ShareMode = True,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: Sequence[float] | np.ndarray | None = None,
    xticks: Sequence[float] | np.ndarray | Locator | None = None,
    yticks: Sequence[float] | np.ndarray | Locator | None = None,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    xtickformatter: Formatter | None = None,
    ytickformatter: Formatter | None = None,
    legend_at: LegendAtArg = (0, 0),
    legend_loc: str | tuple[float, float] | None = None,
    legend: bool = False,
    legend_title: str | None = None,
    bbox_to_anchor: BboxAnchor | None = None,
    outfile: Path | str | None = None,
    style: AbstractStyle | None = None,
    aspect: float | None = None,
    close_fig: bool = True,
    pass_style: bool = False,
    metadata: Mapping[str, Any] | None = None,
    identity: bool | Mapping[str, Any] | None = None,
    hline: GuideLinesArg = None,
    vline: GuideLinesArg = None,
    show: bool = True,
    colorbar: bool = False,
    colorbar_at: tuple[int, int] | None = None,
    colorbar_kw: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Create a rectangular grid of subplots and call a user-provided function to render user-supplied content.

    Parameters
    ----------
    fun
        Callback function that is called for each subplot with arguments
            fun(ax, idx, **kwargs)
        where `ax` is the MPL Axes class, `idx` is a tuple (row, col)
        identifying the current subplot, and `kwargs` are keyword arguments
        passed directly to the callback function.
    nrow
        Number of rows.
    ncol
        Number of columns.
    column_title
        List of column titles. Will be ignored if `title` is given.
    title
        Titles for each subplot. Will be broadcast to match
        the dimensions of the subplot grid.
    suptitle
        Figure-level title.
    figure_kw
        Dictionary of keyword arguments passed to MPL's subplots() function.
    subplot_kw
        Dictionary passed to MPL's subplots() as the `subplot_kw` argument.
    sharex
        Controls sharing of properties among x axes. Valid values are
        True (or 'all'), False (or 'none'), 'row' and 'col'.
    sharey
        Controls sharing of properties among y axes. Valid values are
        True (or 'all'), False (or 'none'), 'row' and 'col'.
    xlabel
        x-axis label(s). If a sequence is passed, it must have length `ncol`
        (or length 1, in which case the value is repeated).
    ylabel
        y-axis label(s). If a sequence is passed, it must have length `nrow`
        (or length 1, in which case the value is repeated).
    xlim
        Lower and upper x-axis limits.
    ylim
        Lower and upper y-axis limits. Can be specified either as a tuple
        if limits are to be applied across all rows / columns, or as an
        array of shape [nrow, ncol, 2] with panel-specific limits.
    xticks
        Location of major x-ticks.
    xticklabels
        Ticklabels for x-ticks. Ignored if x-ticks not given or not used.
    yticks
        Location of major y-ticks.
    yticklabels
        Ticklabels for y-ticks. Ignored if y-ticks not given or not used.
    xtickformatter
        Formatter for x-ticks.
    ytickformatter
        Formatter for y-ticks.
    legend_at
        Subplot in which legend should be placed (default: (0,0)). Accepts
        either a single tuple if legend should be placed in only one subplot,
        a list of tuples for multiple legends, or 'figure' for a figure-level
        legend.
    legend_loc
        MPL-compatible string identifying where the legend should be placed
        within a subplot.
    legend
        If true, legend is displayed in the subplot identified by `legend_at`.
    legend_title
        Title used for the legend.
    bbox_to_anchor
        Passed to legend() call.
    outfile
        If not None, figure is saved into given file.
    style
        Instance of AbstractStyle controlling various rendering options.
    aspect
        Aspect ratio used to construct figure.
    close_fig
        If true (default), close the figure after plotting if an output
        file is specified. This can be disabled if the figure should
        be shown on screen after being saved in a file.
    pass_style
        If true and style is not None, add style to kwargs when calling
        plotting function.
    metadata
        Dictionary of metadata passed to savefig(). Admissible values depend
        on backend used to generate the figure.
    identity
        Plot identity line. If passed as mapping, key/value pairs
        are passed as kwargs to ax.axline() to control plot style.
    hline
        Horizontal guide lines. Can be a sequence of y-values, or a mapping
        ``y -> style_overrides`` where each value overrides style.guideline.
    vline
        Vertical guide lines. Can be a sequence of x-values, or a mapping
        ``x -> style_overrides`` where each value overrides style.guideline.
    show
        If true and `outfile` is None, display figure.
    colorbar
        If True, a colorbar is displayed for the scalar mappable in the subplot.
    colorbar_at
        Subplot in which colorbar should be placed.
    colorbar_kw
        Keyword arguments passed to ``fig.colorbar()``.
    kwargs
        Keyword arguments passed directly to `fun`.

    Returns
    -------
    axes
        Array of MPL Axes objects with shape (`nrow`, `ncol`).
    """
    hline_ = anything_to_dict(hline, force=True)
    vline_ = anything_to_dict(vline, force=True)

    column_title_, title_ = _handle_titles(column_title, title, nrow, ncol)
    legend_at_ = _handle_legend_at(legend_at)

    if style is None:
        style = DefaultStyle()

    ylim_ = broadcast_ylim(nrow, ncol, ylim) if ylim is not None else None

    # Obtain aspect ratio: first try whatever is stored in 'aspect' attribute
    # of style object, then override this with the 'aspect' argument if it's not None.
    aspect_: float = aspect or getattr(style, 'aspect', 1.0)

    ax_aspect = getattr(style, 'ax_aspect', None)
    ax_aspect_default = ax_aspect or aspect_

    if pass_style:
        kwargs = kwargs.copy()
        kwargs['style'] = style

    # Aspect ratio is defined as width / height
    fig_kw: dict[str, Any] = {
        'figsize': (style.cell_size * ncol, style.cell_size * nrow / aspect_)
    }
    fig_kw.update(style.figure)

    if figure_kw is not None:
        fig_kw.update(figure_kw)

    subplot_kw_ = style.subplot.copy()
    if subplot_kw is not None:
        subplot_kw_.update(subplot_kw)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        subplot_kw=subplot_kw_,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        **fig_kw,
    )

    xlabel_ = _handle_axis_label(xlabel, ncol, 'xlabel')
    ylabel_ = _handle_axis_label(ylabel, nrow, 'ylabel')

    if xticks is not None and not isinstance(xticks, Locator):
        xticks_ = np.atleast_1d(xticks)
    else:
        xticks_ = xticks

    if yticks is not None and not isinstance(yticks, Locator):
        yticks_ = np.atleast_1d(yticks)
    else:
        yticks_ = yticks

    margins = style.margins
    if margins is not None:
        margins1d, *_ = np.broadcast_arrays(margins, np.arange(4))
    else:
        margins1d = np.zeros(4)

    # Margins are interpreted as fractions of axis range.
    has_x_margins = not np.allclose(margins1d[[0, 2]], 0.0)
    has_y_margins = not np.allclose(margins1d[[1, 3]], 0.0)

    # determine whether subplots have the same x-axes
    if isinstance(sharex, str):
        has_sharex = (sharex == 'col') or (sharex == 'all')
    else:
        has_sharex = bool(sharex)

    # determine whether subplots have the same y-axes
    if isinstance(sharey, str):
        has_sharey = (sharey == 'row') or (sharey == 'all')
    else:
        has_sharey = bool(sharey)

    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i, j]

            if i == 0 and j < column_title_.shape[0] and column_title_[j]:
                ax.set_title(column_title_[j], **style.title)

            if ttl := title_[i, j]:
                ax.set_title(ttl, **style.title)

            if style.grid:
                ax.grid(**style.grid)

            if xtickformatter is not None:
                ax.xaxis.set_major_formatter(xtickformatter)

            if isinstance(xticks_, Locator):
                # Create a copy of the locator instance as assigning the same locator
                # to multiple Axes seems to harmonize their ticks, which is not what
                # we want if sharex=False.
                ax.xaxis.set_major_locator(copy.deepcopy(xticks_))
            elif xticks_ is not None:
                if (i == (nrow - 1) or not has_sharex) and xticklabels is not None:
                    ax.set_xticks(xticks_, xticklabels, **style.xticklabels)
                else:
                    ax.set_xticks(xticks_)

            if ytickformatter is not None:
                ax.yaxis.set_major_formatter(ytickformatter)

            if isinstance(yticks_, Locator):
                # Create a copy of the locator instance as assigning the same locator
                # to multiple Axes seems to harmonize their ticks, which is not what
                # we want if sharey=False.
                ax.yaxis.set_major_locator(copy.deepcopy(yticks_))
            elif yticks_ is not None:
                if (j == 0 or not has_sharey) and yticklabels is not None:
                    ax.set_yticks(yticks_, yticklabels, **style.yticklabels)
                else:
                    ax.set_yticks(yticks_)

            if getattr(style, 'rotate_yticklabels', False):
                ax.tick_params(axis='y', labelrotation=90)

            # Apply tick_params for both axes. These setting are more robust than
            # arguments passed to set_xticklabels() which may be discarded when the
            # graph is modified in various ways.
            ax.xaxis.set_tick_params(which='major', **style.xtick_params)
            ax.yaxis.set_tick_params(which='major', **style.ytick_params)

            if xlabel_ is not None and (i == (nrow - 1) or not has_sharex):
                ax.set_xlabel(xlabel_[j], **style.xlabel)

            if ylabel_ is not None and (j == 0 or not has_sharey):
                ax.set_ylabel(ylabel_[i], **style.ylabel)

            fun(ax, (i, j), **kwargs)

            # Apply axis aspect
            if style.ax_box_aspect is not None:
                # Prioritize ax_box_aspect over ax_aspect, if both are set.
                ax.set_box_aspect(style.ax_box_aspect)
            elif ax_aspect is not None:
                ax.set_aspect(ax_aspect)

            # Plot identity line
            if identity is not None:
                # If frame / axes are turned off, skip identity
                ax_on = ax.xaxis.get_visible() and ax.yaxis.get_visible()
                frame_on = ax.get_frame_on()
                if ax_on or frame_on:
                    kw = {'lw': 0.5, 'alpha': 0.8, 'zorder': -1, 'color': 'black'}
                    # Update keyword arguments, if applicable
                    if isinstance(identity, collections.abc.Mapping):
                        kw.update(identity)
                    ax.axline((0, 0), slope=1, **kw)

            # Plot horizontal guide lines
            for ycoord, stl in hline_.items():
                kw = style.guideline.copy()
                if stl:
                    kw.update(stl)
                ax.axhline(ycoord, **kw)

            # Plot vertical guide lines
            for xcoord, stl in vline_.items():
                kw = style.guideline.copy()
                if stl:
                    kw.update(stl)
                ax.axvline(xcoord, **kw)

    # === Apply margins and limits ===

    # Apply explicit limits (if provided) and margin offsets. If limits are not
    # provided, use axis limits after all panels have been plotted.
    xlim_base: dict[tuple[str, int] | tuple[str, int, int], tuple[float, float]] = {}
    ylim_base: dict[tuple[str, int] | tuple[str, int, int], tuple[float, float]] = {}

    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i, j]

            if xlim is not None:
                xlim0 = xlim
            elif has_x_margins:
                key = _shared_axis_group(sharex, i, j)
                if key not in xlim_base:
                    xlim_base[key] = ax.get_xlim()
                xlim0 = xlim_base[key]
            else:
                xlim0 = None

            if xlim0 is not None:
                dx = xlim0[1] - xlim0[0]
                xlb = xlim0[0] - margins1d[0] / ax_aspect_default * dx
                xub = xlim0[1] + margins1d[2] / ax_aspect_default * dx
                ax.set_xlim((xlb, xub))

            if ylim_ is not None:
                ylim0 = ylim_[i, j]
            elif has_y_margins:
                key = _shared_axis_group(sharey, i, j)
                if key not in ylim_base:
                    ylim_base[key] = ax.get_ylim()
                ylim0 = ylim_base[key]
            else:
                ylim0 = None

            if ylim0 is not None:
                dy = ylim0[1] - ylim0[0]
                ylb = ylim0[0] - margins1d[1] * dy
                yub = ylim0[1] + margins1d[3] * dy
                ax.set_ylim((ylb, yub))

    # === Legend ===

    if legend:
        # Merge keywords that might be present in style with potential
        # overrides passed as arguments.
        kw = style.legend.copy()
        if bbox_to_anchor is not None:
            kw['bbox_to_anchor'] = bbox_to_anchor
        if legend_loc:
            kw['loc'] = legend_loc
        kw['title'] = legend_title

        if isinstance(legend_at_, str) and legend_at_.lower() == 'figure':
            # Legend should be placed relative to whole figure. This will only
            # work if constrained_layout is NOT used, needs to be turned off
            # in figure kwargs in style!
            fig.legend(**kw)
        elif isinstance(legend_at_, np.ndarray):
            for idx in legend_at_:
                row, col = int(idx[0]), int(idx[1])
                ax_to_legend = axes[row, col]
                assert isinstance(ax_to_legend, Axes)
                ax_to_legend.legend(**kw)

    # === Colorbar ===

    if colorbar:
        cb_kw: dict[str, Any] = {}
        if style:
            style_cb = getattr(style, 'colorbar', None)
            if isinstance(style_cb, Mapping):
                cb_kw.update(style_cb)
        if colorbar_kw is not None:
            cb_kw.update(colorbar_kw)

        if colorbar_at is None:
            cb_row, cb_col = 0, ncol - 1
        else:
            cb_row, cb_col = colorbar_at

        cb_row = cb_row % nrow
        cb_col = cb_col % ncol
        ax_cb = axes[cb_row, cb_col]

        mappable = None
        if ax_cb.collections:
            for coll in reversed(ax_cb.collections):
                if coll.get_array() is not None:
                    mappable = coll
                    break
        if mappable is None and ax_cb.images:
            for img in reversed(ax_cb.images):
                if img.get_array() is not None:
                    mappable = img
                    break

        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=ax_cb, **cb_kw)

            # Apply styling to colorbar tick labels
            cbar_ticklabels = getattr(style, 'cbar_ticklabels', {})
            cbar_tick_params = {}
            for k, v in cbar_ticklabels.items():
                if k == 'fontfamily':
                    cbar_tick_params['labelfontfamily'] = v
                elif k == 'fontsize':
                    cbar_tick_params['labelsize'] = v
                elif k == 'color':
                    cbar_tick_params['labelcolor'] = v
                elif k == 'rotation':
                    cbar_tick_params['labelrotation'] = v
                else:
                    cbar_tick_params[k] = v
            cbar.ax.tick_params(**cbar_tick_params)

    if suptitle:
        fig.suptitle(suptitle, **style.suptitle)

    # === y-ticks for shared ylims ===

    # Turn off ytick labels if ylim are the same for entire row
    # for all but the first column
    if not has_sharey:
        for i in range(nrow):
            # Determine whether ylim in this row are identical for all columns
            ylim_same = False
            if ylim_ is not None:
                ylim_same = all(np.all(ylim_[i] == ylim_[i, 0:1], axis=0))

            yticks_same = True
            yticks0 = axes[i, 0].get_yticks()
            for j in range(1, ncol):
                yticks_j = axes[i, j].get_yticks()
                if (
                    len(yticks0) != len(yticks_j)
                    or np.amax(np.abs(yticks0 - yticks_j)) > 1.0e-8
                ):
                    yticks_same = False
                    break

            if ylim_same and yticks_same:
                for j in range(1, ncol):
                    axes[i, j].set_yticklabels([])

    if outfile:
        kw: dict[str, Any] = {}
        if Path(outfile).suffix == '.pdf':
            kw['metadata'] = metadata
        fig.savefig(outfile, **kw)
        if close_fig:
            plt.close(fig)
    elif show:
        fig.show()

    return axes


def broadcast_ylim(
    nrow: int,
    ncol: int,
    ylim: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Broadcast ylim across rows / columns as needed.

    Parameters
    ----------
    nrow
    ncol
    ylim
        ylim values as passed into plot_grid() by user code.

    Returns
    -------
    ylims broadcast across rows / columns. Return array has shape
    [nrow, ncol, 2]
    """
    # Tile ylim as needed to obtain array dimension (nrow, ncol, 2)
    ylim = np.atleast_1d(ylim)
    if not (1 <= ylim.ndim <= 3):
        raise ValueError('ylim dimension must be between 1 and 3!')

    if ylim.ndim == 1:
        ylim = ylim[np.newaxis, np.newaxis]
    elif ylim.ndim == 2:
        # Insert column dimension, assume that ylims are identical within each
        # row
        ylim = ylim[:, np.newaxis]

    if ylim.shape[0] not in [1, nrow]:
        raise ValueError('Non-conformable argument ylim!')
    if ylim.shape[1] not in [1, ncol]:
        raise ValueError('Non-conformable argument ylim!')

    if ylim.shape[0] != nrow:
        ylim = np.tile(ylim, reps=(nrow, 1, 1))
    if ylim.shape[1] != ncol:
        ylim = np.tile(ylim, reps=(1, ncol, 1))

    return ylim


def hide_subplot(ax: Axes) -> None:
    """
    Set various parameters to hide the frame, axes, ticks, etc. of a subplot.

    This can be used to hide "residual" subplots that are not needed when plotting a
    rectangular grid of subplots.

    Parameters
    ----------
    ax
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(bottom=False, left=False)
    ax.set_frame_on(False)
    ax.grid(None)
