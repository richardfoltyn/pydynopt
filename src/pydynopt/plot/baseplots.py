from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

import numpy as np
import matplotlib.pyplot as plt

from .styles import DefaultStyle


def plot_grid(fun, nrow=1, ncol=1,
              column_title=None, suptitle=None,
              figure_kw=None, subplot_kw=None,
              sharex=True, sharey=True,
              xlabel=None, ylabel=None, xlim=None, ylim=None,
              legend_at=(0, 0), legend_loc='upper left', legend=False,
              outfile=None, style=None, aspect=1.0, *args, **kwargs):
    """
    Creates a rectangular grid of subplots and calls a user-provided function
    for each subplot to render user-supplied content.

    Parameters
    ----------
    fun : callable
        Callback function that is called for each subplot with arguments
            fun(ax, idx, *args, **kwargs)
        where `ax` is the MPL Axes class, `idx` is a tuple (row, col)
        identifying the current subplot, and `args` and `kwargs` are
        the corresponding arguments to `plot_grid` passed directly to the
        callback function.
    nrow : int
        Number of rows
    ncol : int
        Number of columns
    column_title : str or array_like
        List of column titles
    suptitle : str
        Subtitle, currently not properly implemented
    figure_kw : dict
        Dictionary of keyword arguments passed to MPL's subplots() function
        via **kwargs.
    subplot_kw : dict
        Dictionary passed to MPL's subplots() as the `subplot_kw` argument
    sharex : bool
        If true, identical x-limits are enforced across all subplots
    sharey : bool
        If true, identical y-limits are enforced across all subplots
    xlabel : str
        x-axis label
    ylabel : str or array_like or None
        y-axis label
    xlim : iterable
        Lower and upper x-axis limits
    ylim : array_like
        Lower and upper y-axis limits. Can be specified either as a tuple
        if limits are to be applied across all rows / columns, or as an
        array of shape [nrow, ncol, 2] with panel-specific limits.
    legend_at : array_like
        Subplot in which legend should be placed (default: (0,0)). Accepts
        either a single tuple if legend should be placed in only one subplot,
        or a list of tuples for multiple legends.
    legend_loc : str
        MPL-compatible string identifying where the legend should be placed
        within a subplot
    legend : bool
        If true, legend is displayed in the subplot identified by `legend_at`
    outfile : str or None
        If not None, figure is saved into given file
    style : styles.AbstractStyle
        Instance of AbstractStyle controlling various rendering options.
    aspect : float
        Aspect ratio
    args : tuple
        Positional arguments passed directly to `fun`
    kwargs : dict
        Keyword arguments passed directly to `fun`
    """

    if column_title is None:
        column_title = np.ndarray((nrow, ), dtype=object)

    column_title = np.atleast_1d(column_title)

    if legend_at is not None:
        legend_at = np.array(legend_at, dtype=np.int)
        assert 1 <= legend_at.ndim <= 2
        assert legend_at.shape[-1] == 2
        legend_at = legend_at.reshape((-1, 2))

    if style is None:
        style = DefaultStyle()

    if ylim is not None:
        ylim = broadcast_ylim(nrow, ncol, ylim)

    # Aspect ratio is defined as width / height
    fig_kw = {'figsize': (style.cell_size * ncol,
                          style.cell_size * nrow / aspect)}
    fig_kw.update(style.figure)

    if figure_kw is not None:
        style.figure.update(figure_kw)

    if subplot_kw is not None:
        style.subplot.update(subplot_kw)

    fig, axes = plt.subplots(nrow, ncol, subplot_kw=style.subplot,
                             sharex=sharex, sharey=sharey, squeeze=False,
                             **fig_kw)

    if xlabel is not None:
        xlabel = np.atleast_1d(xlabel)
        if len(xlabel) != ncol:
            if len(xlabel) != 1:
                raise ValueError('Non-conformable number of xlabels passed')
            xlabel = np.repeat(xlabel, ncol)

    if ylabel is not None:
        ylabel = np.atleast_1d(ylabel)
        if len(ylabel) != nrow:
            if len(ylabel) != 1:
                raise ValueError('Non-conformable number of ylabels passed')
            ylabel = np.repeat(ylabel, nrow)

    if xlabel is not None:
        for j in range(ncol):
            axes[-1, j].set_xlabel(xlabel[j], **style.xlabel)

    if ylabel is not None:
        for i in range(nrow):
            axes[i, 0].set_ylabel(ylabel[i], **style.ylabel)

    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                if j < column_title.shape[0] and column_title[j]:
                    axes[i, j].set_title(column_title[j], **style.title)

            if xlim is not None:
                axes[i, j].set_xlim(xlim)
            if ylim is not None:
                axes[i, j].set_ylim(ylim[i, j])

            if style.grid and ('b' not in style.grid or not style.grid['b']):
                axes[i, j].grid(**style.grid)

            fun(axes[i, j], (i, j), *args, **kwargs)

    if legend and legend_loc is not None and legend_at is not None:
        for i, idx in enumerate(legend_at):
            axes[idx[0], idx[1]].legend(loc=legend_loc, **style.legend)

    if suptitle is not None and suptitle:
        fig.suptitle(suptitle, **style.suptitle)

    # === y-ticks for shared ylims ===
    # Turn off ytick labels if ylim are the same for entire row
    # for all but the first column
    if not sharey:
        for i in range(nrow):
            # Determine whether ylim in this row are identical for all columns
            ylim_same = False
            if ylim is not None:
                ylim_same = all(np.all(ylim[i] == ylim[i, 0:1], axis=0))

            for j in range(ncol):
                if j > 0 and ylim_same and not sharey:
                    axes[i, j].set_yticklabels([])

    render(fig, outfile)


def render(fig, outfile=None):
    if not outfile:
        fig.show()
    else:
        fig.savefig(outfile)
        plt.close()


def broadcast_ylim(nrow, ncol, ylim):
    """
    Broadcast ylim across rows / columns as needed.

    Parameters
    ----------
    nrow : int
    ncol : int
    ylim : array_like
        ylim values as passed into plot_grid() by user code.

    Returns
    -------
    ylim : np.ndarray
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
