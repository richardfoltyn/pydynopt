__author__ = 'Richard Foltyn'

import numpy as np
import matplotlib.pyplot as plt

from .styles import DefaultStyle


def plot_grid(fun, nrow=1, ncol=1,
              column_title=None, suptitle=None,
              figure_kw=None, subplot_kw=None,
              sharex=True, sharey=True,
              xlabel=None, ylabel=None, xlim=None, ylim=None,
              xticks=None, yticks=None, xticklabels=None, yticklabels=None,
              legend_at=(0, 0), legend_loc='best', legend=False,
              outfile=None, style=None, aspect=None, close_fig=True,
              pass_style=False, metadata=None, *args, **kwargs):
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
    xticks : array_like, optional
        Location of (major) x-ticks. Ignored if subplots don't have shared
        x-values.
    xticklabels : array_like, optional
        Ticklabels for x-ticks. Ignored if x-ticks not given or not used.
    yticks : array_like, optional
        Location of (major) y-ticks. Ignored if subplots don't have shared
        y-values.
    yticklabels : array_like, optional
        Ticklabels for y-ticks. Ignored if y-ticks not given or not used.
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
    aspect : float, optional
        Aspect ratio
    close_fig : bool
        If true (default), close the figure after plotting if an output
        file is specified. This can be disabled if the figure should
        be shown on screen after being saved in a file.
    pass_style : bool, optional
        If true and style is not None, add style to kwargs when calling
        plotting function.
    metadata : dict, optional
        Dictionary of metadata passed to savefig(). Admissible values depend
        on backend used to generate the figure.
    args :
        Positional arguments passed directly to `fun`
    kwargs :
        Keyword arguments passed directly to `fun`
    """

    if column_title is None:
        column_title = np.ndarray((nrow, ), dtype=object)

    column_title = np.atleast_1d(column_title)

    if legend_at is not None:
        if isinstance(legend_at, str):
            if legend_at.lower() != 'figure':
                msg = f'Invalid string value for legend_at: {legend_at}'
                raise ValueError(msg)
        else:
            legend_at = np.array(legend_at, dtype=np.int)
            assert 1 <= legend_at.ndim <= 2
            assert legend_at.shape[-1] == 2
            legend_at = legend_at.reshape((-1, 2))

    if style is None:
        style = DefaultStyle()

    if ylim is not None:
        ylim = broadcast_ylim(nrow, ncol, ylim)

    # Obtain aspect ratio: first try whatever is stored in 'aspect' attribute
    # of style object, then override this with the 'aspect' argument
    # if it's not None.
    aspect_default = 1.0
    if style is not None:
        aspect_default = getattr(style, 'aspect', 1.0)
    if aspect is not None:
        aspect_default = aspect

    aspect = aspect_default

    if pass_style and style is not None:
        kwargs = kwargs.copy()
        kwargs['style'] = style

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

    margins = style.margins
    if margins is not None:
        margins1d, *rest = np.broadcast_arrays(margins, np.arange(4))
    else:
        margins1d = np.zeros(4)

    for i in range(nrow):
        for j in range(ncol):

            ax = axes[i, j]

            if i == 0:
                if j < column_title.shape[0] and column_title[j]:
                    ax.set_title(column_title[j], **style.title)

            if xlim is not None:
                dx = xlim[1] - xlim[0]
                xlb = xlim[0] - margins1d[0] / aspect * dx
                xub = xlim[1] + margins1d[2] / aspect * dx
                ax.set_xlim((xlb, xub))

            if ylim is not None:
                dy = ylim[i, j, 1] - ylim[i, j, 0]
                ylb = ylim[i, j, 0] - margins1d[1] * dy
                yub = ylim[i, j, 1] + margins1d[3] * dy
                ax.set_ylim((ylb, yub))

            # No limits specified, margin is a float applicable to all sides
            if ylim is None and xlim is None and isinstance(margins, float):
                ax.margins(margins)

            if style.grid:
                ax.grid(**style.grid)

            if xticks is not None:
                ax.set_xticks(xticks)
                if (i == (nrow - 1) or not sharex) and xticklabels is not None:
                    ax.set_xticklabels(xticklabels, **style.xticklabels)

            if yticks is not None:
                ax.set_yticks(yticks)
                if (j == (ncol - 1) or not sharey) and yticklabels is not None:
                    ax.set_yticklabels(yticklabels, **style.yticklabels)

            fun(ax, (i, j), *args, **kwargs)

            # Apply tick label styles after calling the function since
            # user actions might have unset style settings.
            for lbl in ax.get_xticklabels():
                _set_properties(lbl, **style.xticklabels)

            for lbl in ax.get_yticklabels():
                _set_properties(lbl, **style.yticklabels)

    if legend:
        if isinstance(legend_at, str) and legend_at.lower() == 'figure':
            # Legend should be placed relative to whole figure. This will only
            # work if constrained_layout is NOT used, needs to be turned off
            # in figure kwargs in style!
            leg = fig.legend(loc=legend_loc, **style.legend)
        elif legend_loc is not None and legend_at is not None:
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

    render(fig, outfile, close_fig, metadata)


def render(fig, outfile=None, close_fig=True, metadata=None):
    if not outfile:
        fig.show()
    else:
        fig.savefig(outfile, metadata=metadata)
        if close_fig:
            plt.close(fig)


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


def _set_properties(obj, **kwargs):
    """
    Apply given properties specified as keyword arguments to given object
    using set_XXX() methods, if present.

    Parameters
    ----------
    obj
    kwargs
    """

    for key, value in kwargs.items():
        if hasattr(type(obj), f'set_{key}'):
            method = getattr(type(obj), f'set_{key}')
            method(obj, value)
        else:
            try:
                setattr(obj, key, value)
            except:
                pass
