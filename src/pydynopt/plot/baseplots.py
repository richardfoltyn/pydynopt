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
              outfile=None, style=None, *args, **kwargs):

    if column_title is None:
        column_title = np.ndarray((nrow, ), dtype=object)

    column_title = np.atleast_1d(column_title)

    if legend_at is not None:
        legend_at = np.array(legend_at, dtype=np.int)
        assert legend_at.shape[0] == 2

    if style is None:
        style = DefaultStyle()

    fig_kw = {'figsize': (style.cell_size * ncol, style.cell_size * nrow)}
    fig_kw.update(style.figure)

    if figure_kw is not None:
        style.figure.update(figure_kw)

    if subplot_kw is not None:
        style.subplot.update(subplot_kw)

    fig, axes = plt.subplots(nrow, ncol, subplot_kw=style.subplot,
                             sharex=sharex, sharey=sharey, squeeze=False,
                             **fig_kw)

    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                if j < column_title.shape[0] and column_title[j]:
                    axes[i, j].set_title(column_title[j], **style.title)

            if xlim is not None:
                axes[i, j].set_xlim(xlim)
            if ylim is not None:
                axes[i, j].set_ylim(ylim)

            if style.grid and ('b' not in style.grid or not style.grid['b']):
                axes[i, j].grid(**style.grid)

            fun(axes[i, j], (i, j), *args, **kwargs)

    if xlabel is not None:
        for j in range(ncol):
            axes[-1, j].set_xlabel(xlabel, **style.xlabel)

    if ylabel is not None:
        for i in range(nrow):
            axes[i, 0].set_ylabel(ylabel, **style.ylabel)

    if legend and legend_loc is not None and legend_at is not None:
        axes[legend_at[0], legend_at[1]].legend(loc=legend_loc, **style.legend)

    if suptitle is not None and suptitle:
        fig.suptitle(suptitle, **style.suptitle)

    render(fig, outfile)


def render(fig, outfile=None):
    if not outfile:
        fig.show()
    else:
        fig.savefig(outfile)
        fig.clear()