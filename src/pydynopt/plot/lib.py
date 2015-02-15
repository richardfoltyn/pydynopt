from __future__ import print_function, division, absolute_import

__author__ = 'Richard Foltyn'

import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools as it

from .style import LBL_KWARGS, LEG_KWARGS, TITLE_KWARGS, SUBPLOT_KWARGS, \
    GRID_KWARGS, CELL_SIZE, SUPTITLE_KWARGS, DEFAULT_LINEWIDTH, TEXT_KWARGS

from .style import default_lstyles, default_colors, default_alphas


def plot_grid(fun, nrow=1, ncol=1,
              column_title=None, suptitle=None,
              plot_kw=None, subplot_kw=None,
              xlabel=None, ylabel=None, xlim=None, ylim=None,
              leg_at=(0, 0), leg_loc='upper left', legend=False,
              grid=True, tight_layout=True,
              outfile=None, *args, **kwargs):

    if column_title is None:
        column_title = np.ndarray((0,), dtype=object)

    column_title = np.atleast_1d(column_title)

    if leg_at is not None:
        leg_at = np.array(leg_at, dtype=np.int)
        assert leg_at.shape[0] == 2

    _plot_kwargs = plot_kw
    plot_kw = {'figsize': (CELL_SIZE * ncol, CELL_SIZE * nrow),
               'sharex': True, 'sharey': True}
    if _plot_kwargs is not None:
        plot_kw.update(_plot_kwargs)

    _subplot_kw = subplot_kw
    subplot_kw = SUBPLOT_KWARGS
    if _subplot_kw is not None:
        subplot_kw.update(_subplot_kw)

    fig, axes = plt.subplots(nrow, ncol, subplot_kw=subplot_kw, **plot_kw)

    axes = np.atleast_2d(axes)

    for i in range(nrow):
        for j in range(ncol):
            if j == 0:
                if i < column_title.shape[0] and column_title[i, j]:
                    axes[i, j].set_title(column_title[i, j], **TITLE_KWARGS)

            if xlim is not None:
                axes[i, j].set_xlim(xlim)
            if ylim is not None:
                axes[i, j].set_ylim(ylim)

            if grid:
                axes[i, j].grid(**GRID_KWARGS)

            fun(axes[i, j], (i, j), *args, **kwargs)

    if xlabel is not None:
        for j in range(ncol):
            axes[-1, j].set_xlabel(xlabel, **LBL_KWARGS)

    if ylabel is not None:
        for i in range(nrow):
            axes[i, 0].set_ylabel(ylabel, **LBL_KWARGS)

    if legend and leg_loc is not None and leg_at is not None:
        axes[leg_at[0], leg_at[1]].legend(loc=leg_loc, **LEG_KWARGS)

    if suptitle is not None and suptitle:
        fig.suptitle(suptitle, **SUPTITLE_KWARGS)

    if tight_layout:
        plt.tight_layout()
    render(fig, outfile)


def plot_simple(x, y=None, plot='scatter', title=None, plot_kw=None,
                subplot_kw=None,
                xlabel=None, ylabel=None, xlim=None, ylim=None,
                legend=False, leg_loc='upper left', grid=True,
                outfile=None, *args, **kwargs):

    _plot_kwargs = plot_kw
    plot_kw = {'figsize': (CELL_SIZE, CELL_SIZE),
               'sharex': True, 'sharey': True}
    if _plot_kwargs is not None:
        plot_kw.update(_plot_kwargs)

    _subplot_kw = subplot_kw
    subplot_kw = SUBPLOT_KWARGS
    if _subplot_kw is not None:
        subplot_kw.update(_subplot_kw)

    fig, ax = plt.subplots(1, 1, subplot_kw=subplot_kw, **plot_kw)

    if plot == 'scatter':
        ax.scatter(x, y, *args, **kwargs)
    elif plot == 'boxplot':
        ax.boxplot(x, *args, **kwargs)
    else:
        raise NotImplementedError()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(**GRID_KWARGS)

    if title is not None and title:
        ax.set_title(title, **TITLE_KWARGS)

    if xlabel is not None:
        ax.set_xlabel(xlabel, **LBL_KWARGS)

    if ylabel is not None:
        ax.set_ylabel(ylabel, **LBL_KWARGS)

    if legend and leg_loc is not None:
        ax.legend(loc=leg_loc, **LEG_KWARGS)

    plt.tight_layout()
    render(fig, outfile)


def axis_title(ax, f_title, *args):
    if isinstance(f_title, collections.Callable):
        ttl = f_title(*args)
        if ttl is not None and ttl:
            ax.set_title(ttl, **TITLE_KWARGS)


def plot_identity(ax, xvals, extend=0.2):
    xx = [xvals[0] - extend*(xvals[-1] - xvals[0]),
          xvals[-1] + extend*(xvals[-1] - xvals[0])]
    ax.plot(xx, xx, ls=':', color='black', alpha=0.6, lw=1, zorder=-500)


def bnd_extend(arr, by=0.025):
    diff = arr[-1] - arr[0]
    return arr[0] - by * diff, arr[1] + by * diff


def plot_sliced(data, slices, f_lbl=None, f_title=None, f_text=None,
                c=None, lw=DEFAULT_LINEWIDTH,
                ylim=None, xlim=None, extend_by=0.0, trim_iqr=2,
                text_kwargs=None,
                identity=False, **kwargs):

    if not isinstance(data, (tuple, list)):
        data = (data, )

    ndat = len(data)
    slices = tuple(np.atleast_1d(slices))
    if ndat > 1:
        if len(slices) == 1:
            slices = tuple(it.repeat(slices[0], ndat))
        elif len(slices) != ndat:
            raise ValueError('Data and slices length not compatible!')

    sl1 = slices[0]
    for sl in slices:
        if sl1.shape != sl.shape:
            raise ValueError('All slice arrays must have identical shape!')

    nrow, ncol, nlayer = sl1.nrow, sl1.ncol, sl1.nlayer

    xmin, xmax = np.inf, -np.inf

    if c is None:
        c = default_colors(nlayer)

    if ndat == 1:
        lstyles = default_lstyles(nlayer)
        alphas = default_alphas(nlayer)
    else:
        lstyles = ('-',) * nlayer + ('--', ) * nlayer
        alphas = (0.7, ) * nlayer + (1.0, ) * nlayer

    yy = []
    trim_iqr = float(trim_iqr)

    txt_kwargs = TEXT_KWARGS.copy()
    txt_kwargs['verticalalignment'] = 'top'
    txt_kwargs['horizontalalignment'] = 'right'
    txt_kwargs['x'] = 0.95
    txt_kwargs['y'] = 0.10

    if text_kwargs is not None:
        txt_kwargs.update(text_kwargs)

    def func(ax, idx):
        i, j = idx

        nonlocal yy, xmin, xmax

        annotations = []
        for idx_dat, (dat, sl) in enumerate(zip(data, slices)):
            for k, sl_k in enumerate(sl[i, j]):
                idx_plt = idx + (k, )
                # ix, irow, icol, iplt = sl.indices(plt_idx)
                vals = sl.values(idx_plt)
                xvals, *rest = vals

                y = dat[sl_k].squeeze()
                if ylim is None or trim_iqr is not None:
                    yy.append(y)

                if xlim is None:
                    xmin = min(np.min(vals[0]), xmin)
                    xmax = max(np.max(vals[0]), xmax)

                fmt_args = (idx_plt, idx_dat, vals)
                if isinstance(f_lbl, collections.Callable):
                    lbl = f_lbl(*fmt_args)
                elif sl.layers.labelfmt is not None:
                    lbl = sl.layers.labelfmt.format(*fmt_args)
                else:
                    lbl = None

                idx_dk = idx_dat * nlayer + k
                ax.plot(xvals, y, color=c[k], ls=lstyles[idx_dk],
                        lw=lw, alpha=alphas[idx_dk], label=lbl)

                if k == 0:
                    txt = ''
                    if isinstance(f_text, collections.Callable):
                        txt = f_text(*fmt_args)
                    else:
                        if sl.rows.labelfmt is not None:
                            txt = sl.rows.labelfmt.format(*fmt_args) + ', '
                        if sl.cols.labelfmt is not None:
                            txt += sl.cols.labelfmt.format(*fmt_args)

                    if txt:
                        annotations.append(txt.rstrip(', '))

        if identity:
            plot_identity(ax, np.array([xmin, xmax]), extend=1)

        y = txt_kwargs['y']
        txt_kwargs['transform'] = ax.transAxes
        for itxt, txt in enumerate(annotations):
            txt_kwargs.update({'s': txt, 'y': y + itxt * 0.075})
            ax.text(**txt_kwargs)

        axis_title(ax, f_title, idx)

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

    plot_grid(func, nrow=nrow, ncol=ncol, **kwargs)


def render(fig, outfile=None):
    if not outfile:
        plt.show()
    else:
        fig.savefig(outfile)
        fig.clear()