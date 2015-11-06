"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from __future__ import print_function, division, absolute_import


from pydynopt.plot import PlotMap, DefaultStyle
import numpy as np

import matplotlib.pyplot as plt


def data_1d():
    xx = np.linspace(0, 1, 101)
    f = np.sqrt(xx)

    return xx, f


def data_4d():
    x1 = np.linspace(0, 1, 101)
    x2 = np.linspace(-2, 2, 5)
    x3 = np.linspace(-2, 2, 3)
    x4 = np.linspace(-1, 1, 7)

    xx1, xx2, xx3, xx4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')
    f = xx2 * np.sqrt(xx1) + xx3 + xx4 * np.sin(xx1)

    return x1, x2, x3, x4, f


def demo_1d():
    """
    Simple examples with 1-dimensional objects to illustrate some key concepts.

    Note: the calls to waitforbuttonpress() are only for demonstration
    purposes and not part of the PlotMap API.

    Note 2: PlotMap.plot() does not require a style argument, but we use it
    here to make demo plots more compact.
    """

    # Create some demo data
    x, f = data_1d()

    # Use a style with smaller subplots
    style = DefaultStyle()
    style.cell_size = 4

    # if no x-axis values are provided, then this is the same as plotting
    # against the array indices.
    pm = PlotMap()
    pm.plot(f, style=style)
    plt.waitforbuttonpress()

    # We can use the values parameter when calling map_xaxis() to specify
    # x-values that will be used on ALL subplots
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x)
    pm.plot(f, style=style)
    plt.waitforbuttonpress()

    # We can create artificial layers even with only one-dimensional objects to
    # create legends or assign row / column labels.
    pm = PlotMap()
    pm.map_xaxis(dim=1, values=x)
    pm.map_layers(dim=0, label='Some static legend')
    # Note that now we explicitly need to pass a 2-dimensional array since we
    # are mapping those dimensions
    pm.plot(f.reshape((1, -1)), style=style)
    plt.waitforbuttonpress()

    # Alternatively, we can create row / column labels in the same way
    pm = PlotMap()
    pm.map_xaxis(dim=1, values=x)
    pm.map_rows(dim=0, label='Static row label')
    pm.plot(f.reshape((1, -1)), style=style)
    plt.waitforbuttonpress()


def demo_nd():
    """
    Demonstrate examples using n-dimensional arrays

    Note: the calls to waitforbuttonpress() are only for demonstration
    purposes and not part of the PlotMap API.

    Note 2: PlotMap.plot() does not require a style argument, but we use it
    here to make demo plots more compact.
    """

    x0, x1, x2, x3, f = data_4d()

    # Use a style with smaller subplots
    style = DefaultStyle()
    style.cell_size = 4

    # When plotting multi-dimensional data, we either have to map each
    # non-trivial dimension (ie. dimensions with length > 1) to some plot
    # axis (rows, columns, layers or the x-axis), or fix excess dimensions at
    # a specific index.

    # First we demonstrate plotting when one dimension is fixed
    fixed_at = f.shape[1] // 2
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x0)
    pm.map_columns(dim=2, values=x2, label_fmt='x2: {value:-04.2f}')
    pm.map_layers(dim=3, values=x3, label_fmt='x3: {value:-04.2f}')
    pm.add_fixed(dim=1, at_idx=fixed_at)
    pm.plot(f, style=style)
    plt.waitforbuttonpress()
    plt.close()

    # Arrays of up to 4 dimensions can be mapped to a grid of plots without
    # needing to fix any dimension. Below we map all three dimensions of
    # data_3d.
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x0)
    pm.map_rows(dim=1, values=x1, label_fmt='x1: {value:-04.2f}')
    pm.map_columns(dim=2, values=x2, label_fmt='x2: {value:-04.2f}')
    pm.map_layers(dim=3, values=x3, label_fmt='x3: {value:-04.2f}')
    pm.plot(f, style=style)
    plt.waitforbuttonpress()
    plt.close()

    # Plotting at specific values / indices
    # Plots can be restricted to specific indices or values. To accomplish
    # this, the `at_idx` argument can be passed a slice() or a list-like
    # object. Alternatively, the `at_val` argument can be used to select
    # specific elements from the `values` list.
    pm = PlotMap()
    # Use a slice to select only the first half of x-values
    pm.map_xaxis(dim=0, at_idx=slice(len(x0) // 2), values=x0)
    # Show only the first and last element of the row dimension
    pm.map_rows(dim=1, at_idx=(0, len(x1) - 1),
                values=x1, label_fmt='x1: {value:-04.2f}')
    # Plot only for values smaller than 2 in column dimension
    val = x2[x2 < 2]
    pm.map_columns(dim=2, values=x2, at_val=val,
                   label_fmt='x2: {value:-04.2f}')

    # Plot only every second value in layer dimension
    pm.map_layers(dim=3, values=x3, at_idx=slice(0, None, 2),
                  label_fmt='x3: {value:-04.2f}')
    pm.plot(f, style=style)
    plt.waitforbuttonpress()
    plt.close()


def demo_annotations():
    """
    Demonstrate the use of labels and legends

    Note: the calls to waitforbuttonpress() are only for demonstration
    purposes and not part of the PlotMap API.

    Note 2: PlotMap.plot() does not require a style argument, but we use it
    here to make demo plots more compact.
    """

    x0, x1, x2, x3, f = data_4d()

    style = DefaultStyle()
    style.cell_size = 5

    # Labels are meant to display row- and column-specific values in plots
    # with multiple rows / columns. They are controlled using the `label`
    # and `label_fmt` argument.

    # The use of `label_fmt` has been demonstrated in demo_nd(). The
    # `label_fmt` string is evaluated as label_fmt.format(**kwargs) using
    # Python's standard format() function. The keyword arguments passed to
    # format() are the attributes of pydynopt.plot.plotmap.LabelArgs that are
    # not None; these are: row, column, layer, index and value. The first
    # three values give the current plot coordinates within the plot grid.
    # They can be used directly in the format string and will be
    # substituted by the appropriate values. Not that `value` can only be
    # substituted if the values= argument was passed to the map_* function.
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x0)
    pm.map_rows(dim=1, values=x1, at_val=(x1[0], x1[-1]),
                label_fmt='i={index:d} v={value: 04.2f}')
    pm.map_columns(dim=2, values=x2, at_val=(x2[0], x2[-1]),
                   label_fmt='row={row:d} col={column:d}')
    pm.map_layers(dim=3, values=x3, label_fmt='i={index:d}; v={value: 04.2f}')
    pm.plot(f, style=style)
    plt.waitforbuttonpress()
    plt.close()

    # Static annotations
    # Alternatively, a static list of labels can be passed that is
    # conformable with the number of conditioned-on values for each dimension:
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x0)
    pm.add_fixed(dim=(1, 2), at_idx=(0, 0))
    pm.map_layers(dim=3, at_idx=(0, f.shape[3] - 1), values=x3,
                  label=['First', 'Last'])
    pm.plot(f, style=style)
    plt.waitforbuttonpress()
    plt.close()

    # Legends
    # The legend location can be adjusted using the legend_loc and legend_at
    # arguments. `legend_loc` takes the same values as the `loc` argument of
    # Matplotlib's legend() method (ie. upper/center/lower +
    # left/center/right). The `legend_at` argument specifies the subplot in
    # which the legend is to be shown, where the argument is a tuple (row,
    # col). The legend is shown only in a single subplot.
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x0)
    pm.map_rows(dim=1, values=x1, at_val=(x1[0], x1[-1]),
                label_fmt='i={index:d} v={value: 04.2f}')
    pm.map_columns(dim=2, values=x2, at_val=(x2[0], x2[-1]),
                   label_fmt='row={row:d} col={column:d}')
    pm.map_layers(dim=3, values=x3, label_fmt='i={index:d}; v={value: 04.2f}')
    pm.plot(f, style=style, legend=True, legend_loc='upper left',
            legend_at=(1, 0))
    plt.waitforbuttonpress()
    plt.close()


def demo_advanced():
    """
    To be done...
    """
    pass

if __name__ == '__main__':
    # demo_1d()
    demo_nd()
    demo_annotations()
    demo_advanced()
