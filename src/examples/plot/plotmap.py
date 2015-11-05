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


def data_3d():
    x = np.linspace(0, 1, 101)
    y = np.linspace(-2, 2, 5)
    z = np.linspace(-2, 2, 3)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    f = yy * np.sqrt(xx) + zz

    return x, y, z, f


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
    plt.waitforbuttonpress(5)

    # We can use the values parameter when calling map_xaxis() to specify
    # x-values that will be used on ALL subplots
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x)
    pm.plot(f, style=style)
    plt.waitforbuttonpress(5)

    # We can create artificial layers even with only one-dimensional objects to
    # create legends or assign row / column labels.
    pm = PlotMap()
    pm.map_xaxis(dim=1, values=x)
    pm.map_layers(dim=0, label='Some static legend')
    # Note that now we explicitly need to pass a 2-dimensional array since we
    # are mapping those dimensions
    pm.plot(f.reshape((1, -1)), style=style)
    plt.waitforbuttonpress(5)

    # Alternatively, we can create row / column labels in the same way
    pm = PlotMap()
    pm.map_xaxis(dim=1, values=x)
    pm.map_rows(dim=0, label='Static row label')
    pm.plot(f.reshape((1, -1)), style=style)
    plt.waitforbuttonpress(5)


def demo_nd():
    """
    Demonstrate examples using n-dimensional arrays

    Note: the calls to waitforbuttonpress() are only for demonstration
    purposes and not part of the PlotMap API.

    Note 2: PlotMap.plot() does not require a style argument, but we use it
    here to make demo plots more compact.
    """

    x, y, z, f = data_3d()

    # Use a style with smaller subplots
    style = DefaultStyle()
    style.cell_size = 4

    # When plotting multi-dimensional data, we either have to map each
    # non-trivial dimension (ie. dimensions with length > 1) to some plot
    # axis (rows, columns, layers or the x-axis), or fix excess dimensions at
    # a specific index.

    # First we demonstrate plotting when one dimension is fixed
    fixed_at = f.shape[2] // 2
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x)
    pm.map_layers(dim=1, values=y, label_fmt='Y: {value:-04.2f}')
    pm.add_fixed(dim=2, at_idx=fixed_at)
    pm.plot(f, style=style)
    plt.waitforbuttonpress(5)

    # Arrays of up to 4 dimensions can be mapped to a grid of plots without
    # needing to fix any dimension. Below we map all three dimensions of
    # data_3d.
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x)
    pm.map_layers(dim=1, values=y, label_fmt='Y: {value:-04.2f}')
    pm.map_columns(dim=2, values=z, label_fmt='Z: {value:-04.2f}')
    pm.plot(f, style=style)
    plt.waitforbuttonpress(5)


def demo_annotations():
    """
    Demonstrate the use of labels and legends

    Note: the calls to waitforbuttonpress() are only for demonstration
    purposes and not part of the PlotMap API.

    Note 2: PlotMap.plot() does not require a style argument, but we use it
    here to make demo plots more compact.
    """

    x, y, z, f = data_3d()

    style = DefaultStyle()
    style.cell_size = 5

    # Labels are meant to display row- and column-specific values in plots
    # with multiple rows / columns. They are controlled using the `label`
    # and `label_fmt` argument.

    # The use of `label_fmt` has been demonstrated in demo_3d(). The
    # `label_fmt` string is evaluated as label_fmt.format(**kwargs) using
    # Python's standard format() function. The keyword arguments passed to
    # format() are the attributes of pydynopt.plot.plotmap.LabelArgs that are
    # not None; these are: row, column, layer, index and value.
    # They can be used directly in the format string and will be
    # substituted by the appropriate values. Not that 'value' can only be
    # substituted if the values= argument was passed to the map_* function.
    pm = PlotMap()
    pm.map_xaxis(dim=0, values=x)
    pm.map_layers(dim=1, values=y, label_fmt='i={index:d}; v={value: 04.2f}')
    pm.map_columns(dim=2, values=z, label_fmt='column: {column:d}')
    pm.plot(f, style=style)
    plt.waitforbuttonpress(5)


def demo_advanced():
    pass

if __name__ == '__main__':
    demo_1d()
    demo_nd()
    demo_annotations()
    demo_advanced()