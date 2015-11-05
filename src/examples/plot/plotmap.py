"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from __future__ import print_function, division, absolute_import


from pydynopt.plot import PlotMap
import numpy as np

### Simple examples involving only a single PlotMap object

## Plotting 1-d objects

# Simple examples with 1-dimensional objects to illustrate some concepts
# Create some mock data
xx = np.linspace(0, 1, 11)
# if no x-axis values are provided, then this is the same as plotting against
# the array indices.
pm = PlotMap()
pm.plot(xx)

# We can use the values parameter when calling map_xaxis() to specify
# x-values that will be used on ALL subplots
xx = np.linspace(0, 1, 101)
yy = np.sqrt(xx)
pm = PlotMap()
pm.map_xaxis(dim=0, values=xx)
pm.plot(yy)

# We can create artificial layers even with only one-dimensional objects to
# create legends or assign row / column labels.
pm = PlotMap()
pm.map_xaxis(dim=1, values=xx)
pm.map_layers(dim=0, label='Some static legend')
# Note that now we explicitly need to pass a 2-dimensional array since we
# are mapping those dimensions
pm.plot(yy.reshape((1, -1)))

# Alternatively, we can create row / column labels in the same way
pm = PlotMap()
pm.map_xaxis(dim=1, values=xx)
pm.map_rows(dim=0, label='Static row label')
pm.plot(yy.reshape((1, -1)))

