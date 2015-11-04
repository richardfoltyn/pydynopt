"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from pydynopt.plot import PlotMap
from common import MockData


@pytest.fixture(scope='module')
def data_1d():
    x = MockData()
    g = np.linspace(0, 10, 10)
    x.data = np.sqrt(g)
    x.grid.append(g)

    return x


@pytest.fixture(scope='module')
def data_2d():
    x = MockData()

    d1 = data_1d()
    g1 = d1.grid[0]
    g2 = np.linspace(10, 20, 10)

    xx, yy = np.meshgrid(g1, g2, indexing='ij')
    x.data = yy + np.sqrt(xx)
    x.grid.extend([g1, g2])

    return x


@pytest.fixture(scope='module')
def data_3d():
    x = MockData()
    d2 = data_2d()

    g1, g2 = d2.grid
    g3 = np.linspace(-1, 1, 10)
    xx, yy, zz = np.meshgrid(g1, g2, g3, indexing='ij')

    x.data = yy + zz * np.sqrt(xx)
    x.grid.extend([g1, g2, g3])


def assert_data_shape(pm, data, desired):
    data_aligned, pm_aligned = pm.apply(data)

    assert np.all(data_aligned.shape == desired.shape)
    assert np.all(data_aligned == desired)


def test_1d(request, data_1d):

    # insert default axes for rows, columns, layers
    desired = data_1d.data[None, None, None, :]

    # most basic setup
    pm = PlotMap()
    assert_data_shape(pm, data_1d.data, desired)

    # Explicitly set x-axis to axis 0
    pm = PlotMap()
    pm.map_xaxis(dim=0)
    assert_data_shape(pm, data_1d.data, desired)

    # Explicitly set x-axis and index
    pm = PlotMap()
    pm.map_xaxis(dim=0, at_idx=np.arange(data_1d.data.shape[0]))
    assert_data_shape(pm, data_1d.data, desired)

    # Pass expanded array, do not specify fixed dimensions
    pm = PlotMap()
    pm.map_xaxis(dim=3)
    assert_data_shape(pm, desired, desired)

    # Pass expanded array, map some dimensions explicitly
    pm = PlotMap()
    pm.map_xaxis(dim=3)
    pm.map_rows(dim=0)
    assert_data_shape(pm, desired, desired)

    # Pass expanded array, map rows and columns explicitly
    pm = PlotMap()
    pm.map_xaxis(dim=3)
    pm.map_rows(dim=0)
    pm.map_columns(dim=1)
    assert_data_shape(pm, desired, desired)

    # Pass expanded array, map rows, columns and layers explicitly
    pm = PlotMap()
    pm.map_xaxis(dim=3)
    pm.map_rows(dim=0)
    pm.map_columns(dim=1)
    pm.map_layers(dim=2)
    assert_data_shape(pm, desired, desired)


def test_2d(request, data_2d):
    # insert default axes for rows and columns
    desired = data_2d.data[None, None, :, :]

    # check that exception is thrown when we do not map non-degenerate dimension
    pm = PlotMap()
    pm.map_xaxis(dim=1)
    with pytest.raises(RuntimeError):
        _ = pm.apply(data_2d.data)

    pm.map_layers(dim=0)
    assert_data_shape(pm, data_2d.data, desired)

    # Check auto-drop of degenerate unmapped dimensions
    pm = PlotMap()
    pm.map_xaxis(dim=3)
    pm.map_layers(dim=2)
    assert_data_shape(pm, desired, desired)

    # Check some other combination of rows/cols/layers spec: map axis=0 into
    # rows, and degenerate axis=1 into layers.
    xx = np.expand_dims(data_2d.data, axis=1)
    desired = xx[slice(0, None, 2), None, :, :]
    pm = PlotMap()
    pm.map_rows(dim=0, at_idx=slice(0, None, 2))
    pm.map_xaxis(dim=2)
    pm.map_layers(dim=1)
    assert_data_shape(pm, xx, desired)

    # Fix dimension 0 at particular value
    idx = data_2d.data.shape[0] // 2
    desired = data_2d.data[None, None, (idx, ), :]
    pm = PlotMap()
    pm.map_xaxis(dim=1)
    pm.add_fixed(dim=0, at_idx=idx)
    assert_data_shape(pm, data_2d.data, desired)






