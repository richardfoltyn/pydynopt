"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from pydynopt.plot.plotmap import PlotMap, PlotDimension
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
    g2 = np.linspace(10, 20, d1.data.shape[0] + 1)

    xx, yy = np.meshgrid(g1, g2, indexing='ij')
    x.data = yy + np.sqrt(xx)
    x.grid.extend([g1, g2])

    return x


@pytest.fixture(scope='module')
def data_3d():
    x = MockData()
    d2 = data_2d()

    g1, g2 = d2.grid
    g3 = np.linspace(-1, 1, d2.data.shape[1] + 1)
    xx, yy, zz = np.meshgrid(g1, g2, g3, indexing='ij')

    x.data = yy + zz * np.sqrt(xx)
    x.grid.extend([g1, g2, g3])

    return x


def test_plotdim(request):
    """
    Test PlotDimensions creation
    """

    # Check that exception in raised when at_idx and at_val are non-conformable
    idx = np.arange(10)
    val = np.linspace(0, 10, 9)

    with pytest.raises(ValueError):
        _ = PlotDimension(dim=0, at_idx=idx, at_val=val)

    # check that fixed dimension required integer at_idx
    with pytest.raises(TypeError):
        _ = PlotDimension(dim=0, at_idx=slice(None), fixed=True)

    # Check that fixed dimension can be constructed by value
    pd = PlotDimension(dim=0, at_val=1, values=np.arange(10), fixed=True)
    assert pd.index == 1


def test_plotmap(request):

    # Check that detecting duplicate mappings works
    pd1 = PlotDimension(dim=0)
    pd2 = PlotDimension(dim=0)

    with pytest.raises(ValueError):
        _ = PlotMap(xaxis=pd1, rows=pd2)
    with pytest.raises(ValueError):
        _ = PlotMap(xaxis=pd1, fixed=PlotDimension(dim=0, at_idx=0))

    # Check that this works when adding via methods
    pm = PlotMap(xaxis=PlotDimension(dim=0))
    with pytest.raises(ValueError):
        pm.map_rows(dim=0)
    with pytest.raises(ValueError):
        pm.map_columns(dim=0)
    with pytest.raises(ValueError):
        pm.map_layers(dim=0)
    with pytest.raises(ValueError):
        pm.add_fixed(dim=0, at_idx=0)


def assert_data_shape(pm, data, desired):
    data_aligned, pm_aligned = pm.apply(data)

    assert data_aligned.ndim == 4
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

    # Determine index from at_val and values arguments
    pm = PlotMap()
    values = data_1d.data
    at_val = values[::2]
    pm.map_xaxis(dim=0, at_val=at_val, values=values)
    desired = at_val[None, None, None, :]
    assert_data_shape(pm, values, desired)


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

    # Determine row indices from at_val and values
    values = data_2d.grid[0]
    at_val = values[:len(values)//2]
    desired = data_2d.data[:len(values)//2, None, None, :]
    pm = PlotMap()
    pm.map_xaxis(dim=1)
    pm.map_rows(dim=0, at_val=at_val, values=values)
    assert_data_shape(pm, data_2d.data, desired)

    # Determine row indices and x-axis from at_val and values
    values1 = data_2d.grid[1]
    at_val1 = values1[len(values1)//2:]
    desired = data_2d.data[None, :len(values)//2, None, len(values1)//2:]
    pm = PlotMap()
    pm.map_xaxis(dim=1, at_val=at_val1, values=values1)
    pm.map_columns(dim=0, at_val=at_val, values=values)
    assert_data_shape(pm, data_2d.data, desired)


def test_3d(request, data_3d):
    """
    Test PlotMap with 3-dimensional data
    """

    dd = data_3d.data

    # Test 1: Plot everything
    desired = dd[:, :, None, :]
    pm = PlotMap()
    pm.map_xaxis(dim=2)
    pm.map_rows(dim=0)
    pm.map_columns(dim=1)

    assert_data_shape(pm, dd, desired)

    # Test 2: Plot everything, change axis order
    pm = PlotMap()
    pm.map_xaxis(dim=2)
    pm.map_rows(dim=1)
    pm.map_columns(dim=0)
    desired = np.swapaxes(data_3d.data, 0, 1)
    desired = desired[:, :, None, :]

    assert_data_shape(pm, dd, desired)

    # Test 3: select some subset of indices on each axis
    i1 = np.random.randint(0, dd.shape[0] - 1, size=dd.shape[0] // 2)
    i2 = np.random.randint(0, dd.shape[1] - 1, size=dd.shape[1] // 2)
    i3 = np.random.randint(0, dd.shape[2] - 1, size=dd.shape[2] // 2)

    idx = np.ix_(i1, i2, i3)
    desired = data_3d.data[idx][:, :, None, :]

    pm = PlotMap()
    pm.map_xaxis(dim=2, at_idx=i3)
    pm.map_rows(dim=0, at_idx=i1)
    pm.map_columns(dim=1, at_idx=i2)

    assert_data_shape(pm, dd, desired)




