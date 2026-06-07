"""
Unit tests for linear model utilities, specifically demean_within.
"""

import numpy as np
import pandas as pd
import pytest

from pydynopt.stats.lm import demean_within


def test_demean_within_series_no_weights():
    # Setup data: 2 groups, 3 observations in group A, 2 in group B
    index = pd.Index(['A', 'A', 'A', 'B', 'B'], name='group')
    x = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0], index=index, name='val')

    # Demean without weights
    x_dem, x_mean = demean_within(x, 'group')

    # Expected means: group A mean is 2.0, group B mean is 15.0
    # Expected demeaned:
    # A: [1.0 - 2.0, 2.0 - 2.0, 3.0 - 2.0] = [-1.0, 0.0, 1.0]
    # B: [10.0 - 15.0, 20.0 - 15.0] = [-5.0, 5.0]
    expected_dem = pd.Series([-1.0, 0.0, 1.0, -5.0, 5.0], index=index, name='val')
    
    # Expected x_mean_first (group means, one per group)
    expected_mean = pd.Series([2.0, 15.0], index=pd.Index(['A', 'B'], name='group'), name='val')

    pd.testing.assert_series_equal(x_dem, expected_dem)
    pd.testing.assert_series_equal(x_mean, expected_mean)


def test_demean_within_series_weighted():
    # Setup data
    index = pd.Index(['A', 'A', 'B', 'B'], name='group')
    x = pd.Series([1.0, 2.0, 10.0, 20.0], index=index, name='val')
    
    # Case 1: rescale_weights=False (weights already sum to 1 within group)
    # Group A weights: 0.25, 0.75 (sum to 1) -> mean: 1.0*0.25 + 2.0*0.75 = 1.75
    # Group B weights: 0.4, 0.6 (sum to 1) -> mean: 10.0*0.4 + 20.0*0.6 = 16.0
    weights = pd.Series([0.25, 0.75, 0.4, 0.6], index=index, name='weight')
    
    x_dem, x_mean = demean_within(x, 'group', weights=weights, rescale_weights=False)
    
    expected_dem = pd.Series([1.0 - 1.75, 2.0 - 1.75, 10.0 - 16.0, 20.0 - 16.0], index=index, name='val')
    expected_mean = pd.Series([1.75, 16.0], index=pd.Index(['A', 'B'], name='group'), name='val')
    
    pd.testing.assert_series_equal(x_dem, expected_dem, check_names=False)
    pd.testing.assert_series_equal(x_mean, expected_mean, check_names=False)

    # Case 2: rescale_weights=True (weights do not sum to 1 within group)
    # Group A weights: 1.0, 3.0 -> sum=4.0 -> normalized: 0.25, 0.75
    # Group B weights: 2.0, 3.0 -> sum=5.0 -> normalized: 0.4, 0.6
    weights_raw = pd.Series([1.0, 3.0, 2.0, 3.0], index=index, name='weight')
    
    x_dem, x_mean = demean_within(x, 'group', weights=weights_raw, rescale_weights=True)
    
    pd.testing.assert_series_equal(x_dem, expected_dem, check_names=False)
    pd.testing.assert_series_equal(x_mean, expected_mean, check_names=False)


def test_demean_within_dataframe():
    # Setup data with DataFrame
    index = pd.Index(['A', 'A', 'B', 'B'], name='group')
    df = pd.DataFrame({
        'val1': [1.0, 3.0, 10.0, 30.0],
        'val2': [2.0, 4.0, 20.0, 40.0]
    }, index=index)
    
    # Without weights
    df_dem, df_mean = demean_within(df, 'group')
    
    # Expected means:
    # val1: A -> 2.0, B -> 20.0
    # val2: A -> 3.0, B -> 30.0
    # Expected demeaned:
    # val1: A -> [-1.0, 1.0], B -> [-10.0, 10.0]
    # val2: A -> [-1.0, 1.0], B -> [-10.0, 10.0]
    expected_dem = pd.DataFrame({
        'val1': [-1.0, 1.0, -10.0, 10.0],
        'val2': [-1.0, 1.0, -10.0, 10.0]
    }, index=index)
    
    expected_mean = pd.DataFrame({
        'val1': [2.0, 20.0],
        'val2': [3.0, 30.0]
    }, index=pd.Index(['A', 'B'], name='group'))
    
    pd.testing.assert_frame_equal(df_dem, expected_dem)
    pd.testing.assert_frame_equal(df_mean, expected_mean)


def test_demean_within_multiple_groups():
    # Setup data with multi-group variables
    index = pd.MultiIndex.from_tuples([
        ('A', 1), ('A', 1), ('A', 2), ('B', 1), ('B', 1)
    ], names=['group1', 'group2'])
    
    x = pd.Series([1.0, 3.0, 10.0, 100.0, 200.0], index=index, name='val')
    
    x_dem, x_mean = demean_within(x, ['group1', 'group2'])
    
    # Expected means:
    # (A, 1) -> mean of [1.0, 3.0] = 2.0
    # (A, 2) -> mean of [10.0] = 10.0
    # (B, 1) -> mean of [100.0, 200.0] = 150.0
    # Expected demeaned:
    # (A, 1) -> [-1.0, 1.0]
    # (A, 2) -> [0.0]
    # (B, 1) -> [-50.0, 50.0]
    expected_dem = pd.Series([-1.0, 1.0, 0.0, -50.0, 50.0], index=index, name='val')
    
    expected_mean = pd.Series(
        [2.0, 10.0, 150.0],
        index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['group1', 'group2']),
        name='val'
    )
    
    pd.testing.assert_series_equal(x_dem, expected_dem)
    pd.testing.assert_series_equal(x_mean, expected_mean)


if __name__ == '__main__':
    pytest.main([__file__])
