"""Test core statistical functions."""

from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import uniform

from pydynopt.stats import gini, percentile, percentile_rank, quantile, quantile_rank
from pydynopt.stats.stats import create_unique_pmf, discretize_rv


def test_gini_equal_outcomes() -> None:
    """Equal outcomes have no inequality."""
    states = np.ones(3)
    pmf = np.full(3, 1.0 / 3.0)

    assert gini(states, pmf) == pytest.approx(0.0)


def test_gini_two_point_distribution() -> None:
    """A two-point distribution has the expected Gini coefficient."""
    states = np.array([0.0, 1.0])
    pmf = np.array([0.5, 0.5])

    assert gini(states, pmf) == pytest.approx(0.5)


def test_gini_is_invariant_to_input_order() -> None:
    """Sorting states and probabilities together does not change the result."""
    states = np.array([4.0, 1.0, 2.0])
    pmf = np.array([0.5, 0.2, 0.3])
    order = np.argsort(states)

    expected = gini(states[order], pmf[order], assume_sorted=True)
    assert gini(states, pmf) == pytest.approx(expected)


def test_gini_flattens_multidimensional_inputs() -> None:
    """Multidimensional state and probability arrays are flattened."""
    states = np.array([[3.0, 1.0], [4.0, 2.0]])
    pmf = np.array([[0.3, 0.1], [0.4, 0.2]])
    expected = gini(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        assume_sorted=True,
    )

    assert gini(states, pmf) == pytest.approx(expected)


def test_create_unique_pmf_combines_and_normalizes_duplicates() -> None:
    """Duplicate states are combined into a normalized PMF."""
    states = np.array([2.0, 1.0, 2.0, 1.0])
    pmf = np.array([1.0, 2.0, 3.0, 4.0])

    states_unique, pmf_unique = create_unique_pmf(states, pmf)

    np.testing.assert_array_equal(states_unique, [1.0, 2.0])
    np.testing.assert_allclose(pmf_unique, [0.6, 0.4])

    states_sorted, pmf_sorted = create_unique_pmf(
        np.array([1.0, 1.0, 2.0, 2.0]),
        np.array([2.0, 4.0, 1.0, 3.0]),
        assume_sorted=True,
    )
    np.testing.assert_array_equal(states_sorted, states_unique)
    np.testing.assert_allclose(pmf_sorted, pmf_unique)


def test_quantile_nearest_discrete_boundaries() -> None:
    """Nearest quantiles select the correct state at CDF boundaries."""
    states = np.array([1.0, 2.0, 3.0])
    pmf = np.array([0.2, 0.3, 0.5])
    ranks = np.array([0.0, 0.2, 0.2001, 0.5, 0.5001, 1.0])

    result = quantile(states, pmf, ranks)

    np.testing.assert_array_equal(result, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])


def test_quantile_sorts_discrete_support() -> None:
    """Quantiles sort unsorted states together with their probabilities."""
    states = np.array([3.0, 1.0, 2.0])
    pmf = np.array([0.5, 0.2, 0.3])
    ranks = np.array([0.2, 0.5, 1.0])

    result = quantile(states, pmf, ranks)

    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_quantile_combines_duplicate_states() -> None:
    """Quantiles use aggregated probabilities for duplicate states."""
    states = np.array([2.0, 1.0, 2.0, 1.0])
    pmf = np.array([0.1, 0.2, 0.3, 0.4])
    ranks = np.array([0.0, 0.6, 0.6001, 1.0])

    result = quantile(states, pmf, ranks)

    np.testing.assert_array_equal(result, [1.0, 1.0, 2.0, 2.0])


def test_quantile_linearly_interpolates_discrete_support() -> None:
    """Linear interpolation uses adjacent support points and CDF values."""
    states = np.array([0.0, 10.0])
    pmf = np.array([0.25, 0.75])
    ranks = np.array([0.0, 0.25, 0.625, 1.0])

    result = quantile(states, pmf, ranks, interpolation='linear')

    np.testing.assert_allclose(result, [0.0, 0.0, 5.0, 10.0])


@pytest.mark.parametrize(
    ('interpolation', 'expected'),
    [
        ('nearest', [0.0, 2.0, 2.0]),
        ('linear', [0.0, 1.000004, 2.0]),
    ],
)
def test_quantile_handles_flat_cdf_regions(
    interpolation: str,
    expected: list[float],
) -> None:
    """Zero-probability states do not prevent quantile lookup."""
    states = np.array([0.0, 1.0, 2.0])
    pmf = np.array([0.25, 0.0, 0.75])
    ranks = np.array([0.25, 0.250003, 1.0])

    result = quantile(states, pmf, ranks, interpolation=interpolation)

    np.testing.assert_allclose(result, expected)


def test_quantile_interpolates_within_bins() -> None:
    """Bin-edge distributions force interpolation within each bin."""
    edges = np.array([0.0, 1.0, 3.0])
    pmf = np.array([0.25, 0.75])
    ranks = np.array([0.0, 0.125, 0.25, 0.625, 1.0])

    result = quantile(edges, pmf, ranks, interpolation='nearest')

    np.testing.assert_allclose(result, [0.0, 0.5, 1.0, 2.0, 3.0])


def test_quantile_and_percentile_scalar_and_array_returns() -> None:
    """Scalar ranks return scalars while rank arrays return arrays."""
    states = np.array([1.0, 2.0, 3.0])
    pmf = np.array([0.2, 0.3, 0.5])

    q_scalar = quantile(states, pmf, 0.5)
    p_scalar = percentile(states, pmf, 50.0)
    q_array = quantile(states, pmf, np.array([0.5]))
    p_array = percentile(states, pmf, np.array([50.0]))

    assert np.isscalar(q_scalar)
    assert np.isscalar(p_scalar)
    assert q_scalar == p_scalar == pytest.approx(2.0)
    assert isinstance(q_array, np.ndarray)
    assert isinstance(p_array, np.ndarray)
    np.testing.assert_array_equal(q_array, [2.0])
    np.testing.assert_array_equal(p_array, [2.0])


def test_percentile_rescales_quantile_ranks() -> None:
    """Percentiles agree with quantiles after rank rescaling."""
    states = np.array([1.0, 2.0, 3.0])
    pmf = np.array([0.2, 0.3, 0.5])
    percentile_ranks = np.array([0.0, 20.0, 50.0, 100.0])

    result = percentile(states, pmf, percentile_ranks)
    expected = quantile(states, pmf, percentile_ranks / 100.0)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ('function', 'rank'),
    [
        (quantile, -0.01),
        (quantile, 1.01),
        (percentile, -1.0),
        (percentile, 101.0),
    ],
)
def test_quantile_and_percentile_reject_invalid_ranks(
    function: Callable[..., object],
    rank: float,
) -> None:
    """Ranks outside the supported interval are rejected."""
    with pytest.raises(ValueError, match='Invalid percentile rank'):
        function(np.array([0.0, 1.0]), np.array([0.5, 0.5]), rank)


def test_quantile_rejects_incompatible_lengths() -> None:
    """State and PMF lengths must describe points or bins."""
    with pytest.raises(ValueError, match='Non-conformable'):
        quantile(
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0]),
            np.array([0.5]),
        )


def test_quantile_rejects_unsupported_interpolation() -> None:
    """Unknown interpolation methods are rejected for discrete support."""
    with pytest.raises(ValueError, match='Unsupported interpolation'):
        quantile(
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            np.array([0.5]),
            interpolation='cubic',
        )


def test_quantile_accepts_empty_rank_array() -> None:
    """An empty rank array produces an empty quantile array."""
    result = quantile(
        np.array([0.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([]),
    )

    assert isinstance(result, np.ndarray)
    assert result.size == 0


@pytest.mark.parametrize(
    'pmf',
    [
        np.array([0.0, 0.0]),
        np.array([0.5, np.nan]),
    ],
)
def test_quantile_returns_nan_for_invalid_pmf(pmf: np.ndarray) -> None:
    """Zero-mass and NaN PMFs produce NaN quantiles."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = quantile(
            np.array([0.0, 1.0]),
            pmf,
            np.array([0.25, 0.75]),
        )

    assert np.all(np.isnan(result))


def test_quantile_rank_discrete_distribution() -> None:
    """Quantile ranks interpolate over a discrete distribution's CDF."""
    states = np.array([0.0, 1.0, 2.0])
    pmf = np.array([0.2, 0.3, 0.5])
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0])

    result = quantile_rank(states, pmf, values)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(
        result,
        [np.nan, 0.2, 0.35, 0.5, 1.0, np.nan],
        equal_nan=True,
    )


def test_quantile_rank_scalar_and_shaped_returns() -> None:
    """Quantile-rank output follows scalar status and input shape."""
    states = np.array([0.0, 1.0, 2.0])
    pmf = np.array([0.2, 0.3, 0.5])
    values = np.array([[0.0, 0.5], [1.0, 2.0]])

    scalar = quantile_rank(states, pmf, 1.0)
    percentile_scalar = percentile_rank(states, pmf, 1.0)
    shaped = quantile_rank(states, pmf, values)

    assert isinstance(scalar, float)
    assert scalar == pytest.approx(0.5)
    assert isinstance(percentile_scalar, float)
    assert percentile_scalar == pytest.approx(50.0)
    assert isinstance(shaped, np.ndarray)
    assert shaped.shape == values.shape
    np.testing.assert_allclose(shaped, [[0.2, 0.35], [0.5, 1.0]])


def test_quantile_rank_interpolates_bin_edges() -> None:
    """Quantile ranks interpolate over a distribution represented by bins."""
    edges = np.array([0.0, 1.0, 3.0])
    pmf = np.array([0.25, 0.75])
    values = np.array([0.0, 0.5, 1.0, 2.0, 3.0])

    result = quantile_rank(edges, pmf, values)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.0, 0.125, 0.25, 0.625, 1.0])


def test_percentile_rank_rescales_quantile_rank() -> None:
    """Percentile ranks are quantile ranks expressed on a 0--100 scale."""
    states = np.array([0.0, 1.0, 2.0])
    pmf = np.array([0.2, 0.3, 0.5])
    values = np.array([0.0, 0.5, 1.0, 2.0])

    result = percentile_rank(states, pmf, values)
    expected = quantile_rank(states, pmf, values)

    assert isinstance(result, np.ndarray)
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(result, expected * 100.0)


def test_rank_functions_return_none_for_incompatible_lengths() -> None:
    """Rank functions signal incompatible state and PMF lengths with None."""
    states = np.array([0.0, 1.0, 2.0])
    pmf = np.array([1.0])

    assert quantile_rank(states, pmf, 0.5) is None
    assert percentile_rank(states, pmf, 0.5) is None


def test_discretize_rv_defaults_to_one_standard_normal_bin() -> None:
    """The default discretization has one unit-mass bin centered at zero."""
    result = discretize_rv()

    assert len(result) == 2
    grid, pmf = result[0], result[1]
    np.testing.assert_allclose(grid, [0.0], atol=1.0e-12)
    np.testing.assert_allclose(pmf, [1.0])


def test_discretize_rv_creates_equal_probability_bins() -> None:
    """A requested bin count creates equally probable quantile bins."""
    result = discretize_rv(n=2, dist=uniform)

    assert len(result) == 2
    grid, pmf = result[0], result[1]
    np.testing.assert_allclose(grid, [0.25, 0.75])
    np.testing.assert_allclose(pmf, [0.5, 0.5])


def test_discretize_rv_quantiles_override_n_and_forward_kwargs() -> None:
    """Explicit quantiles override n and distribution kwargs are forwarded."""
    result = discretize_rv(
        n=99,
        q=[0.0, 0.25, 1.0],
        dist=uniform,
        return_edges=True,
        loc=2.0,
        scale=4.0,
    )

    assert len(result) == 3
    grid, pmf, edges = result[0], result[1], result[-1]
    np.testing.assert_allclose(edges, [2.0, 3.0, 6.0])
    np.testing.assert_allclose(pmf, [0.25, 0.75])
    np.testing.assert_allclose(grid, [2.5, 4.5])
