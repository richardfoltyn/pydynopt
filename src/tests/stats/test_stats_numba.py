"""Test statistical functions from actual Numba-compiled callers."""

from typing import Any

from numba import njit
import numpy as np
from numpy.typing import NDArray
import pytest

from pydynopt.stats import gini, percentile, quantile
from pydynopt.stats.stats import create_unique_pmf

type FloatArray = NDArray[np.float64]

_gini_any: Any = gini
_create_unique_pmf_any: Any = create_unique_pmf
_quantile_any: Any = quantile
_percentile_any: Any = percentile


@njit
def _quantile_scalar(states: FloatArray, pmf: FloatArray, rank: float) -> float:
    return _quantile_any(states, pmf, rank)


@njit
def _quantile_array(
    states: FloatArray,
    pmf: FloatArray,
    rank: FloatArray,
) -> FloatArray:
    return _quantile_any(states, pmf, rank)


@njit
def _quantile_array_linear(
    states: FloatArray,
    pmf: FloatArray,
    rank: FloatArray,
) -> FloatArray:
    return _quantile_any(states, pmf, rank, interpolation='linear')


@njit
def _percentile_scalar(states: FloatArray, pmf: FloatArray, rank: float) -> float:
    return _percentile_any(states, pmf, rank)


@njit
def _percentile_array(
    states: FloatArray,
    pmf: FloatArray,
    rank: FloatArray,
) -> FloatArray:
    return _percentile_any(states, pmf, rank)


def test_compiled_quantile_scalar_and_array_paths() -> None:
    """Compiled quantile calls match scalar and array Python results."""
    states = np.array([1.0, 2.0, 3.0])
    pmf = np.array([0.2, 0.3, 0.5])
    ranks = np.array([0.0, 0.2, 0.2001, 0.5, 1.0])

    assert _quantile_scalar(states, pmf, 0.5) == pytest.approx(
        quantile(states, pmf, 0.5)
    )
    np.testing.assert_allclose(
        _quantile_array(states, pmf, ranks),
        quantile(states, pmf, ranks),
    )

    assert _quantile_scalar.nopython_signatures
    assert _quantile_array.nopython_signatures


def test_compiled_quantile_linear_and_bin_edge_paths() -> None:
    """Compiled quantiles support explicit linear interpolation and bin edges."""
    states = np.array([0.0, 10.0])
    pmf = np.array([0.25, 0.75])
    ranks = np.array([0.25, 0.625, 1.0])

    np.testing.assert_allclose(
        _quantile_array_linear(states, pmf, ranks),
        quantile(states, pmf, ranks, interpolation='linear'),
    )

    edges = np.array([0.0, 1.0, 3.0])
    bin_pmf = np.array([0.25, 0.75])
    bin_ranks = np.array([0.0, 0.125, 0.25, 0.625, 1.0])
    np.testing.assert_allclose(
        _quantile_array(edges, bin_pmf, bin_ranks),
        quantile(edges, bin_pmf, bin_ranks),
    )

    assert _quantile_array_linear.nopython_signatures
    assert _quantile_array.nopython_signatures


def test_compiled_percentile_scalar_and_array_paths() -> None:
    """Compiled percentile calls match scalar and array Python results."""
    states = np.array([1.0, 2.0, 3.0])
    pmf = np.array([0.2, 0.3, 0.5])
    ranks = np.array([0.0, 20.0, 50.0, 100.0])

    assert _percentile_scalar(states, pmf, 50.0) == pytest.approx(
        percentile(states, pmf, 50.0)
    )
    np.testing.assert_allclose(
        _percentile_array(states, pmf, ranks),
        percentile(states, pmf, ranks),
    )

    assert _percentile_scalar.nopython_signatures
    assert _percentile_array.nopython_signatures


def test_direct_jitted_statistical_kernels() -> None:
    """Directly decorated kernels compile and return expected values."""
    states = np.array([0.0, 1.0])
    pmf = np.array([0.5, 0.5])
    assert gini(states, pmf) == pytest.approx(0.5)

    states_unique, pmf_unique = create_unique_pmf(
        np.array([2.0, 1.0, 2.0, 1.0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
    )
    np.testing.assert_array_equal(states_unique, [1.0, 2.0])
    np.testing.assert_allclose(pmf_unique, [0.6, 0.4])

    assert _gini_any.nopython_signatures
    assert _create_unique_pmf_any.nopython_signatures
