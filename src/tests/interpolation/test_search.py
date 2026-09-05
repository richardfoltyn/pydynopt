"""Test interpolation binary-search entries and kernels."""

from numba import njit
import numpy as np
import pytest

from pydynopt.interpolate.numba.search import bsearch, bsearch_impl


@njit
def _jitted_bsearch(x: float, xp: np.ndarray, ilb: int) -> int:
    return bsearch(x, xp, ilb)


@njit
def _jitted_bsearch_impl(x: float, xp: np.ndarray, ilb: int) -> int:
    return bsearch_impl(x, xp, ilb)


@pytest.mark.parametrize(
    ('x', 'expected'),
    [(-2.0, 0), (0.0, 0), (0.5, 0), (1.0, 1), (3.0, 2), (8.0, 2)],
)
def test_bsearch_boundaries_and_knots(x: float, expected: int) -> None:
    xp = np.array([0.0, 1.0, 1.5, 3.0])
    for ilb in range(xp.size - 1):
        assert bsearch(x, xp, ilb) == expected
        assert bsearch_impl(x, xp, ilb) == expected
        assert _jitted_bsearch(x, xp, ilb) == expected
        assert _jitted_bsearch_impl(x, xp, ilb) == expected


@pytest.mark.parametrize(
    'xp',
    [
        np.array([]),
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_bsearch_rejects_invalid_grids(xp: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bsearch(0.5, xp)


def test_bsearch_two_point_grid_and_clamped_guess() -> None:
    xp = np.array([1.0, 4.0])
    assert bsearch(-1.0, xp, -10) == 0
    assert bsearch(5.0, xp, 10) == 0
    assert _jitted_bsearch(2.0, xp, 10) == 0
    assert _jitted_bsearch.nopython_signatures
