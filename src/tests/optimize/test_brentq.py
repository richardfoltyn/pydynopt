"""Test SciPy-compatible Brent errors in Python and compiled callers."""

from typing import Any

from numba import njit
import numpy as np
import pytest
from scipy.optimize import brentq

from pydynopt.optimize._zeros_scipy import (
    _ECONVERR,
    RootResult,
    _brentq_full,
    _brentq_impl,
    _brentq_simple,
)

_RTOL = 4.0 * np.finfo(float).eps


def _objective(x: float) -> float:
    return x * x - 2.0


def _positive(x: float) -> float:
    return x * x + 1.0


def _nan_at_zero(x: float) -> float:
    if x == 0.0:
        return np.nan
    return x - 1.0


@njit
def _objective_jit(x: float) -> float:
    return x * x - 2.0


@njit
def _positive_jit(x: float) -> float:
    return x * x + 1.0


@njit
def _nan_at_zero_jit(x: float) -> float:
    if x == 0.0:
        return np.nan
    return x - 1.0


@njit
def _brentq_options(
    a: float,
    b: float,
    xtol: float,
    rtol: float,
    maxiter: int,
    disp: bool,
) -> float:
    return brentq(
        _objective_jit,
        a,
        b,
        xtol=xtol,
        rtol=rtol,
        maxiter=maxiter,
        disp=disp,
    )


@njit
def _brentq_explicit_false(disp: bool) -> float:
    return brentq(
        _objective_jit,
        0.0,
        2.0,
        maxiter=0,
        full_output=False,
        disp=disp,
    )


@njit
def _brentq_full_output(disp: bool) -> tuple[float, RootResult]:
    return brentq(
        _objective_jit,
        0.0,
        2.0,
        maxiter=0,
        full_output=True,
        disp=disp,
    )


@njit
def _brentq_positive() -> float:
    return brentq(_positive_jit, 0.0, 2.0)


@njit
def _brentq_nan() -> float:
    return brentq(_nan_at_zero_jit, 0.0, 2.0)


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'xtol': 0.0}, 'xtol too small'),
        ({'rtol': 0.0}, 'rtol too small'),
        ({'maxiter': -1}, 'maxiter must be >= 0'),
    ],
)
def test_brentq_impl_validates_parameters(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _brentq_impl(_objective, 0.0, 2.0, **kwargs)


def test_brentq_impl_rejects_invalid_function_values() -> None:
    with pytest.raises(ValueError, match='different signs'):
        _brentq_impl(_positive, 0.0, 2.0)

    with pytest.raises(ValueError, match='NaN'):
        _brentq_impl(_nan_at_zero, 0.0, 2.0)


def test_brentq_impl_reports_nonconvergence() -> None:
    root, fx, converged, flag, iterations, function_calls = _brentq_impl(
        _objective, 0.0, 2.0, maxiter=0
    )

    assert root == 2.0
    assert fx == 2.0
    assert not converged
    assert flag == _ECONVERR
    assert iterations == 0
    assert function_calls == 2


def test_brentq_helpers_honor_disp() -> None:
    assert _brentq_simple(_objective, 0.0, 2.0, maxiter=0, disp=False) == 2.0

    root, result = _brentq_full(
        _objective, 0.0, 2.0, maxiter=0, full_output=True, disp=False
    )
    assert root == 2.0
    assert result.root == root
    assert result.fx == 2.0
    assert not result.converged
    assert result.flag == 'convergence error'
    assert result.iterations == 0
    assert result.function_calls == 2

    with pytest.raises(RuntimeError, match='Failed to converge'):
        _brentq_simple(_objective, 0.0, 2.0, maxiter=0, disp=True)
    with pytest.raises(RuntimeError, match='Failed to converge'):
        _brentq_full(_objective, 0.0, 2.0, maxiter=0, disp=True)


def test_compiled_brentq_validates_parameters() -> None:
    with pytest.raises(ValueError, match='xtol too small'):
        _brentq_options(0.0, 2.0, 0.0, _RTOL, 100, True)
    with pytest.raises(ValueError, match='rtol too small'):
        _brentq_options(0.0, 2.0, 2.0e-12, 0.0, 100, True)
    with pytest.raises(ValueError, match='maxiter must be >= 0'):
        _brentq_options(0.0, 2.0, 2.0e-12, _RTOL, -1, True)


def test_compiled_brentq_rejects_invalid_function_values() -> None:
    with pytest.raises(ValueError, match='different signs'):
        _brentq_positive()
    with pytest.raises(ValueError, match='NaN'):
        _brentq_nan()


def test_compiled_brentq_honors_disp() -> None:
    assert _brentq_options(0.0, 2.0, 2.0e-12, _RTOL, 0, False) == 2.0

    with pytest.raises(RuntimeError, match='Failed to converge'):
        _brentq_options(0.0, 2.0, 2.0e-12, _RTOL, 0, True)


def test_compiled_brentq_selects_explicit_full_output() -> None:
    assert _brentq_explicit_false(False) == 2.0

    root, result = _brentq_full_output(False)
    assert root == 2.0
    assert result.root == root
    assert result.fx == 2.0
    assert not result.converged
    assert result.flag == 'convergence error'
    assert result.iterations == 0
    assert result.function_calls == 2
