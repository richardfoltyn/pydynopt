"""Test checked Brent and Newton-bisection root finders."""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import brentq as scipy_brentq

from pydynopt import optimize
from pydynopt.optimize import RootResult, brentq, newton_bisect


def _quadratic(x: float) -> float:
    return x * x - 2.0


def _quadratic_combined(x: float) -> tuple[float, float]:
    return x * x - 2.0, 2.0 * x


def _quadratic_jac(x: float) -> float:
    return 2.0 * x


def _shifted(x: float, root: float) -> float:
    return x - root


def _shifted_jac(x: float, root: float) -> float:
    return 1.0


def test_package_exports_are_explicit() -> None:
    assert optimize.__all__ == ['OptimResult', 'RootResult', 'brentq', 'newton_bisect']


def test_brentq_matches_scipy() -> None:
    expected = scipy_brentq(_quadratic, 0.0, 2.0)

    root = brentq(_quadratic, 0, 2)
    assert root == pytest.approx(expected)
    assert isinstance(root, float)


def test_brentq_full_output_uses_public_result() -> None:
    root, result = brentq(_quadratic, 0.0, 2.0, full_output=True)

    assert isinstance(result, RootResult)
    assert result.root == root
    assert result.fx == pytest.approx(0.0, abs=1.0e-12)
    assert result.converged
    assert result.flag == 'converged'
    assert result.iterations > 0
    assert result.function_calls >= result.iterations


def test_brentq_disp_false_returns_nonconverged_result() -> None:
    root, result = brentq(_quadratic, 0.0, 2.0, maxiter=0, full_output=True, disp=False)

    assert root == 2.0
    assert not result.converged
    assert result.flag == 'convergence error'

    with pytest.raises(RuntimeError, match='Failed to converge'):
        brentq(_quadratic, 0.0, 2.0, maxiter=0)


@pytest.mark.parametrize(
    ('func', 'jac'),
    [
        (_quadratic, False),
        (_quadratic_combined, True),
        (_quadratic, _quadratic_jac),
    ],
)
def test_newton_bisect_derivative_modes(func: Any, jac: Any) -> None:
    root, fx = newton_bisect(func, 1, 0, 2, jac=jac)

    assert root == pytest.approx(np.sqrt(2.0))
    assert fx == pytest.approx(0.0, abs=1.0e-8)
    assert isinstance(root, float)
    assert isinstance(fx, float)


def test_newton_bisect_forwards_arguments() -> None:
    root, fx = newton_bisect(
        _shifted,
        0.0,
        args=(3.0,),
        jac=_shifted_jac,
    )

    assert root == pytest.approx(3.0)
    assert fx == pytest.approx(0.0)


def test_newton_bisect_handles_brackets_and_endpoints() -> None:
    root, fx = newton_bisect(_quadratic, 1.0, 2.0, 0.0)
    assert root == pytest.approx(np.sqrt(2.0))
    assert fx == pytest.approx(0.0, abs=1.0e-8)

    assert newton_bisect(_shifted, 1.0, args=(1.0,)) == (1.0, 0.0)
    assert newton_bisect(_shifted, 1.5, 1.0, 2.0, args=(1.0,)) == (1.0, 0.0)
    assert newton_bisect(_shifted, 1.5, 0.0, 2.0, args=(2.0,)) == (2.0, 0.0)


def test_newton_bisect_returns_before_endpoint_derivative() -> None:
    def combined(x: float) -> tuple[float, float]:
        return x - 1.0, np.nan

    root, fx = newton_bisect(combined, 1.5, 1.0, 2.0, jac=True)

    assert root == 1.0
    assert fx == 0.0


def test_newton_bisect_full_output() -> None:
    root, result = newton_bisect(
        _quadratic_combined,
        1.0,
        0.0,
        2.0,
        jac=True,
        full_output=True,
    )

    assert isinstance(result, RootResult)
    assert result.root == root
    assert result.fx == pytest.approx(0.0, abs=1.0e-8)
    assert result.converged
    assert result.flag == 'converged'
    assert result.iterations > 0
    assert result.function_calls >= result.iterations


def test_newton_bisect_failure_results() -> None:
    def zero_derivative(x: float) -> tuple[float, float]:
        return x + 1.0, 0.0

    _, result = newton_bisect(zero_derivative, 0.0, jac=True, full_output=True)
    assert not result.converged
    assert result.flag == 'value error'

    _, result = newton_bisect(
        _quadratic_combined, 1.0, jac=True, maxiter=1, full_output=True
    )
    assert not result.converged
    assert result.flag == 'maximum iterations exceeded'
    assert result.iterations == 1


@pytest.mark.parametrize(
    'kwargs',
    [
        {'eps': 0.0},
        {'eps': np.inf},
        {'xtol': -1.0},
        {'xtol': np.inf},
        {'tol': -1.0},
        {'tol': np.inf},
        {'maxiter': 0},
    ],
)
def test_newton_bisect_rejects_invalid_parameters(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        newton_bisect(_quadratic, 1.0, **kwargs)


def test_newton_bisect_rejects_invalid_bounds_and_values() -> None:
    with pytest.raises(ValueError, match='within'):
        newton_bisect(_quadratic, 3.0, 0.0, 2.0)
    with pytest.raises(ValueError, match='bracket'):
        newton_bisect(lambda x: x * x + 1.0, 1.0, 0.0, 2.0)
    with pytest.raises(ValueError, match='finite residual'):
        newton_bisect(lambda x: np.nan, 1.0)
    with pytest.raises(ValueError, match='derivative must be finite'):
        newton_bisect(_quadratic, 1.0, jac=lambda x: np.inf)
