"""Test optimization utilities from actual Numba-compiled callers."""

from collections.abc import Callable

from numba import njit
import numpy as np
import pytest

from pydynopt.optimize import RootResult, brentq, newton_bisect
from pydynopt.optimize.common import nderiv


@njit
def _quadratic(x: float) -> float:
    return x * x - 2.0


@njit
def _quadratic_combined(x: float) -> tuple[float, float]:
    return x * x - 2.0, 2.0 * x


@njit
def _quadratic_jac(x: float) -> float:
    return 2.0 * x


@njit
def _array_quadratic(x: np.ndarray) -> float:
    return np.sum(x * x)


@njit
def _shifted(x: float, root: float) -> float:
    return x - root


@njit
def _shifted_jac(x: float, root: float) -> float:
    return 1.0


@njit
def _endpoint_combined(x: float) -> tuple[float, float]:
    return x - 1.0, np.nan


@njit
def _nderiv_scalar_call() -> float:
    return nderiv(_quadratic, 2)


@njit
def _nderiv_array_call() -> np.ndarray:
    return nderiv(_array_quadratic, np.array([1, 2], dtype=np.int64))


@njit
def _newton_numerical() -> tuple[float, float]:
    return newton_bisect(_quadratic, 1, 0, 2)


@njit
def _newton_explicit_false() -> tuple[float, float]:
    return newton_bisect(_quadratic, 1, 0, 2, full_output=False)


@njit
def _newton_combined() -> tuple[float, float]:
    return newton_bisect(_quadratic_combined, 1, 0, 2, jac=True)


@njit
def _newton_callable() -> tuple[float, float]:
    return newton_bisect(_quadratic, 1, 0, 2, jac=_quadratic_jac)


@njit
def _newton_args() -> tuple[float, float]:
    return newton_bisect(_shifted, 0.0, args=(3.0,), jac=_shifted_jac)


@njit
def _newton_endpoint() -> tuple[float, float]:
    return newton_bisect(_endpoint_combined, 1.5, 1.0, 2.0, jac=True)


@njit
def _newton_full() -> tuple[float, RootResult]:
    return newton_bisect(
        _quadratic_combined,
        1,
        0,
        2,
        jac=True,
        full_output=True,
    )


@njit
def _newton_maxiter() -> tuple[float, RootResult]:
    return newton_bisect(
        _quadratic_combined,
        1.0,
        jac=True,
        maxiter=1,
        full_output=True,
    )


@njit
def _newton_invalid_bound() -> tuple[float, float]:
    return newton_bisect(_quadratic, 3.0, 0.0, 2.0)


@njit
def _brent_simple() -> float:
    return brentq(_quadratic, 0, 2)


@njit
def _brent_explicit_false() -> float:
    return brentq(_quadratic, 0, 2, full_output=False)


@njit
def _brent_full() -> tuple[float, RootResult]:
    return brentq(_quadratic, 0, 2, full_output=True)


def test_compiled_nderiv_scalar_and_array() -> None:
    assert _nderiv_scalar_call() == pytest.approx(4.0)
    derivative = _nderiv_array_call()
    assert derivative.dtype == np.float64
    assert derivative == pytest.approx([2.0, 4.0])


@pytest.mark.parametrize(
    'root_call',
    [
        _newton_numerical,
        _newton_explicit_false,
        _newton_combined,
        _newton_callable,
    ],
)
def test_compiled_newton_derivative_and_output_modes(
    root_call: Callable[[], tuple[float, float]],
) -> None:
    root, fx = root_call()
    assert root == pytest.approx(np.sqrt(2.0))
    assert fx == pytest.approx(0.0, abs=1.0e-8)


def test_compiled_newton_arguments_and_endpoint() -> None:
    assert _newton_args() == pytest.approx((3.0, 0.0))
    assert _newton_endpoint() == (1.0, 0.0)


def test_compiled_newton_full_output() -> None:
    root, result = _newton_full()

    assert result.root == root
    assert result.fx == pytest.approx(0.0, abs=1.0e-8)
    assert result.converged
    assert result.flag == 'converged'

    _, result = _newton_maxiter()
    assert not result.converged
    assert result.flag == 'maximum iterations exceeded'
    assert result.iterations == 1


def test_compiled_newton_validates_bounds() -> None:
    with pytest.raises(ValueError, match='within'):
        _newton_invalid_bound()


def test_compiled_package_brent_output_modes() -> None:
    expected = np.sqrt(2.0)
    assert _brent_simple() == pytest.approx(expected)
    assert _brent_explicit_false() == pytest.approx(expected)

    root, result = _brent_full()
    assert root == pytest.approx(expected)
    assert result.root == root
    assert result.converged
    assert result.flag == 'converged'
