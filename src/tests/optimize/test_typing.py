"""Exercise public optimize annotations as a package consumer."""

from typing import Any, assert_type

import numpy as np
from numpy.typing import NDArray

from pydynopt.optimize import OptimResult, RootResult, brentq, newton_bisect
from pydynopt.optimize.common import nderiv


def _objective(x: float, *args: Any) -> float:
    return x


def _array_objective(x: np.ndarray, *args: Any) -> float:
    return float(np.sum(x))


def _objective_jac(x: float, *args: Any) -> tuple[float, float]:
    return x, 1.0


def _jacobian(x: float, *args: Any) -> float:
    return 1.0


def test_consumer_return_types() -> None:
    point = np.array([1.0, 2.0])
    dynamic = bool(point[0])

    assert_type(nderiv(_objective, 1.0), float)
    assert_type(nderiv(_objective, np.int64(1)), float)
    assert_type(nderiv(_array_objective, [1.0, 2.0]), NDArray[np.float64])
    assert_type(nderiv(_array_objective, point), NDArray[np.float64])

    assert_type(newton_bisect(_objective, 1.0), tuple[float, float])
    assert_type(
        newton_bisect(_objective, 1.0, full_output=False),
        tuple[float, float],
    )
    assert_type(
        newton_bisect(_objective_jac, 1.0, jac=True),
        tuple[float, float],
    )
    assert_type(
        newton_bisect(_objective, 1.0, jac=_jacobian),
        tuple[float, float],
    )
    assert_type(
        newton_bisect(_objective_jac, 1.0, jac=True, full_output=True),
        tuple[float, RootResult],
    )
    assert_type(
        newton_bisect(_objective, 1.0, full_output=dynamic),
        tuple[float, float] | tuple[float, RootResult],
    )

    assert_type(brentq(_objective, -1.0, 1.0), float)
    assert_type(brentq(_objective, -1.0, 1.0, full_output=False), float)
    assert_type(
        brentq(_objective, -1.0, 1.0, full_output=True),
        tuple[float, RootResult],
    )
    assert_type(
        brentq(_objective, -1.0, 1.0, full_output=dynamic),
        float | tuple[float, RootResult],
    )


def test_result_field_types() -> None:
    root_result = RootResult()
    assert_type(root_result.root, float)
    assert_type(root_result.fx, float)
    assert_type(root_result.iterations, int)
    assert_type(root_result.function_calls, int)
    assert_type(root_result.converged, bool)
    assert_type(root_result.flag, str)

    optim_result = OptimResult()
    assert_type(optim_result.x, float)
    assert_type(optim_result.fx, float)
    assert_type(optim_result.iterations, int)
    assert_type(optim_result.function_calls, int)
    assert_type(optim_result.converged, bool)
    assert_type(optim_result.flag, str)


def test_ebl_newton_pattern() -> None:
    args: tuple[Any, ...] = ()
    stocks, fx = assert_type(
        newton_bisect(
            _objective_jac,
            1.0,
            0.0,
            2.0,
            maxiter=20,
            jac=True,
            tol=1.0e-8,
            args=args,
        ),
        tuple[float, float],
    )
    assert_type(stocks, float)
    assert_type(fx, float)
