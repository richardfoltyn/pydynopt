"""Test optimization results and numerical differentiation."""

from typing import Any

import numpy as np
import pytest

from pydynopt.optimize import OptimResult
from pydynopt.optimize.common import nderiv


def _quadratic(x: float, scale: float = 1.0) -> float:
    return scale * x * x


def _array_quadratic(x: np.ndarray, scale: float = 1.0) -> float:
    return float(scale * np.sum(x * x))


def test_optim_result_defaults_and_repr() -> None:
    result = OptimResult()

    assert result.x == 0.0
    assert result.fx == 0.0
    assert result.iterations == 0
    assert result.function_calls == 0
    assert not result.converged
    assert result.flag == ''

    text = repr(result)
    assert 'converged: False' in text
    assert 'flag:' in text
    assert 'x: 0' in text
    assert 'fx: 0' in text


def test_nderiv_scalar_inputs_and_arguments() -> None:
    assert nderiv(_quadratic, 2) == pytest.approx(4.0)
    assert nderiv(_quadratic, np.int64(2)) == pytest.approx(4.0)
    assert nderiv(_quadratic, np.float32(2.0)) == pytest.approx(4.0)
    assert nderiv(_quadratic, 2.0, np.nan, 1.0e-8, 3.0) == pytest.approx(12.0)


def test_nderiv_uses_supplied_function_value() -> None:
    calls = 0

    def objective(x: float) -> float:
        nonlocal calls
        calls += 1
        return x * x

    derivative = nderiv(objective, 2.0, fx=4.0)

    assert derivative == pytest.approx(4.0)
    assert calls == 1


@pytest.mark.parametrize(
    'point',
    [
        [1.0, 2.0],
        (1.0, 2.0),
        np.array([1.0, 2.0]),
        np.array([1, 2], dtype=np.int64),
    ],
)
def test_nderiv_array_like_inputs(point: Any) -> None:
    derivative = nderiv(_array_quadratic, point)

    assert derivative.dtype == np.float64
    assert derivative == pytest.approx([2.0, 4.0])


def test_nderiv_array_arguments_and_signed_step() -> None:
    point = np.array([1.0, 2.0])

    forward = nderiv(_array_quadratic, point, np.nan, 1.0e-8, 3.0)
    backward = nderiv(_array_quadratic, point, np.nan, -1.0e-8, 3.0)

    assert forward == pytest.approx([6.0, 12.0])
    assert backward == pytest.approx([6.0, 12.0])


@pytest.mark.parametrize('eps', [0.0, np.inf, -np.inf, np.nan])
def test_nderiv_rejects_invalid_steps(eps: float) -> None:
    with pytest.raises(ValueError):
        nderiv(_quadratic, 2.0, eps=eps)


@pytest.mark.parametrize(
    'point',
    [
        True,
        np.array([[1.0, 2.0]]),
        np.array([True, False]),
        np.array([1.0 + 1.0j]),
        ['a', 'b'],
    ],
)
def test_nderiv_rejects_invalid_points(point: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        nderiv(_array_quadratic, point)


def test_nderiv_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match='finite'):
        nderiv(_quadratic, np.inf)
    with pytest.raises(ValueError, match='finite'):
        nderiv(_array_quadratic, [1.0, np.nan])
    with pytest.raises(ValueError, match='func result'):
        nderiv(lambda x: np.nan, 1.0)
