"""Provide optimization result containers and numerical differentiation.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Sequence
from typing import Any, overload as typing_overload

import numpy as np
from numpy.typing import NDArray

from pydynopt.numba import JIT_OPTIONS, overload as numba_overload

from .numba.common import nderiv_array, nderiv_scalar

__all__ = ['OptimResult', 'nderiv']

type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type RealArrayInput = Sequence[RealScalar] | np.ndarray
type FloatArray = NDArray[np.float64]
type ScalarFunc = Callable[..., RealScalar]
type ValueJacFunc = Callable[..., tuple[RealScalar, RealScalar]]


class OptimResult:
    """Store the result of an optimization operation.

    Attributes
    ----------
    x
        Optimizer or root estimate.
    fx
        Objective value at ``x``.
    iterations
        Number of iterations performed.
    function_calls
        Number of objective evaluations.
    converged
        Whether the routine converged.
    flag
        Description of the termination status.
    """

    x: float
    fx: float
    iterations: int
    function_calls: int
    converged: bool
    flag: str

    def __init__(self) -> None:
        self.x = 0.0
        self.fx = 0.0
        self.iterations = 0
        self.function_calls = 0
        self.converged = False
        self.flag = ''

    def __repr__(self) -> str:
        attrs = ['converged', 'flag', 'function_calls', 'iterations', 'x', 'fx']
        tokens: list[str] = []

        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, (bool, np.bool_)):
                formatted = str(bool(value))
            elif isinstance(value, (float, np.floating)):
                formatted = f'{float(value):g}'
            elif isinstance(value, (int, np.integer)):
                formatted = f'{int(value):d}'
            elif isinstance(value, np.ndarray):
                values = ', '.join(f'{float(item):g}' for item in np.ravel(value))
                formatted = f'[{values}]'
            else:
                formatted = str(value)
            tokens.append(f'{attr:>20s}: {formatted}')

        return '\n'.join(tokens)


def _normalize_real_scalar(value: object, name: str, *, finite: bool = True) -> float:
    """Normalize a real scalar to a Python ``float``."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'iuf':
        msg = f'{name} must be a real scalar'
        raise TypeError(msg)

    result = float(array.item())
    if finite and not np.isfinite(result):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    return result


def _normalize_integer(value: object, name: str, minimum: int) -> int:
    """Normalize a non-boolean integer and enforce its lower bound."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'iu':
        msg = f'{name} must be an integer'
        raise TypeError(msg)

    result = int(array.item())
    if result < minimum:
        msg = f'{name} must be at least {minimum}'
        raise ValueError(msg)
    return result


def _normalize_bool(value: object, name: str) -> bool:
    """Normalize a Python or NumPy boolean scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind != 'b':
        msg = f'{name} must be a boolean'
        raise TypeError(msg)
    return bool(array.item())


@typing_overload
def nderiv(
    func: ScalarFunc,
    x: RealScalar,
    fx: RealScalar = np.nan,
    eps: RealScalar = 1.0e-8,
    *args: Any,
) -> float: ...


@typing_overload
def nderiv(
    func: ScalarFunc,
    x: RealArrayInput,
    fx: RealScalar = np.nan,
    eps: RealScalar = 1.0e-8,
    *args: Any,
) -> FloatArray: ...


def nderiv(
    func: ScalarFunc,
    x: RealScalar | RealArrayInput,
    fx: RealScalar = np.nan,
    eps: RealScalar = 1.0e-8,
    *args: Any,
) -> float | FloatArray:
    """Numerically forward-differentiate a function at a given point.

    Parameters
    ----------
    func
        Function returning a real scalar.
    x
        Scalar or one-dimensional array-like point.
    fx
        Function value at ``x``. A NaN value requests evaluation of ``func``.
    eps
        Signed finite-difference step.
    *args
        Additional arguments passed to ``func``.

    Returns
    -------
    Scalar derivative or one-dimensional gradient.

    Raises
    ------
    TypeError
        If an input is not real-valued or ``x`` is not scalar or array-like.
    ValueError
        If an input has an invalid dimension or contains a non-finite value, or
        if ``eps`` is zero.
    """
    step = _normalize_real_scalar(eps, 'eps')
    if step == 0.0:
        raise ValueError('eps must be nonzero')

    array = np.asarray(x)
    if array.ndim == 0:
        point = _normalize_real_scalar(x, 'x')
        value = _normalize_real_scalar(fx, 'fx', finite=False)
        if np.isnan(value):
            value = _normalize_real_scalar(func(point, *args), 'func result')
        return nderiv_scalar(func, point, value, step, *args)

    if array.ndim != 1:
        raise ValueError('x must be one-dimensional')
    if array.dtype.kind not in 'iuf':
        raise TypeError('x must contain real values')

    point_array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(point_array)):
        raise ValueError('x must contain only finite values')

    value = _normalize_real_scalar(fx, 'fx', finite=False)
    if np.isnan(value):
        value = _normalize_real_scalar(func(point_array, *args), 'func result')
    return nderiv_array(func, point_array, value, step, *args)


@numba_overload(nderiv, jit_options=JIT_OPTIONS)
def _overload_nderiv(
    func: Any,
    x: Any,
    fx: Any = np.nan,
    eps: Any = 1.0e-8,
    *args: Any,
) -> Any:
    """Select the scalar or array numerical-derivative kernel for Numba."""
    from numba import types

    if x in types.integer_domain or x in types.real_domain:
        return nderiv_scalar
    if (
        isinstance(x, types.Array)
        and x.ndim == 1
        and (x.dtype in types.integer_domain or x.dtype in types.real_domain)
    ):
        return nderiv_array
    return None
