"""Provide Numba-compatible numerical differentiation kernels.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pydynopt.numba import JIT_OPTIONS, overload as numba_overload, register_jitable

__all__ = ['nderiv_array', 'nderiv_scalar']

type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type FloatArray = NDArray[np.float64]
type ScalarFunc = Callable[..., Any]


def _extract_arg_identity(arg: Any, index: int = 0) -> Any:
    """Return a scalar or array function result unchanged."""
    return arg


def _extract_arg_by_index(arg: Any, index: int = 0) -> Any:
    """Extract one element from a compound function result."""
    return arg[index]


def _extract_arg(arg: Any, index: int = 0) -> Any:
    """Return a scalar result or one element from a compound result."""
    if isinstance(arg, tuple):
        return arg[index]
    return arg


@numba_overload(_extract_arg, jit_options=JIT_OPTIONS)
def _overload_extract_arg(arg: Any, index: Any = 0) -> Any:
    """Select result extraction for a Numba function-result type."""
    from numba import types

    if isinstance(arg, (types.Number, types.Array)):
        return _extract_arg_identity
    return _extract_arg_by_index


@register_jitable(**JIT_OPTIONS)
def nderiv_scalar(
    func: ScalarFunc,
    x: RealScalar,
    fx: RealScalar = np.nan,
    eps: RealScalar = 1.0e-8,
    *args: Any,
) -> float:
    """Numerically differentiate a scalar function argument.

    Parameters
    ----------
    func
        Function returning a real scalar or a compound result whose first
        element is a real scalar.
    x
        Point at which to evaluate the derivative.
    fx
        Function value at ``x``. A NaN value requests evaluation of ``func``.
    eps
        Signed finite-difference step.
    *args
        Additional arguments passed to ``func``.

    Returns
    -------
    Forward-difference derivative.

    Raises
    ------
    ValueError
        If ``x`` or ``eps`` is invalid or ``func`` returns a non-finite value.
    """
    if not np.isfinite(x):
        raise ValueError('x must be finite')
    if not np.isfinite(eps) or eps == 0.0:
        raise ValueError('eps must be finite and nonzero')

    value = fx
    if np.isnan(value):
        value = _extract_arg(func(x, *args), 0)
    if not np.isfinite(value):
        raise ValueError('func must return a finite real scalar')

    next_value = _extract_arg(func(x + eps, *args), 0)
    if not np.isfinite(next_value):
        raise ValueError('func must return a finite real scalar')

    derivative = (next_value - value) / eps
    return float(derivative)


@register_jitable(**JIT_OPTIONS)
def nderiv_array(
    func: ScalarFunc,
    x: np.ndarray,
    fx: RealScalar = np.nan,
    eps: RealScalar = 1.0e-8,
    *args: Any,
) -> FloatArray:
    """Numerically differentiate a function of a one-dimensional array.

    Parameters
    ----------
    func
        Function returning a real scalar or a compound result whose first
        element is a real scalar.
    x
        One-dimensional point at which to evaluate the gradient.
    fx
        Function value at ``x``. A NaN value requests evaluation of ``func``.
    eps
        Signed finite-difference step applied to each coordinate.
    *args
        Additional arguments passed to ``func``.

    Returns
    -------
    One-dimensional forward-difference gradient.

    Raises
    ------
    ValueError
        If ``x`` or ``eps`` is invalid or ``func`` returns a non-finite value.
    """
    if x.ndim != 1:
        raise ValueError('x must be one-dimensional')
    if not np.isfinite(eps) or eps == 0.0:
        raise ValueError('eps must be finite and nonzero')

    work = np.empty(x.shape, dtype=np.float64)
    for i in range(x.size):
        if not np.isfinite(x[i]):
            raise ValueError('x must contain only finite values')
        work[i] = x[i]

    value = fx
    if np.isnan(value):
        value = _extract_arg(func(work, *args), 0)
    if not np.isfinite(value):
        raise ValueError('func must return a finite real scalar')

    derivative = np.empty(x.shape, dtype=np.float64)
    for i in range(work.size):
        xi = work[i]
        work[i] = xi + eps
        next_value = _extract_arg(func(work, *args), 0)
        work[i] = xi
        if not np.isfinite(next_value):
            raise ValueError('func must return a finite real scalar')
        derivative[i] = (next_value - value) / eps

    return derivative
