"""Provide Numba-compatible Newton-bisection root-finding kernels.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from pydynopt.numba import JIT_OPTIONS, register_jitable

from .._zeros_scipy import (
    _ECONVERGED,
    _EMAXITER,
    _EVALUEERR,
    RootResult,
    _status_message,
)
from .common import nderiv_scalar

__all__ = [
    'newton_bisect_callable_full',
    'newton_bisect_callable_simple',
    'newton_bisect_full',
    'newton_bisect_impl',
    'newton_bisect_simple',
]

type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type ScalarFunc = Callable[..., RealScalar]
type ValueJacFunc = Callable[..., tuple[RealScalar, RealScalar]]
type JacFunc = Callable[..., RealScalar]
type RootKernelResult = tuple[float, float, bool, int, int, int]


@register_jitable(**JIT_OPTIONS)
def newton_bisect_impl(
    func: ScalarFunc | ValueJacFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: bool = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    jac_func: JacFunc | None = None,
) -> RootKernelResult:
    """Find a scalar root using Newton steps constrained by bisection.

    Parameters
    ----------
    func
        Scalar residual function. If ``jac`` is true, the function must return
        the residual and derivative.
    x0
        Initial root estimate.
    a, b
        Optional lower and upper bounds. Reversed bounds are normalized.
    args
        Additional arguments passed to ``func`` and ``jac_func``.
    jac
        Whether ``func`` returns the residual and derivative together.
    eps
        Positive finite-difference step used for a numerical derivative.
    xtol
        Absolute tolerance for changes in the root estimate.
    tol
        Absolute tolerance for the residual.
    maxiter
        Maximum number of iterations.
    jac_func
        Optional function that evaluates the derivative separately.

    Returns
    -------
    root
        Estimated root location.
    fx
        Residual at ``root``.
    converged
        Whether the routine converged.
    flag
        Internal termination status.
    iterations
        Number of iterations performed.
    function_calls
        Number of residual evaluations, including numerical-difference calls.

    Raises
    ------
    ValueError
        If an option, bound, residual, or derivative is invalid, or a supplied
        two-sided bracket does not contain a sign change.
    """
    if not np.isfinite(xtol) or xtol < 0.0:
        raise ValueError('xtol must be finite and non-negative')
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError('tol must be finite and non-negative')
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError('eps must be finite and positive')
    if maxiter < 1:
        raise ValueError('maxiter must be positive')

    x = float(x0)
    if not np.isfinite(x):
        raise ValueError('x0 must be finite')

    xa = -np.inf if a is None else float(a)
    xb = np.inf if b is None else float(b)
    if np.isnan(xa) or np.isnan(xb):
        raise ValueError('bounds must not be NaN')
    if xa > xb:
        xa, xb = xb, xa
    if x < xa or x > xb:
        raise ValueError('x0 must lie within the supplied bounds')

    values = np.empty(2, dtype=np.float64)
    values[:] = func(x, *args)
    fx = float(values[0])
    nfev = 1
    if not np.isfinite(fx):
        raise ValueError('func must return a finite residual')
    if abs(fx) <= tol:
        return x, fx, True, _ECONVERGED, 0, nfev
    initial_fpx = float(values[1])

    fa = fx
    fb = fx
    slb = np.sign(fx)
    sub = np.sign(fx)
    xlb = x
    xub = x

    if np.isfinite(xa):
        if xa == x:
            fa = fx
        else:
            values[:] = func(xa, *args)
            fa = float(values[0])
            nfev += 1
        if not np.isfinite(fa):
            raise ValueError('func must return a finite residual')
        if abs(fa) <= tol:
            return xa, fa, True, _ECONVERGED, 0, nfev
        slb = np.sign(fa)
        xlb = xa

    if np.isfinite(xb):
        if xb == x:
            fb = fx
        else:
            values[:] = func(xb, *args)
            fb = float(values[0])
            nfev += 1
        if not np.isfinite(fb):
            raise ValueError('func must return a finite residual')
        if abs(fb) <= tol:
            return xb, fb, True, _ECONVERGED, 0, nfev
        sub = np.sign(fb)
        xub = xb

    if np.isfinite(xa) and np.isfinite(xb) and slb * sub > 0.0:
        raise ValueError('Invalid initial bracket')

    has_bracket = slb * sub < 0.0
    xstart = x

    if jac:
        fpx = initial_fpx
    elif jac_func is not None:
        fpx = float(jac_func(x, *args))
    else:
        if x + eps <= xb or x - eps < xa:
            fpx = nderiv_scalar(func, x, fx, eps, *args)
        else:
            fpx = nderiv_scalar(func, x, fx, -eps, *args)
        nfev += 1
    if not np.isfinite(fpx):
        raise ValueError('derivative must be finite')

    for it in range(1, maxiter + 1):
        if fpx == 0.0:
            return x, fx, False, _EVALUEERR, it, nfev

        candidate = float(x - fx / fpx)
        if has_bracket and (
            not np.isfinite(candidate) or candidate < xlb or candidate > xub
        ):
            s = slb * np.sign(fx)
            if s > 0.0:
                xlb = x
            else:
                xub = x
            candidate = (xlb + xub) / 2.0
        elif not np.isfinite(candidate):
            raise ValueError('Newton step is not finite')

        previous = x
        x = candidate
        values[:] = func(x, *args)
        fx = float(values[0])
        nfev += 1
        if not np.isfinite(fx):
            raise ValueError('func must return a finite residual')

        if abs(fx) <= tol or abs(x - previous) <= xtol:
            return x, fx, True, _ECONVERGED, it, nfev

        if jac:
            fpx = float(values[1])
        elif jac_func is not None:
            fpx = float(jac_func(x, *args))
        else:
            if x + eps <= xb or x - eps < xa:
                fpx = nderiv_scalar(func, x, fx, eps, *args)
            else:
                fpx = nderiv_scalar(func, x, fx, -eps, *args)
            nfev += 1
        if not np.isfinite(fpx):
            raise ValueError('derivative must be finite')

        s = slb * np.sign(fx)
        if not has_bracket:
            if s < 0.0:
                if x > xub:
                    xlb = xub
                    xub = x
                    sub = np.sign(fx)
                elif x < xlb:
                    xub = xlb
                    xlb = x
                    sub = slb
                    slb = np.sign(fx)
                else:
                    dub = abs(xstart - xub)
                    dlb = abs(xstart - xlb)
                    if dub < dlb:
                        xlb = x
                        sub = slb
                        slb = np.sign(fx)
                    else:
                        xub = x
                        sub = np.sign(fx)
                has_bracket = True
            else:
                xlb = min(xlb, x)
                xub = max(xub, x)
        elif s > 0.0:
            xlb = x
        else:
            xub = x

    return x, fx, False, _EMAXITER, maxiter, nfev


@register_jitable(**JIT_OPTIONS)
def newton_bisect_simple(
    func: ScalarFunc | ValueJacFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: bool = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: bool = False,
) -> tuple[float, float]:
    """Return the root and residual using a boolean derivative mode."""
    root, fx, _converged, _flag, _it, _nfev = newton_bisect_impl(
        func, x0, a, b, args, jac, eps, xtol, tol, maxiter
    )
    return root, fx


@register_jitable(**JIT_OPTIONS)
def newton_bisect_full(
    func: ScalarFunc | ValueJacFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: bool = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: bool = False,
) -> tuple[float, RootResult]:
    """Return the root and convergence result using a boolean derivative mode."""
    root, fx, converged, flag, it, nfev = newton_bisect_impl(
        func, x0, a, b, args, jac, eps, xtol, tol, maxiter
    )

    result = RootResult()
    result.root = root
    result.fx = fx
    result.converged = converged
    result.flag = _status_message(flag)
    result.iterations = it
    result.function_calls = nfev
    return root, result


@register_jitable(**JIT_OPTIONS)
def newton_bisect_callable_simple(
    func: ScalarFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: Any = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: bool = False,
) -> tuple[float, float]:
    """Return the root and residual using a separate derivative function."""
    root, fx, _converged, _flag, _it, _nfev = newton_bisect_impl(
        func, x0, a, b, args, False, eps, xtol, tol, maxiter, jac
    )
    return root, fx


@register_jitable(**JIT_OPTIONS)
def newton_bisect_callable_full(
    func: ScalarFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: Any = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: bool = False,
) -> tuple[float, RootResult]:
    """Return the root and result using a separate derivative function."""
    root, fx, converged, flag, it, nfev = newton_bisect_impl(
        func, x0, a, b, args, False, eps, xtol, tol, maxiter, jac
    )

    result = RootResult()
    result.root = root
    result.fx = fx
    result.converged = converged
    result.flag = _status_message(flag)
    result.iterations = it
    result.function_calls = nfev
    return root, result
