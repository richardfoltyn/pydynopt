"""Provide checked scalar root-finding functions with Numba support.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable
from typing import Any, Literal, overload as typing_overload

import numpy as np

from pydynopt.numba import JIT_OPTIONS, overload as numba_overload

from ._zeros_scipy import (
    RootResult,
    _brentq_full,
    _brentq_simple,
    _select_brentq_impl,
    _status_message,
)
from .common import (
    RealScalar,
    ScalarFunc,
    ValueJacFunc,
    _normalize_bool,
    _normalize_integer,
    _normalize_real_scalar,
)
from .numba.zeros import (
    newton_bisect_callable_full,
    newton_bisect_callable_simple,
    newton_bisect_full,
    newton_bisect_impl,
    newton_bisect_simple,
)

__all__ = ['brentq', 'newton_bisect']

type JacFunc = Callable[..., RealScalar]
type JacOption = bool | np.bool_ | JacFunc

_BRENT_XTOL = 2.0e-12
_BRENT_RTOL = 4.0 * np.finfo(float).eps
_BRENT_MAXITER = 100


@typing_overload
def brentq(
    f: ScalarFunc,
    a: RealScalar,
    b: RealScalar,
    args: tuple[Any, ...] = (),
    xtol: RealScalar = _BRENT_XTOL,
    rtol: RealScalar = _BRENT_RTOL,
    maxiter: int = _BRENT_MAXITER,
    full_output: Literal[False] = False,
    disp: bool = True,
) -> float: ...


@typing_overload
def brentq(
    f: ScalarFunc,
    a: RealScalar,
    b: RealScalar,
    args: tuple[Any, ...] = (),
    xtol: RealScalar = _BRENT_XTOL,
    rtol: RealScalar = _BRENT_RTOL,
    maxiter: int = _BRENT_MAXITER,
    full_output: Literal[True] = True,
    disp: bool = True,
) -> tuple[float, RootResult]: ...


@typing_overload
def brentq(
    f: ScalarFunc,
    a: RealScalar,
    b: RealScalar,
    args: tuple[Any, ...] = (),
    xtol: RealScalar = _BRENT_XTOL,
    rtol: RealScalar = _BRENT_RTOL,
    maxiter: int = _BRENT_MAXITER,
    full_output: bool = False,
    disp: bool = True,
) -> float | tuple[float, RootResult]: ...


def brentq(
    f: ScalarFunc,
    a: RealScalar,
    b: RealScalar,
    args: tuple[Any, ...] = (),
    xtol: RealScalar = _BRENT_XTOL,
    rtol: RealScalar = _BRENT_RTOL,
    maxiter: int = _BRENT_MAXITER,
    full_output: bool = False,
    disp: bool = True,
) -> float | tuple[float, RootResult]:
    """Find a root in a bracketing interval using Brent's method.

    Parameters
    ----------
    f
        Continuous scalar function whose root is sought.
    a, b
        Endpoints with opposite-signed function values.
    args
        Additional arguments passed to ``f``.
    xtol, rtol
        Absolute and relative root tolerances.
    maxiter
        Maximum number of iterations.
    full_output
        Whether to return convergence information with the root.
    disp
        Whether non-convergence raises ``RuntimeError``.

    Returns
    -------
    root
        Estimated root location.
    result
        Convergence information, returned only when ``full_output`` is true.

    Raises
    ------
    TypeError
        If a scalar or boolean option has an invalid type.
    ValueError
        If an option or endpoint is invalid, the endpoint values have the same
        sign, or an objective evaluation is NaN.
    RuntimeError
        If the routine does not converge and ``disp`` is true.
    """
    lower = _normalize_real_scalar(a, 'a', finite=False)
    upper = _normalize_real_scalar(b, 'b', finite=False)
    if np.isnan(lower) or np.isnan(upper):
        raise ValueError('a and b must not be NaN')

    abs_tol = _normalize_real_scalar(xtol, 'xtol')
    rel_tol = _normalize_real_scalar(rtol, 'rtol')
    count = _normalize_integer(maxiter, 'maxiter', 0)
    return_result = _normalize_bool(full_output, 'full_output')
    raise_failure = _normalize_bool(disp, 'disp')

    if return_result:
        return _brentq_full(
            f,
            lower,
            upper,
            args,
            abs_tol,
            rel_tol,
            count,
            full_output=True,
            disp=raise_failure,
        )
    return _brentq_simple(
        f,
        lower,
        upper,
        args,
        abs_tol,
        rel_tol,
        count,
        full_output=False,
        disp=raise_failure,
    )


@numba_overload(brentq, jit_options=JIT_OPTIONS)
def _overload_brentq(
    f: Any,
    a: Any,
    b: Any,
    args: Any = (),
    xtol: Any = _BRENT_XTOL,
    rtol: Any = _BRENT_RTOL,
    maxiter: Any = _BRENT_MAXITER,
    full_output: Any = False,
    disp: Any = True,
) -> Any:
    """Select a Numba-compatible implementation of pydynopt's Brent solver."""
    return _select_brentq_impl(full_output)


@typing_overload
def newton_bisect(
    func: ScalarFunc | ValueJacFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: JacOption = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: Literal[False] = False,
) -> tuple[float, float]: ...


@typing_overload
def newton_bisect(
    func: ScalarFunc | ValueJacFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: JacOption = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: Literal[True] = True,
) -> tuple[float, RootResult]: ...


@typing_overload
def newton_bisect(
    func: ScalarFunc | ValueJacFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: JacOption = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: bool = False,
) -> tuple[float, float] | tuple[float, RootResult]: ...


def newton_bisect(
    func: ScalarFunc | ValueJacFunc,
    x0: RealScalar,
    a: RealScalar | None = None,
    b: RealScalar | None = None,
    args: tuple[Any, ...] = (),
    jac: JacOption = False,
    eps: RealScalar = 1.0e-8,
    xtol: RealScalar = 1.0e-8,
    tol: RealScalar = 1.0e-8,
    maxiter: int = 50,
    full_output: bool = False,
) -> tuple[float, float] | tuple[float, RootResult]:
    """Find a scalar root using Newton steps constrained by bisection.

    ``jac=False`` numerically differentiates ``func``. With ``jac=True``,
    ``func`` returns the residual and derivative. A callable ``jac`` evaluates
    the derivative separately.

    Parameters
    ----------
    func
        Scalar residual function, or a function returning the residual and
        derivative when ``jac`` is true.
    x0
        Initial root estimate.
    a, b
        Optional lower and upper bounds. Reversed bounds are normalized.
    args
        Additional arguments passed to ``func`` and a callable ``jac``.
    jac
        Derivative mode or separate derivative function.
    eps
        Positive finite-difference step used when ``jac`` is false.
    xtol
        Absolute tolerance for changes in the root estimate.
    tol
        Absolute tolerance for the residual.
    maxiter
        Maximum number of iterations.
    full_output
        Whether to return convergence information instead of the residual.

    Returns
    -------
    root
        Estimated root location.
    result
        Residual at ``root``, or convergence information when ``full_output``
        is true.

    Raises
    ------
    TypeError
        If a scalar or boolean option has an invalid type.
    ValueError
        If an option, bound, residual, or derivative is invalid, or a supplied
        two-sided bracket does not contain a sign change.
    """
    point = _normalize_real_scalar(x0, 'x0')
    lower = None if a is None else _normalize_real_scalar(a, 'a', finite=False)
    upper = None if b is None else _normalize_real_scalar(b, 'b', finite=False)
    step = _normalize_real_scalar(eps, 'eps')
    abs_tol = _normalize_real_scalar(xtol, 'xtol')
    value_tol = _normalize_real_scalar(tol, 'tol')
    count = _normalize_integer(maxiter, 'maxiter', 1)
    return_result = _normalize_bool(full_output, 'full_output')

    if not isinstance(jac, (bool, np.bool_)):
        result = newton_bisect_impl(
            func,
            point,
            lower,
            upper,
            args,
            False,
            step,
            abs_tol,
            value_tol,
            count,
            jac,
        )
    else:
        use_combined = _normalize_bool(jac, 'jac')
        result = newton_bisect_impl(
            func,
            point,
            lower,
            upper,
            args,
            use_combined,
            step,
            abs_tol,
            value_tol,
            count,
        )

    root, fx, converged, flag, it, nfev = result
    if return_result:
        output = RootResult()
        output.root = root
        output.fx = fx
        output.converged = converged
        output.flag = _status_message(flag)
        output.iterations = it
        output.function_calls = nfev
        return root, output
    return root, fx


@numba_overload(newton_bisect, jit_options=JIT_OPTIONS)
def _overload_newton_bisect(
    func: Any,
    x0: Any,
    a: Any = None,
    b: Any = None,
    args: Any = (),
    jac: Any = False,
    eps: Any = 1.0e-8,
    xtol: Any = 1.0e-8,
    tol: Any = 1.0e-8,
    maxiter: Any = 50,
    full_output: Any = False,
) -> Any:
    """Select a Numba implementation by derivative and result mode."""
    from numba import types

    if full_output is False or isinstance(full_output, types.Omitted):
        return_result = False
    elif isinstance(full_output, types.BooleanLiteral):
        return_result = full_output.literal_value
    else:
        return None

    if jac is False or isinstance(jac, (types.Omitted, types.BooleanLiteral)):
        callable_jac = False
    elif jac == types.boolean:
        return None
    else:
        callable_jac = True

    if callable_jac:
        if return_result:
            return newton_bisect_callable_full
        return newton_bisect_callable_simple
    if return_result:
        return newton_bisect_full
    return newton_bisect_simple
