"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import overload
from pydynopt.numba import register_jitable
from ._zeros_scipy import _ECONVERGED, _EVALUEERR, _EMAXITER, \
    RootResult
from .common import nderiv


__all__ = ['newton_bisect']


@register_jitable(parallel=False, nogil=False)
def _newton_bisect(func, x0, a=None, b=None, args=(), jac=False,
                   eps=1.0e-8, xtol=1.0e-8, tol=1.0e-8, maxiter=50,
                   full_output=False):
    """
    Find the root of a scalar function using a hybrid approach that
    combines Newton-Raphson and bisection. The idea is to speed up
    convergence using Newton-Raphson steps, but constrain the function domain
    to some interval for functions that are not overly well behaved.

    The algorithm accepts an optional initial bracket [a,b] that restricts the
    step size of any subsequent Newton updates to lie within this bracket.
    The bracket is updated as more points are sampled.

    If no initial bracket is provided, the algorithm attempts to automatically
    create one as points are sampled using the initially unrestricted
    Newton-Raphson updating.

    Parameters
    ----------
    func : callable
        Function whose root should be determined
    x0 : float
        Initial guess for root
    a : float
        Optional bracket lower bound
    b : float
        Optional bracket upper bound
    args : tuple
        Optional arguments passed to `func` as func(x, *args)
    jac : bool or callable
        If True, `func` is assumed to return the function derivative
        together with the function value. If False, the derivative
        is approximated using forward differencing with step size `eps`.
        If `jac` is a callable, it is called to evaluate derivatives.
    eps : float
        Step size use for numerical forward differencing (only for `jac`=False)
    xtol : float
        Termination criterion in terms of function domain: if in the n-th
        iteration |x(n) - x(n-1)| < `xtol`, the algorithm terminates.
    tol : float
        Tolerance for termination such that the algorithm exits when
        |func(x)| < `tol`.
    maxiter : int
        Maximum number of iterations
    full_output : bool
        Ignored in the Numba-compatible version, only present for API
        compatibility. RootResult object is always returned.

    Returns
    -------
    x : float
        Contains root if algorithm terminates successfully
    res : pydynopt.optimize._zeros_scipy.RootResult
        Result object containing additional data.
    """

    it = 0
    nfev = 0

    res = RootResult()

    if xtol < 0.0:
        raise ValueError('xtol >= 0 required')
    if tol < 0.0:
        raise ValueError('tol >= 0 required')
    if eps <= 0.0:
        raise ValueError('eps > 0 required')
    if maxiter < 1:
        raise ValueError('maxiter > 0 required')

    x = x0
    xstart = x0

    xa = -np.inf if a is None else a
    xb = np.inf if b is None else b

    xarr = np.array(x)
    fx_all = np.empty(2, dtype=xarr.dtype)

    fx_all[:] = func(x, *args)
    fx = fx_all[0]
    nfev += 1
    if jac:
        fpx = fx_all[1]
    else:
        if (x + eps) < xb or (x - eps) <= xa:
            # Compute numerical derivative as (f(x+eps)-f(x)) / eps
            # either if x+eps < xub, which avoids evaluating the function
            # outside of the original bounded interval.
            # If either step takes us out of (xa,xb), then use this as
            # the fallback and hope for the best.
            fpx = nderiv(func, x, fx, eps, *args)
        else:
            # Evaluate numerical derivative as (f(x-eps) - f(x))/-eps
            fpx = nderiv(func, x, fx, -eps, *args)
        nfev += 1

    if np.abs(fx) < tol:
        res.converged = True
        res.root = x
        res.fx = fx
        res.flag = _ECONVERGED
        res.iterations = it
        res.function_calls = nfev
        return x, res

    fa = 0.0
    fb = 0.0

    if np.isfinite(xa):
        fx_all[:] = func(xa, *args)
        fa = fx_all[0]
        nfev += 1
        slb = np.sign(fa)
        xlb = xa
    else:
        slb = np.sign(fx)
        xlb = x

    if np.isfinite(xb):
        fx_all[:] = func(xb, *args)
        fb = fx_all[0]
        nfev += 1
        sub = np.sign(fb)
        xub = xb
    else:
        sub = np.sign(fx)
        xub = x

    # Check that initial bracket contains a root
    if np.isfinite(xa) and np.isfinite(xb):
        s = np.sign(fa)*np.sign(fb)
        if s > 0.0:
            msg = 'Invalid initial bracket'
            raise ValueError(msg)

    if xlb > xub:
        # Flip values
        xlb, xub = xub, xlb
        slb, sub = sub, slb

    has_bracket = slb*sub < 0.0

    for it in range(1, maxiter + 1):

        if np.abs(fx) < tol:
            res.converged = True
            res.root = x
            res.fx = fx
            res.iterations = it
            res.function_calls = nfev
            res.flag = _ECONVERGED
            return x, res

        if fpx == 0.0:
            res.converged = False
            res.root = x
            res.fx = fx
            res.iterations = it
            res.function_calls = nfev
            res.flag = _EVALUEERR
            return x, res

        # Newton step
        x = x0 - fx/fpx

        if has_bracket:
            if x < xlb or x > xub:
                # First, update bracket with newly computed function value.
                # This prevents that routine exits immediately if the initial
                # value is the exact midpoint between initial [a,b] and
                # the first Newton step is outside of [a,b].
                # Note: fx contains function value evaluated at what is now
                # stored in x0.
                s = slb*np.sign(fx)
                if s > 0.0:
                    # f(x) has the same sign as f(xlb)
                    xlb = x0
                else:
                    xub = x0

                # Bisection step: set next candidate to midpoint
                x = (xlb + xub)/2.0

        # Compute function value and derivative for the NEXT iteration
        fx_all[:] = func(x, *args)
        fx = fx_all[0]
        nfev += 1

        # Exit if tolerance level on function domain is achieved
        # We do this AFTER computing fx, which needs to be returned, but
        # before potentially numerically differentiating, which may no longer
        # be necessary.
        if np.abs(x - x0) < xtol:
            res.converged = True
            res.root = x
            res.fx = fx
            res.iterations = it
            res.function_calls = nfev
            res.flag = _ECONVERGED
            return x, res

        if jac:
            fpx = fx_all[1]
        else:
            if (x + eps) < xb or (x - eps) <= xa:
                # Compute numerical derivative as (f(x+eps)-f(x)) / eps
                # either if x+eps < xub, which avoids evaluating the function
                # outside of the original bounded interval.
                # If either step takes us out of (xa,xb), then use this as
                # the fallback and hope for the best.
                fpx = nderiv(func, x, fx, eps, *args)
            else:
                # Evaluate numerical derivative as (f(x-eps) - f(x))/-eps
                fpx = nderiv(func, x, fx, -eps, *args)
            nfev += 1

        s = slb*np.sign(fx)
        if not has_bracket:
            if s < 0.0:
                # Create initial bracket
                if x > xub:
                    xlb = xub
                    xub = x
                    # SLB remains unchanged
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
                        # xub remains unchanged
                        sub = slb
                        slb = np.sign(fx)
                    else:
                        xub = x
                        sub = np.sign(fx)
                        # xlb, slb remain unchanged

                has_bracket = True

            else:
                # Note: if s = 0.0 then we must have fx = 0.0 and the algorithm
                # will terminate in the next iteration, so we can ignore the
                # case.

                # Update boundaries if sign has not changed
                xlb = min(xlb, x)
                xub = max(xub, x)

        else:
            # Update existing bracket
            if s > 0.0:
                # f(x) has same sign as f(xlb)
                xlb = x
            else:
                xub = x

        # Store last result for next iteration
        x0 = x

    else:
        # max. number of iterations exceeded
        res.converged = False
        res.root = x
        res.fx = fx
        res.flag = _EMAXITER
        res.iterations = it
        res.function_calls = nfev
        return x, res


def newton_bisect(func, x0, a=None, b=None, args=(), jac=False,
                  eps=1.0e-8, xtol=1.0e-8, tol=1.0e-8, maxiter=50,
                  full_output=False):

    xtol = float(xtol)
    tol = float(tol)
    maxiter = int(maxiter)
    eps = float(eps)

    root, res = _newton_bisect(func, x0, a, b, args, jac, eps, xtol, tol, maxiter)

    if full_output:
        return root, res
    else:
        return root


@overload(newton_bisect, jit_options={'parallel': False, 'nogil': True})
def newton_bisect_generic(func, x0, a=None, b=None, args=(), jac=False,
                  eps=1.0e-8, xtol=1.0e-8, tol=1.0e-8, maxiter=50,
                  full_output=False):

    f = _newton_bisect

    return f
