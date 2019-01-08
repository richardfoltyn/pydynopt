
from .common import FunctionWrapper
from .common import OptimResult

import numpy as np


def newton_bisect(func, x0, a=None, b=None, args=(), jac=False,
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
    a : float or None
        Optional bracket lower bound
    b : float or None
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
        If true, return OptimResult as a second return value.

    Returns
    -------
    x : float
        Contains root if algorithm terminates successfully
    res : OptimResult
        Optional additional return value if `full_output` is True.
    """

    fcn = FunctionWrapper(func, jac=jac, eps=eps)
    it = 0

    xtol = float(xtol)
    tol = float(tol)
    maxiter = int(maxiter)

    def build_res(x, fx, flag, converged):
        if full_output:
            res = OptimResult()
            res.x = x
            res.fx = fx
            res.iterations = it
            res.flag = flag
            res.function_calls = fcn.ncalls
            res.converged = converged
            return x, res
        else:
            return x

    x = x0
    xstart = x0

    fx, fpx = fcn.eval_func_jac(x, *args)
    if np.abs(fx) < tol:
        msg = 'Convergence achieved; |f(x)| < tol'
        return build_res(x, fx, msg, converged=True)

    fa = 0.0
    fb = 0.0

    if a is not None:
        fa = fcn.eval_func(a, *args)
        slb = np.sign(fa)
        xlb = a
    else:
        slb = np.sign(fx)
        xlb = x

    if b is not None:
        fb = fcn.eval_func(b, *args)
        sub = np.sign(fb)
        xub = b
    else:
        sub = np.sign(fx)
        xub = x

    # Check that initial bracket contains a root
    if a is not None and b is not None:
        s = np.sign(fa) * np.sign(fb)
        if s > 0.0:
            msg = 'Invalid initial bracket'
            raise ValueError(msg)

    if xlb > xub:
        # Flip values
        xlb, xub = xub, xlb
        slb, sub = sub, slb

    has_bracket = slb*sub < 0.0

    for it in range(1, maxiter+1):

        if np.abs(fx) < tol:
            msg = 'Convergence achieved; |f(x)| < tol'
            return build_res(x, fx, msg, converged=True)

        if fpx == 0.0:
            msg = 'Derivative evaluated to 0.0'
            return build_res(x, fx, msg, converged=False)

        # Newton step
        x = x0 - fx / fpx

        if has_bracket:
            if x < xlb or x > xub:
                # First, update bracket with newly computed function value.
                # This prevents that routine exits immediately if the initial
                # value is the exact midpoint between initial [a,b] and
                # the first Newton step is outside of [a,b].
                # Note: fx contains function value evaluated at what is now
                # stored in x0.
                s = slb * np.sign(fx)
                if s > 0.0:
                    # f(x) has the same sign as f(xlb)
                    xlb = x0
                else:
                    xub = x0

                # Bisection step: set next candidate to midpoint
                x = (xlb + xub) / 2.0

        # Compute function value and derivative for the NEXT iteration
        fx, fpx = fcn.eval_func_jac(x, *args)

        # Exit if tolerance level on function domain is achieved
        if np.abs(x - x0) < xtol:
            msg = 'Convergence achieved: |x(n) - x(n-1)| < xtol'
            return build_res(x, fx, msg, converged=True)

        s = slb * np.sign(fx)
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
        msg = 'Max. number of iterations exceeded'
        return build_res(x, fx, msg, converged=False)


