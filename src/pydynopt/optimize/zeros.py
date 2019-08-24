from pydynopt.optimize.common import FunctionJacWrapper, FunctionWrapper
from .common import nderiv


from pydynopt.numba import jitclass, float64, int64, boolean, overload
import numpy as np

overload_scipy = False
try:
    from scipy.optimize import brentq
    overload_scipy = True
except ImportError:
    brentq = None
    pass


__all__ = ['brentq', 'newton_bisect']

_iter = 100
_xtol = 2.0e-12
_rtol = 4.0 * np.finfo(float).eps

_ECONVERGED = 0
_ESIGNERR = -1
_ECONVERR = -2
_EVALUEERR = -3
_EMAXITER = -4
_EINPROGRESS = 1

CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
MAXITER = 'maximum iterations exceeded'
INPROGRESS = 'No error'

flag_map = {_ECONVERGED: CONVERGED, _ESIGNERR: SIGNERR, _ECONVERR: CONVERR,
            _EVALUEERR: VALUEERR, _EINPROGRESS: INPROGRESS}


@jitclass([('root', float64),
           ('fx', float64),
           ('iterations', int64),
           ('function_calls', int64),
           ('converged', boolean),
           ('flag', int64)])
class RootResult(object):
    """
    Represents the root finding result.

    Attributes
    ----------
    root : float
        Estimated root location.
    fx : float
        Function value at estimated root location.
    iterations : int
        Number of iterations needed to find the root.
    function_calls : int
        Number of times the function was called.
    converged : bool
        True if the routine converged.
    flag : str
        Description of the cause of termination.
    """

    def __init__(self):
        self.root = 0.0
        self.fx = 0.0
        self.iterations = 0
        self.function_calls = 0
        self.converged = False
        self.flag = 0


def _brentq(f, a, b, args=(), xtol=2.0e-12, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=False):
    """

    Find a root of a function in a bracketing interval using Brent's method.

    This function is a Python port of Scipy's C implementation, with the
    aim of making this implementation compatible with Numba.

    The major difference wo Scipy's version is that a RootResult object
    is always returned, as Numba cannot handle non-uniform return values.

    NOTE: If not called from Numba-compiled code, there is no reason to
    call this function instead of Scipy's implementation.

    Parameters
    ----------
    f : function
        Python function returning a number.  The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)` must
        have opposite signs.
    a : scalar
        One end of the bracketing interval :math:`[a, b]`.
    b : scalar
        The other end of the bracketing interval :math:`[a, b]`.
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be nonnegative. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    maxiter : int, optional
        if convergence is not achieved in `maxiter` iterations, an error is
        raised.  Must be >= 0.
    args : tuple, optional
        containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        Ignored, only present for compatibility with Scipy.
    disp : bool, optional
        Ignored, only present for compatibility with Scipy.

    Returns
    -------
    x0 : float
        Zero of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence.

    """

    xpre = a
    xcur = b
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    res = RootResult()
    res.flag = _EINPROGRESS
    res.converged = False

    fpre = f(xpre, *args)
    fcur = f(xcur, *args)

    res.function_calls = 2

    if fpre*fcur > 0:
        res.flag = _ESIGNERR
        return 0.0, res

    if fpre == 0.0:
        res.flag = _ECONVERGED
        res.converged = True
        return xpre, res

    if fcur == 0.0:
        res.flag = _ECONVERGED
        res.converged = True
        return xcur, res

    res.iterations = 0
    for i in range(maxiter):
        res.iterations += 1

        if fpre*fcur < 0.0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol*abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        if fcur == 0 or (abs(sbis) < delta):
            res.flag = _ECONVERGED
            res.converged = True
            return xcur, res

        if (abs(spre) > delta) and (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                # interpolate
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))

            if 2.0*abs(stry) < min(abs(spre), 3.0*abs(sbis) - delta):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = f(xcur, *args)
        res.function_calls += 1

    res.flag = _ECONVERGED
    res.converged = True

    return xcur, res


if overload_scipy:
    @overload(brentq)
    def brentq_generic(f, a, b, args=(), xtol=2.0e-12, rtol=_rtol,
                       maxiter=_iter, full_output=False, disp=False):
        """
        Returns a Numba-compatible implementation of Brent's method that
        can be used to override Scipy's implementation.

        Returns
        -------
        f : callable
        """
        f = _brentq
        return f


def _newton_bisect(fcn, x0, a=-np.inf, b=np.inf, args=(), jac=False,
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
    func : FunctionWrapper or FunctionJacWrapper
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
    res : RootResult
        Result object containing additional data.
    """

    it = 0

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

    fx, fpx = fcn.eval_func_jac(x, *args)

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

    if np.isfinite(a):
        if jac:
            fa, fignore = func(a, *args)
        else:
            fa = func(a, *args)
        nfev += 1
        slb = np.sign(fa)
        xlb = a
    else:
        slb = np.sign(fx)
        xlb = x

    if np.isfinite(b):
        if jac:
            fb, fignore = func(b, *args)
        else:
            fb = func(b, *args)
        nfev += 1
        sub = np.sign(fb)
        xub = b
    else:
        sub = np.sign(fx)
        xub = x

    # Check that initial bracket contains a root
    if np.isfinite(a) and np.isfinite(b):
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
        if jac:
            fx, fpx = func(x, *args)
            nfev += 1
        else:
            fx = func(x, *args)
            fpx = nderiv(func, x, fx, eps, *args)
            nfev += 2

        # Exit if tolerance level on function domain is achieved
        if np.abs(x - x0) < xtol:
            res.converged = True
            res.root = x
            res.fx = fx
            res.iterations = it
            res.nfev = nfev
            res.flag = _ECONVERGED
            return x, res

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
        res.converged = False
        res.root = x
        res.fx = fx
        res.flag = _EMAXITER
        res.iterations = it
        res.function_calls = nfev
        return x, res


def newton_bisect(func, x0, a=-np.inf, b=np.inf, args=(), jac=False,
                  eps=1.0e-8, xtol=1.0e-8, tol=1.0e-8, maxiter=50,
                  full_output=False):

    xtol = float(xtol)
    tol = float(tol)
    maxiter = int(maxiter)
    eps = float(eps)

    fcn_obj = None
    if jac:
        fcn_obj = FunctionJacWrapper(func)
    else:
        fcn_obj = FunctionWrapper(func, eps=eps)

    root, res = _newton_bisect(fcn_obj, x0, a, b, args, jac, eps, xtol, tol,
                               maxiter)

    if full_output:
        return root, res
    else:
        return res


@overload(newton_bisect, jit_options={'parallel': False, 'nogil': True})
def newton_bisect_generic(func, x0, a=-np.inf, b=np.inf, args=(), jac=False,
                  eps=1.0e-8, xtol=1.0e-8, tol=1.0e-8, maxiter=50,
                  full_output=False):

    f = _newton_bisect
    return f
