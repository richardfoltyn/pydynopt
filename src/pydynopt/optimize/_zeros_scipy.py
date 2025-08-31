"""
Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

from pydynopt.numba import (
    JIT_OPTIONS,
    overload,
    jitclass,
    register_jitable,
    float64,
    int64,
    boolean,
)

__all__ = ['brentq', 'RootResult']

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

flag_map = {
    _ECONVERGED: CONVERGED,
    _ESIGNERR: SIGNERR,
    _ECONVERR: CONVERR,
    _EVALUEERR: VALUEERR,
    _EINPROGRESS: INPROGRESS,
}


@jitclass(
    [
        ('root', float64),
        ('fx', float64),
        ('iterations', int64),
        ('function_calls', int64),
        ('converged', boolean),
        ('flag', int64),
    ]
)
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


@register_jitable(**JIT_OPTIONS)
def _brentq_impl(
    f,
    a,
    b,
    args=(),
    xtol=2.0e-12,
    rtol=_rtol,
    maxiter=_iter
):
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

    Returns
    -------
    x0 : float
        Zero of `f` between `a` and `b`.
    fx : float
        Function value at the found root.
    converged : bool
        True if the routine converged.
    flag : int
        Integer flag indicating the status of the computation. See flag_map for meanings.
    it : int
        Number of iterations needed to find the root.
    nfev : int
        Number of function evaluations made.

    """

    xpre = a
    xcur = b
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    fpre = f(xpre, *args)
    fcur = f(xcur, *args)

    nfev = 2
    it = 0

    if fpre * fcur > 0:
        flag = _ESIGNERR
        converged = False
        root = np.nan
        return root, np.nan, converged, flag, it, nfev

    if fpre == 0.0:
        flag = _ECONVERGED
        converged = True
        return xpre, fpre, converged, flag, it, nfev

    if fcur == 0.0:
        flag = _ECONVERGED
        converged = True
        return xcur, fcur, converged, flag, it, nfev

    for it in range(1, maxiter+1):

        if fpre * fcur < 0.0:
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

        delta = (xtol + rtol * abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        if fcur == 0 or (abs(sbis) < delta):
            flag = _ECONVERGED
            converged = True
            return xcur, fcur, converged, flag, it, nfev

        if (abs(spre) > delta) and (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = (
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                )

            if 2.0 * abs(stry) < min(abs(spre), 3.0 * abs(sbis) - delta):
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
        nfev += 1

    flag = _ECONVERGED
    converged = True

    return xcur, fcur, converged, flag, it, nfev


@register_jitable(**JIT_OPTIONS)
def _brentq_simple(
    f,
    a,
    b,
    args=(),
    xtol=2.0e-12,
    rtol=_rtol,
    maxiter=_iter,
):

    root, fx, converged, flag, it, nfev = _brentq_impl(f, a, b, args, xtol, rtol, maxiter)

    return root


@register_jitable(**JIT_OPTIONS)
def _brentq_full(
    f,
    a,
    b,
    args=(),
    xtol=2.0e-12,
    rtol=_rtol,
    maxiter=_iter,
    full_output=True
):

    res = RootResult()

    root, fx, converged, flag, it, nfev = _brentq_impl(f, a, b, args, xtol, rtol, maxiter)

    res.root = root
    res.fx = fx
    res.converged = converged
    res.flag = flag
    res.iterations = it
    res.function_calls = nfev

    return res



try:
    from scipy.optimize import brentq

    @overload(brentq, jit_options=JIT_OPTIONS)
    def brentq_generic(
        f, a, b, args=(), xtol=2.0e-12, rtol=_rtol, maxiter=_iter
    ):
        """
        Returns a Numba-compatible implementation of Brent's method that
        can be used to override Scipy's implementation.
        """
        f = _brentq_simple
        return f


    @overload(brentq, jit_options=JIT_OPTIONS)
    def brentq_generic(
        f,
        a,
        b,
        args=(),
        xtol=2.0e-12,
        rtol=_rtol,
        maxiter=_iter,
        full_output=True,
    ):
        """
        Returns a Numba-compatible implementation of Brent's method that
        can be used to override Scipy's implementation.
        """
        f = _brentq_full
        return f


except ImportError:
    brentq = None
    pass
