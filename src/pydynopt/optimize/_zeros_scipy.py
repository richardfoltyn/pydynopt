"""Provide a Numba-compatible implementation of SciPy's Brent root finder.

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

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import brentq as scipy_brentq

from pydynopt.numba import (
    JIT_OPTIONS,
    boolean,
    float64,
    int64,
    jitclass,
    overload,
    register_jitable,
    string,
)

__all__ = ['RootResult']

type ScalarFunc = Callable[..., Any]

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
    _EMAXITER: MAXITER,
    _EINPROGRESS: INPROGRESS,
}


@register_jitable(**JIT_OPTIONS)
def _status_message(flag: int) -> str:
    """Return the public message for an internal termination status."""
    if flag == _ECONVERGED:
        return CONVERGED
    if flag == _ESIGNERR:
        return SIGNERR
    if flag == _ECONVERR:
        return CONVERR
    if flag == _EVALUEERR:
        return VALUEERR
    if flag == _EMAXITER:
        return MAXITER
    return INPROGRESS


@jitclass(
    [
        ('root', float64),
        ('fx', float64),
        ('iterations', int64),
        ('function_calls', int64),
        ('converged', boolean),
        ('flag', string),
    ]
)
class RootResult:
    """Represent the result of a root-finding operation.

    Attributes
    ----------
    root
        Estimated root location.
    fx
        Function value at the estimated root.
    iterations
        Number of iterations performed.
    function_calls
        Number of objective evaluations.
    converged
        Whether the routine converged.
    flag
        Description of the termination status.
    """

    def __init__(self) -> None:
        self.root = 0.0
        self.fx = 0.0
        self.iterations = 0
        self.function_calls = 0
        self.converged = False
        self.flag = ''


@register_jitable(**JIT_OPTIONS)
def _brentq_impl(
    f: ScalarFunc,
    a: float,
    b: float,
    args: tuple[Any, ...] = (),
    xtol: float = _xtol,
    rtol: float = _rtol,
    maxiter: int = _iter,
) -> tuple[float, float, bool, int, int, int]:
    """Find a root in a bracketing interval using Brent's method.

    This is a Python port of SciPy's C implementation for use from Numba.

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

    Returns
    -------
    root
        Estimated root location.
    fx
        Function value at ``root``.
    converged
        Whether the routine converged.
    flag
        Internal termination status.
    iterations
        Number of iterations performed.
    function_calls
        Number of objective evaluations.

    Raises
    ------
    ValueError
        If a tolerance or iteration limit is invalid, the endpoint values have
        the same sign, or an objective evaluation is NaN.
    """
    if xtol <= 0.0:
        raise ValueError('xtol too small')
    if rtol < _rtol:
        raise ValueError('rtol too small')
    if maxiter < 0:
        raise ValueError('maxiter must be >= 0')

    xpre = a
    xcur = b
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    fpre = f(xpre, *args)
    if np.isnan(fpre):
        raise ValueError('The function value at a is NaN; solver cannot continue')

    fcur = f(xcur, *args)
    if np.isnan(fcur):
        raise ValueError('The function value at b is NaN; solver cannot continue')

    nfev = 2
    it = 0

    if fpre == 0.0:
        flag = _ECONVERGED
        converged = True
        return xpre, fpre, converged, flag, it, nfev

    if fcur == 0.0:
        flag = _ECONVERGED
        converged = True
        return xcur, fcur, converged, flag, it, nfev

    if np.signbit(fpre) == np.signbit(fcur):
        raise ValueError('f(a) and f(b) must have different signs')

    for it in range(1, maxiter + 1):
        if np.signbit(fpre) != np.signbit(fcur):
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
        if np.isnan(fcur):
            raise ValueError('The function value is NaN; solver cannot continue')
        nfev += 1

    flag = _ECONVERR
    converged = False

    return xcur, fcur, converged, flag, it, nfev


@register_jitable(**JIT_OPTIONS)
def _brentq_simple(
    f: ScalarFunc,
    a: float,
    b: float,
    args: tuple[Any, ...] = (),
    xtol: float = _xtol,
    rtol: float = _rtol,
    maxiter: int = _iter,
    full_output: bool = False,
    disp: bool = True,
) -> float:
    """Return only the root from the Brent kernel."""
    root, _fx, converged, _flag, _it, _nfev = _brentq_impl(
        f, a, b, args, xtol, rtol, maxiter
    )
    if not converged and disp:
        raise RuntimeError('Failed to converge')

    return root


@register_jitable(**JIT_OPTIONS)
def _brentq_full(
    f: ScalarFunc,
    a: float,
    b: float,
    args: tuple[Any, ...] = (),
    xtol: float = _xtol,
    rtol: float = _rtol,
    maxiter: int = _iter,
    full_output: bool = False,
    disp: bool = True,
) -> tuple[float, RootResult]:
    """Return the root and convergence information from the Brent kernel."""
    root, fx, converged, flag, it, nfev = _brentq_impl(
        f, a, b, args, xtol, rtol, maxiter
    )
    if not converged and disp:
        raise RuntimeError('Failed to converge')

    res = RootResult()
    res.root = root
    res.fx = fx
    res.converged = converged
    res.flag = _status_message(flag)
    res.iterations = it
    res.function_calls = nfev

    return root, res


def _select_brentq_impl(full_output: Any) -> Any:
    """Select a Brent implementation from a Numba ``full_output`` type."""
    from numba import types

    if full_output is False or isinstance(full_output, types.Omitted):
        return _brentq_simple
    if isinstance(full_output, types.BooleanLiteral):
        if full_output.literal_value:
            return _brentq_full
        return _brentq_simple
    return None


@overload(scipy_brentq, jit_options=JIT_OPTIONS)
def _overload_scipy_brentq(
    f: Any,
    a: Any,
    b: Any,
    args: Any = (),
    xtol: Any = _xtol,
    rtol: Any = _rtol,
    maxiter: Any = _iter,
    full_output: Any = False,
    disp: Any = True,
) -> Any:
    """Select a Numba-compatible implementation of SciPy's Brent solver."""
    return _select_brentq_impl(full_output)
