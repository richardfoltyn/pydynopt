"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import overload, jitclass, register_jitable
from pydynopt.numba import int64, float64


class OptimResult:
    def __init__(self):
        self.x = 0.0
        self.fx = 0.0
        self.iterations = 0
        self.function_calls = 0
        self.converged = False
        # Some status message or flag of type string
        self.flag = ""

    def __repr__(self):
        """
        Return string representation of result object.

        Returns
        -------
        s : str
        """

        fmt_float = '{:>20s}: {:g}'
        fmt_int = '{:>20s}: {:d}'
        fmt_default = '{:>20s}: {}'

        attrs = ['converged', 'flag', 'function_calls', 'iterations', 'x']

        tokens = []

        for attr in attrs:
            if not hasattr(self, attr):
                continue

            value = getattr(self, attr)
            if isinstance(value, float):
                s = fmt_float.format(attr, value)
            elif isinstance(value, int):
                s = fmt_int.format(attr, value)
            elif isinstance(value, np.ndarray):
                if np.isscalar(value):
                    s = fmt_float.format(attr, value)
                else:
                    x = np.atleast_1d(value)
                    s = ', '.join('{:g}'.format(xi) for xi in x)
                    s = '[' + s + ']'
            else:
                s = fmt_default.format(attr, value)

            tokens.append(s)

        s = '\n'.join(tokens)
        return s


def _extract_arg_identity(arg, index=0):
    return arg


def _extract_arg_by_index(arg, index=0):
    return arg[index]


def _extract_arg(arg, index=0):
    if isinstance(arg, tuple):
        return arg[index]
    else:
        return arg


@overload(_extract_arg, jit_options={'nogil': True})
def _extract_arg_generic(arg, index=0):

    from numba import types

    f = None
    if isinstance(arg, types.Number):
        f = _extract_arg_identity
    elif isinstance(arg, types.Array):
        f = _extract_arg_identity
    else:
        f = _extract_arg_by_index
    return f


@register_jitable(parallel=False, nogil=True)
def _nderiv_array(func, x, fx=np.nan, eps=1.0e-8, *args):
    """
    Numerically forward-differentiate function and given point.

    Parameters
    ----------
    func : callable
    x : np.ndarray
    fx : float
    eps : float
    args

    Returns
    -------
    fpx : np.ndarray
    """

    n = len(x) + 1
    fx_all = np.empty(n, dtype=x.dtype)

    if np.isnan(fx):
        fx_all[:] = func(x, *args)
        fx = fx_all[0]

    fpx = np.zeros_like(x)
    xxi = np.empty_like(x)

    for i, xi in enumerate(x):
        xxi[:] = np.copy(x)
        xxi[i] += eps
        fx_all[:] = func(xxi, *args)
        fpx[i] = (fx_all[0] - fx) / eps

    return fpx


def _nderiv_scalar(func, x, fx=np.nan, eps=1.0e-8, *args):
    """

    Parameters
    ----------
    func : callable
    x : float
    fx : float
    eps : float
    args

    Returns
    -------
    fpx : float
    """

    xarr = np.array([x])
    fx_all = np.empty(2, dtype=xarr.dtype)

    if np.isnan(fx):
        fx_all[:] = func(x, *args)
        fx = fx_all[0]

    fx_all[:] = func(x + eps, *args)
    dfx = fx_all[0] - fx
    dx = eps
    fpx = dfx / dx

    return fpx


@register_jitable(nogil=True, parallel=False)
def nderiv_multi(func, x, fx=None, eps=1.0e-8, *args):
    """
    Compute derivative of scalar function on scalar domain at multiple values
    at once.

    Parameters
    ----------
    func : callable
    x : np.ndarray
    fx : np.ndarray
    eps : float
    args

    Returns
    -------
    fpx : np.ndarray
    """

    if fx is None:
        obj = func(x, *args)
        fx0 = _extract_arg(obj)
    else:
        fx0 = fx

    obj = func(x + eps, *args)
    fx_eps = _extract_arg(obj)
    fpx = (fx_eps - fx0) / eps

    return fpx


def nderiv(func, x, fx=np.nan, eps=1.0e-8, *args):
    """
    Numerically forward-differentiate function and given point.

    Parameters
    ----------
    func : callable
    x : float or array_like
    fx : float
    eps : float
    args

    Returns
    -------
    fpx : float or np.ndaray
    """

    eps = float(eps)

    if np.isscalar(x):
        fpx = _nderiv_scalar(func, x, fx, eps, *args)
    elif np.array(x).ndim == 1:
        fpx = _nderiv_array(func, x, fx, eps, *args)
    else:
        msg = 'Argument x must be either a scalar of a 1d-array'
        raise ValueError(msg)

    return fpx


@overload(nderiv, jit_options={'parallel': False, 'nogil': True})
def nderiv_generic(func, x, fx=np.nan, eps=1.0e-8, *args):

    from numba import types

    f = None
    if isinstance(x, types.Number):
        f = _nderiv_scalar
    elif isinstance(x, types.Array):
        f = _nderiv_array

    return f

