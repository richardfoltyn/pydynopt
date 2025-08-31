"""
Module implementing basic array creation and manipulation routines that
can be compiled by Numba.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import register_jitable, JIT_OPTIONS


def clip_prob_scalar(value, tol, out=None):
    """
    Clip probabilities close to 0 or 1 (scalar implementation).

    Parameters
    ----------
    value : float
    tol : float
        Clip value < `tol` to 0, and value > (1.0 - `tol`) to 1.
    out : Any, optional
        Ignored, only present for API compatibility

    Returns
    -------
    float
    """

    if value < tol:
        value = 0.0
    elif value > (1.0 - tol):
        value = 1.0
    return value


@register_jitable(**JIT_OPTIONS)
def clip_prob_array_impl(value, tol, out):
    """
    Clip probabilities close to 0 or 1 (array implementation when `out` is not None).

    Parameters
    ----------
    value : np.ndarray
    tol : float
        Clip value < `tol` to 0, and value > (1.0 - `tol`) to 1.
    out : np.narray
        Array to store return value

    Returns
    -------
    np.ndarray
    """

    out[value < tol] = 0.0
    out[value > (1.0 - tol)] = 1.0
    return out


def clip_prob_array(value, tol, out=None):
    """
    Clip probabilities close to 0 or 1 (array implementation when `out` is None).

    Parameters
    ----------
    value : np.ndarray
    tol : float
        Clip value < `tol` to 0, and value > (1.0 - `tol`) to 1.
    out : Any, optional
        Ignored, only present for API compatibility

    Returns
    -------
    np.ndarray
    """
    if out is not None:
        # Keep this for calls from Python.
        return clip_prob_array_impl(value, tol, out)
    else:
        out1 = np.copy(value)
        return clip_prob_array_impl(value, tol, out1)
