"""
Basic routines to create and manipulate arrays.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from math import log

import numpy as np

from pydynopt.numba import register_jitable, overload, JIT_OPTIONS
from .numba.arrays import clip_prob_array, clip_prob_scalar


@register_jitable(**JIT_OPTIONS)
def powerspace(xmin: float, xmax: float, n: int, exponent: float) -> np.ndarray:
    """
    Create a "power-spaced" grid of size n.

    Parameters
    ----------
    xmin : float
        Lower bound
    xmax : float
        Upper bound
    n : int
        Number of grid points
    exponent : float
        Shape parameter of "power-spaced" grid.

    Returns
    -------
    xx : np.ndarray
        Array containing "power-spaced" grid
    """

    N = int(n)
    ffrom, fto = float(xmin), float(xmax)
    fexponent = float(exponent)

    zz = np.linspace(0.0, 1.0, N)
    if fto > ffrom:
        xx = ffrom + (fto - ffrom) * zz**fexponent
        # Prevent rounding errors
        xx[-1] = fto
    else:
        xx = ffrom - (ffrom - fto) * zz**fexponent
        xx[0] = ffrom
        xx = xx[::-1]

    return xx


def logspace(
    start, stop, num, log_shift=0.0, x0=None, frac_at_x0=None, insert_vals=None
):
    """
    Create grid that is by default uniformly spaced in logs. Alternatively,
    additional arguments can be specified to alter the grid point density,
    particularly in the left tail of the grid.

    Parameters
    ----------
    start : float
    stop : float
    num : int
    log_shift : float, optional
    x0 : float, optional
    frac_at_x0 : float, optional
    insert_vals : array_like, optional

    Returns
    -------
    grid : np.ndarray
    """

    from scipy.optimize import brentq

    if insert_vals:
        insert_vals = np.atleast_1d(insert_vals)

    if frac_at_x0 is not None:
        frac_at_x0 = float(frac_at_x0)
        if frac_at_x0 <= 0.0 or frac_at_x0 >= 1.0:
            msg = f'Invalid argument frac_at_x0: {frac_at_x0}'
            raise ValueError(msg)

        if x0 is None:
            x0 = (stop + start) / 2.0
        elif x0 <= start:
            msg = 'Invalid argument: x0 > start required!'
            raise ValueError(msg)

        def fobj(x):
            dist = np.log(stop + x) - np.log(start + x)
            fx = np.log(x0 + x) - np.log(start + x) - frac_at_x0 * dist
            return fx

        ub = stop - start
        for it in range(10):
            if fobj(ub) < 0:
                break
            else:
                ub *= 10
        else:
            msg = (
                f'Cannot find grid spacing for parameters x0={x0:g} and '
                f'frac_at_x0={frac_at_x0:g}'
            )
            raise ValueError(msg)

        x = brentq(fobj, -start + 1.0e-12, ub)
        log_shift = x

    lstart, lstop = log(start + log_shift), log(stop + log_shift)

    rem = 0 if insert_vals is None else len(insert_vals)

    grid = np.linspace(lstart, lstop, num - rem)
    grid = np.exp(grid) - log_shift

    if insert_vals is not None and len(insert_vals) > 0:
        idx_insert = np.searchsorted(grid, insert_vals) + 1
        grid = np.insert(grid, idx_insert, insert_vals)

    # there may be some precision issues resulting in
    # x != exp(log(x + log_shift) - log_shift
    # so we replace the start and stop values with the requested values
    grid[0] = start
    grid[-1] = stop

    return grid


def clip_prob(value, tol, out=None):
    """
    Clip probabilities close to 0 or 1 (array implementation when `out` is None).

    Parameters
    ----------
    value : float or np.ndarray
    tol : float
        Clip value < `tol` to 0, and value > (1.0 - `tol`) to 1.
    out : np.ndarray, optional
        Output array (ignored for scalars)

    Returns
    -------
    float or np.ndarray
    """

    if np.isscalar(value):
        return clip_prob_scalar(value, tol)
    else:
        return clip_prob_array(value, tol, out)


@overload(clip_prob, jit_options=JIT_OPTIONS)
def clip_prob_generic(value, tol, out=None):
    """
    Generic for scalar arguments and array arguments without a return array `out`.
    """

    from numba import types
    from .numba.arrays import clip_prob_scalar, clip_prob_array

    f = None
    if isinstance(value, types.Float):
        f = clip_prob_scalar
    elif isinstance(value, types.Array) and out is None:
        f = clip_prob_array

    return f


@overload(clip_prob, jit_options=JIT_OPTIONS)
def clip_prob_generic(value, tol, out):
    """
    Generic fo array arguments with an `out` argument that is not None.
    """

    from numba import types
    from .numba.arrays import clip_prob_array_impl

    f = None
    if isinstance(value, types.Array) and out is not None:
        f = clip_prob_array_impl

    return f
