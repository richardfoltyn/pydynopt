"""
Basic routines to create and manipulate arrays.

- Generation of power-spaced and logarithmically-spaced 1D grids
- JIT-compiled probability clipping for scalars and arrays

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Sequence
from math import log
from typing import Any

import numpy as np

from pydynopt.numba import JIT_OPTIONS, overload, register_jitable

from .numba.arrays import clip_prob_array, clip_prob_array_impl, clip_prob_scalar

__all__ = [
    'clip_prob',
    'logspace',
    'powerspace',
]


@register_jitable(**JIT_OPTIONS)
def powerspace(xmin: float, xmax: float, n: int, exponent: float) -> np.ndarray:
    """
    Create a power-spaced grid of size n.

    Parameters
    ----------
    xmin
        Lower bound of the grid.
    xmax
        Upper bound of the grid.
    n
        Number of grid points.
    exponent
        Shape parameter of the power-spaced grid.

    Returns
    -------
    Array containing the power-spaced grid.
    """
    n_pts = int(n)
    ffrom, fto = float(xmin), float(xmax)
    fexponent = float(exponent)

    zz = np.linspace(0.0, 1.0, n_pts)
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
    start: float,
    stop: float,
    num: int,
    log_shift: float = 0.0,
    x0: float | None = None,
    frac_at_x0: float | None = None,
    insert_vals: Sequence[float] | np.ndarray | float | None = None,
) -> np.ndarray:
    """
    Create a grid that is by default uniformly spaced in logarithms.

    Alternatively, additional arguments can be specified to alter the grid
    point density, particularly in the left tail of the grid.

    Parameters
    ----------
    start
        Lower bound of the grid.
    stop
        Upper bound of the grid.
    num
        Number of grid points.
    log_shift
        Shift parameter added before taking logarithms.
    x0
        Reference point at which ``frac_at_x0`` fraction of grid points
        is placed. Defaults to ``(stop + start) / 2.0``.
    frac_at_x0
        Fraction of grid points located in the interval ``[start, x0]``.
    insert_vals
        Values to insert into the generated grid while preserving order.

    Returns
    -------
    Array containing the generated grid.
    """
    from scipy.optimize import brentq

    inserted: np.ndarray | None = None
    if insert_vals is not None:
        inserted = np.atleast_1d(insert_vals)

    if frac_at_x0 is not None:
        frac = float(frac_at_x0)
        if frac <= 0.0 or frac >= 1.0:
            msg = f'Invalid argument frac_at_x0: {frac_at_x0}'
            raise ValueError(msg)

        if x0 is None:
            x0 = (stop + start) / 2.0
        elif x0 <= start:
            msg = 'Invalid argument: x0 > start required!'
            raise ValueError(msg)

        def fobj(x: float) -> float:
            dist = np.log(stop + x) - np.log(start + x)
            fx = np.log(x0 + x) - np.log(start + x) - frac * dist
            return float(fx)

        ub = stop - start
        for _ in range(10):
            if fobj(ub) < 0:
                break
            ub *= 10
        else:
            msg = (
                f'Cannot find grid spacing for parameters x0={x0:g} and '
                f'frac_at_x0={frac_at_x0:g}'
            )
            raise ValueError(msg)

        log_shift = float(brentq(fobj, -start + 1.0e-12, ub))

    lstart, lstop = log(start + log_shift), log(stop + log_shift)

    rem = 0 if inserted is None else len(inserted)

    grid = np.linspace(lstart, lstop, num - rem)
    grid = np.exp(grid) - log_shift

    if inserted is not None and len(inserted) > 0:
        idx_insert = np.searchsorted(grid, inserted) + 1
        grid = np.insert(grid, idx_insert, inserted)

    # There may be precision issues resulting in
    # x != exp(log(x + log_shift) - log_shift)
    # so replace the start and stop values with the requested values
    grid[0] = start
    grid[-1] = stop

    return grid


def clip_prob(
    value: float | np.ndarray, tol: float, out: np.ndarray | None = None
) -> float | np.ndarray:
    """
    Clip probabilities close to 0 or 1.

    Parameters
    ----------
    value
        Probability value or array of probabilities to clip.
    tol
        Clipping tolerance. Values strictly less than ``tol`` are set to 0.0,
        and values strictly greater than ``1.0 - tol`` are set to 1.0.
    out
        Optional output array for array inputs (ignored for scalar inputs).

    Returns
    -------
    Clipped probability value or array of values.
    """
    if isinstance(value, np.ndarray):
        return clip_prob_array(value, tol, out)
    return clip_prob_scalar(float(value), tol)


@overload(clip_prob, jit_options=JIT_OPTIONS)
def clip_prob_generic(
    value: Any, tol: Any, out: Any = None
) -> Callable[..., Any] | None:
    """
    Generic for scalar arguments and array arguments without a return array ``out``.
    """
    from numba import types

    from .numba.arrays import clip_prob_array, clip_prob_scalar

    f = None
    if isinstance(value, types.Float):
        f = clip_prob_scalar
    elif isinstance(value, types.Array) and out is None:
        f = clip_prob_array

    return f


@overload(clip_prob, jit_options=JIT_OPTIONS)
def clip_prob_impl_generic(value: Any, tol: Any, out: Any) -> Callable[..., Any] | None:
    """
    Generic for array arguments with an ``out`` argument that is not None.
    """
    from numba import types

    f = None
    if isinstance(value, types.Array) and out is not None:
        f = clip_prob_array_impl

    return f
