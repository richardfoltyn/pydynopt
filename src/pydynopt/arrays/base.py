"""
Basic routines to create and manipulate arrays.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""
from math import log

import numpy as np
from scipy.optimize import brentq

from pydynopt.numba import register_jitable, jit
from .numba.arrays import ind2sub_array, ind2sub_scalar
from .numba.arrays import ind2sub_array_impl, ind2sub_scalar_impl
from .numba.arrays import ind2sub_axis_array, ind2sub_axis_scalar
from .numba.arrays import ind2sub_axis_array_impl, ind2sub_axis_scalar_impl
from .numba.arrays import sub2ind_array, sub2ind_scalar
from .numba.arrays import clip_prob_array, clip_prob_scalar

JIT_OPTIONS = {'nopython': True, 'nogil': True, 'parallel': False,
               'cache': True}


@register_jitable(parallel=False)
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


ind2sub_scalar_jit = jit(ind2sub_scalar, **JIT_OPTIONS)
ind2sub_array_jit = jit(ind2sub_array, **JIT_OPTIONS)

ind2sub_scalar_impl_jit = jit(ind2sub_scalar_impl, **JIT_OPTIONS)
ind2sub_array_impl_jit = jit(ind2sub_array_impl, **JIT_OPTIONS)

ind2sub_axis_scalar_jit = jit(ind2sub_axis_scalar, **JIT_OPTIONS)
ind2sub_axis_array_jit = jit(ind2sub_axis_array, **JIT_OPTIONS)

ind2sub_axis_scalar_impl_jit = jit(ind2sub_axis_scalar_impl, **JIT_OPTIONS)
ind2sub_axis_array_impl_jit = jit(ind2sub_axis_array_impl, **JIT_OPTIONS)


def ind2sub(indices, shape, axis=None, out=None):
    """
    Converts a flat index or array of flat indices into a tuple of coordinate
    arrays.

    Equivalent to Numpy's unravel_index(), but with fewer features and thus
    hopefully faster.

    Parameters
    ----------
    indices : int or array_like
        An integer or integer array whose elements are indices into the
        flattened version of an array of dimensions `shape`.
    axis : int, optional
        If not None, restricts the return array to contain only the coordinates
        along `axis`.
    shape : array_like
        The shape of the array to use for unraveling indices.
    out : np.ndarray or None
        Optional output array (only Numpy arrays supported in Numba mode)

    Returns
    -------
    coords : int or np.ndarray
        Array of coordinates
    """

    if np.isscalar(indices):
        if axis is None:
            if out is None:
                # Pre-allocate array here so we don't have array dtype
                # conflicts in the JIT-able code.
                out = np.empty(len(shape), dtype=np.asarray(indices).dtype)

            out = ind2sub_scalar_impl_jit(indices, shape, axis, out)
        else:
            if out is not None:
                # Implementation routine ignores out argument and only returns
                # a scalar.
                out = ind2sub_axis_scalar_impl_jit(indices, shape, axis)
            else:
                # JIT-able routine writes index into first element of out!
                out = ind2sub_axis_scalar_jit(indices, shape, axis, out)
    else:
        if out is None:
            shp = (len(indices), )
            if axis is None:
                # With axis=None, output will be a 2d-array.
                shp += (len(shape), )

        if axis is None:
            out = ind2sub_array_impl_jit(indices, shape, axis, out)
        else:
            out = ind2sub_axis_array_impl_jit(indices, shape, axis, out)

    return out


sub2ind_scalar_jit = jit(sub2ind_scalar, **JIT_OPTIONS)
sub2ind_array_jit = jit(sub2ind_array, **JIT_OPTIONS)


def sub2ind(coords, shape, out=None):
    """
    Converts an array of indices (coordinates) into a multi-dimensional array
    into an array of flat indices.

    Parameters
    ----------
    coords : array_like
        Integer array of coordinates. Coordinates for each dimension are
        arranged along the first axis.
    shape : array_like
        Shape of array into which indices from `coords` apply.
    out : np.ndarray or None
        Optional output array of flat indices.

    Returns
    -------
    out : np.ndarray
        Array of indices into flatted array.
    """

    coords = np.atleast_1d(coords)

    if coords.ndim == 1:
        out = sub2ind_scalar_jit(coords, shape, out)
    elif coords.ndim == 2:
        out = sub2ind_array_jit(coords, shape, out)

    return out


def logspace(start, stop, num, log_shift=0.0, x0=None, frac_at_x0=None,
             insert_vals=None):
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

    if insert_vals:
        insert_vals = np.atleast_1d(insert_vals)

    if frac_at_x0 is not None:
        frac_at_x0 = float(frac_at_x0)
        if frac_at_x0 <= 0.0 or frac_at_x0 >= 1.0:
            msg = f'Invalid argument frac_at_x0: {frac_at_x0}'
            raise ValueError(msg)

        if x0 is None:
            x0 = (stop+start)/2.0
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
            msg = f'Cannot find grid spacing for parameters x0={x0:g} and ' \
                  f'frac_at_x0={frac_at_x0:g}'
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

