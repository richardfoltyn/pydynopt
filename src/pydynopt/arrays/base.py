"""
Basic routines to create and manipulate arrays.

Author: Richard Foltyn
"""
import numpy as np

from pydynopt.numba import register_jitable, jit
from .numba.arrays import ind2sub_array, ind2sub_scalar
from .numba.arrays import sub2ind_array, sub2ind_scalar


JIT_OPTIONS = {'nopython': True, 'nogil': True, 'parallel': False,
               'cache': True}


@register_jitable(parallel=False)
def powerspace(xmin, xmax, n, exponent):
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


def ind2sub(indices, shape, out=None):
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
        out = ind2sub_scalar_jit(indices, shape, out)
    else:
        out = ind2sub_array_jit(indices, shape, out)

    return out


sub2ind_scalar_jit = jit(sub2ind_scalar, **JIT_OPTIONS)
sub2ind_array_jit = jit(sub2ind_array, **JIT_OPTIONS)


def sub2ind(coords, shape, out=None):
    """
    Converts an array of indices (coordinates) into a multi-dimensional array
    into an array of flat indices.

    Parameters
    ----------
    coords : np.ndarray
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
