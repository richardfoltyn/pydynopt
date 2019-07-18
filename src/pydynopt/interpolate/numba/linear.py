"""
INSERT MODULE DOCSTRING HERE

Author: Richard Foltyn
"""

import numpy as np
import numba
from numba import jit
from numba.extending import overload

from .search import bsearch


def interp_bilinear(*args, **kwargs):
    """
    Generic function used to create overloaded function instances for Numba
    calls.

    Parameters
    ----------
    args
    kwargs
    """
    pass


def interp_bilinear_scalar(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Perform bilinear interpolation at given tuple of scalar sample points.

    Parameters
    ----------
    x1 : float
        Sample point in first dimension
    x2 : float
        Sample point in second dimension
    xp1 : np.ndarray
        Grid in first dimension
    xp2 : np.ndarray
        Grid in second dimension
    fp : np.ndarray
        Function evaluated at Cartesian product of `xp1` and  `xp2`
    extrapolate : bool
        If true, extrapolate values at points outside of given domain. Otherwise
        non-interior points will be set to NaN.
    out : None
        Ignored by scalar implementation, present for compatibility with
        array-valued function.

    Returns
    -------
    out : float
        Interpolated function value at given sample point
    """

    if not extrapolate:
        interior = xp1[0] <= x1 <= xp1[-1] and xp2[0] <= x2 <= xp2[-1]
        if not interior:
            fx = np.nan
            return fx

    ilb1 = bsearch(x1, xp1)
    ilb2 = bsearch(x2, xp2)

    # Interpolate in dimension 1
    w1 = (x1 - xp1[ilb1]) / (xp1[ilb1+1] - xp1[ilb1])
    w2 = (x2 - xp2[ilb2]) / (xp2[ilb2+1] - xp2[ilb2])

    fx1_lb = (1.0-w1) * fp[ilb1,ilb2] + w1 * fp[ilb1+1,ilb2]
    fx1_ub = (1.0-w1) * fp[ilb1,ilb2+1] + w1 * fp[ilb1+1,ilb2+1]

    # Interpolate in dimension 2
    fx = (1.0 - w2) * fx1_lb + w2 * fx1_ub

    return fx


def interp_bilinear_array(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Perform bilinear interpolation at array-valued collection of sample points.

    Parameters
    ----------
    x1 : np.ndarray
        Sample points in first dimension
    x2 : np.ndarray
        Sample points in second dimension
    xp1 : np.ndarray
        Grid in first dimension
    xp2 : np.ndarray
        Grid in second dimension
    fp : np.ndarray
        Function evaluated at Cartesian product of `xp1` and  `xp2`
    extrapolate : bool
        If true, extrapolate values at points outside of given domain. Otherwise
        non-interior points will be set to NaN.
    out : np.ndarray or None
        Optional output array

    Returns
    -------
    out : np.ndarray
        Interpolated function values at given sample points
    """

    nx = x1.shape[0]

    if out is None:
        out = np.empty_like(x1)

    for i in range(nx):
        if not extrapolate:
            interior = xp1[0] <= x1[i] <= xp1[-1] and xp2[0] <= x2[i] <= xp2[-1]
            if not interior:
                out[i] = np.nan
                continue

        ilb1 = bsearch(x1[i], xp1)
        ilb2 = bsearch(x2[i], xp2)

        # Interpolate in dimension 1
        w1 = (x1[i] - xp1[ilb1])/(xp1[ilb1 + 1] - xp1[ilb1])
        w2 = (x2[i] - xp2[ilb2])/(xp2[ilb2 + 1] - xp2[ilb2])

        fx1_lb = (1.0 - w1)*fp[ilb1, ilb2] + w1*fp[ilb1 + 1, ilb2]
        fx1_ub = (1.0 - w1)*fp[ilb1, ilb2 + 1] + w1*fp[ilb1 + 1, ilb2 + 1]

        # Interpolate in dimension 2
        out[i] = (1.0 - w2)*fx1_lb + w2*fx1_ub

    return out


@overload(interp_bilinear)
def interp_bilinear_generic(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Generic interface around interp_bilinear() to be used to return the correct
    function to numbafy depending on argument types.

    Parameters
    ----------
    x1
    x2
    xp1
    xp2
    fp
    extrapolate
    out

    Returns
    -------
    fcn : callable or None
    """

    fcn = None

    if all(isinstance(x, numba.types.scalars.Number) for x in (x1, x2)):
        fcn = interp_bilinear_scalar
    elif all(isinstance(x, numba.types.npytypes.Array) for x in (x1, x2)):
        fcn = interp_bilinear_array

    return fcn
