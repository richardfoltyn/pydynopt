"""
Module that implements routines for linear interpolation.

Author: Richard Foltyn
"""

import numpy as np
from scipy.interpolate import interpn


def interp_bilinear(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Perform bilinear interpolation at given sample points.

    This is just a wrapper around scipy's implementation in interpn() and is
    only provided such that the Numba-fied version can overload this function,
    and so that non-Numba-fied code works as well.

    Should not be used in code which will never be run via Numba.

    Parameters
    ----------
    x1 : float or np.ndarray
        Sample points in first dimension
    x2 : float or np.ndarray
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
    out : float or np.ndarray
        Interpolated function values at given sample points
    """

    isscalar = np.isscalar(x1) and np.isscalar(x2)
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)

    xx = np.vstack((x1, x2)).T

    if out is None:
        out = np.empty_like(x1)

    fill_value = None if extrapolate else np.nan
    out[:] = interpn((xp1, xp2), fp, xx, bounds_error=False,
                     fill_value=fill_value)

    if isscalar:
        out = out.item()

    return out
