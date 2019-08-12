"""
Basic routines to create and manipulate arrays.

Author: Richard Foltyn
"""
import numpy as np

from pydynopt.numba import register_jitable


@register_jitable(nopython=True, parallel=False)
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
    fxmin, fxmax = float(xmin), float(xmax)
    fexponent = float(exponent)

    xx = np.linspace(0.0, 1.0, N)
    xx = fxmin + (fxmax - fxmin) * xx**fexponent
    # Prevent rounding errors
    xx[-1] = fxmax

    return xx
