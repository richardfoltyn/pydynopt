"""
Basic array creation and manipulation routines compiled by Numba.

- Probability clipping for scalar values
- Probability clipping for NumPy arrays (in-place and allocating)

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import JIT_OPTIONS, register_jitable

__all__ = [
    'clip_prob_array',
    'clip_prob_array_impl',
    'clip_prob_scalar',
]


def clip_prob_scalar(value: float, tol: float, out: object = None) -> float:
    """
    Clip probabilities close to 0 or 1 for scalar values.

    Parameters
    ----------
    value
        Probability value to clip.
    tol
        Clipping tolerance. Values strictly less than ``tol`` are set to 0.0,
        and values strictly greater than ``1.0 - tol`` are set to 1.0.
    out
        Ignored, present only for API compatibility.

    Returns
    -------
    Clipped probability.
    """
    if value < tol:
        return 0.0
    if value > (1.0 - tol):
        return 1.0
    return value


@register_jitable(**JIT_OPTIONS)
def clip_prob_array_impl(value: np.ndarray, tol: float, out: np.ndarray) -> np.ndarray:
    """
    Clip probabilities close to 0 or 1 using a pre-allocated output array.

    Parameters
    ----------
    value
        Array containing probability values to clip.
    tol
        Clipping tolerance. Values strictly less than ``tol`` are set to 0.0,
        and values strictly greater than ``1.0 - tol`` are set to 1.0.
    out
        Pre-allocated output array to store the result.

    Returns
    -------
    Output array with clipped probabilities.
    """
    out[value < tol] = 0.0
    out[value > (1.0 - tol)] = 1.0
    return out


def clip_prob_array(
    value: np.ndarray, tol: float, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Clip probabilities close to 0 or 1 for array arguments.

    Parameters
    ----------
    value
        Array containing probability values to clip.
    tol
        Clipping tolerance. Values strictly less than ``tol`` are set to 0.0,
        and values strictly greater than ``1.0 - tol`` are set to 1.0.
    out
        Optional output array. If None, a copy of ``value`` is allocated.

    Returns
    -------
    Array with clipped probabilities.
    """
    if out is not None:
        return clip_prob_array_impl(value, tol, out)
    out1 = np.copy(value)
    return clip_prob_array_impl(value, tol, out1)
