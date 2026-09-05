"""Provide Numba-compatible kernels for array creation and manipulation.

- Clip scalar and array probabilities.
- Create power-spaced one-dimensional grids.

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
    'powerspace_impl',
]


@register_jitable(**JIT_OPTIONS)
def clip_prob_scalar(value: float | np.number, tol: float) -> float:
    """Clip one probability.

    Parameters
    ----------
    value
        Value to clip.
    tol
        Finite tolerance satisfying ``0 <= tol <= 0.5``.

    Returns
    -------
    The clipped value.

    Raises
    ------
    ValueError
        If ``tol`` is outside its valid range.
    """
    if not np.isfinite(tol) or tol < 0.0 or tol > 0.5:
        msg = 'tol must satisfy 0 <= tol <= 0.5'
        raise ValueError(msg)
    if value < tol:
        return 0.0
    if value > 1.0 - tol:
        return 1.0
    return float(value)


@register_jitable(**JIT_OPTIONS)
def clip_prob_array_impl(
    value: np.ndarray,
    tol: float,
    out: np.ndarray,
) -> None:
    """Clip an array into a required output buffer.

    Parameters
    ----------
    value
        Values to clip.
    tol
        Finite tolerance satisfying ``0 <= tol <= 0.5``.
    out
        Writable floating-point buffer with the same shape as ``value``. Every
        element is overwritten.

    Raises
    ------
    ValueError
        If ``tol`` or the output shape is invalid.
    """
    if not np.isfinite(tol) or tol < 0.0 or tol > 0.5:
        msg = 'tol must satisfy 0 <= tol <= 0.5'
        raise ValueError(msg)
    if value.shape != out.shape:
        msg = 'value and out must have equal shapes'
        raise ValueError(msg)

    upper = 1.0 - tol
    for i in range(value.size):
        item = value.flat[i]
        if item < tol:
            out.flat[i] = 0.0
        elif item > upper:
            out.flat[i] = 1.0
        else:
            out.flat[i] = item


def clip_prob_array(
    value: np.ndarray,
    tol: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Clip an array and optionally allocate its output.

    Parameters
    ----------
    value
        Values to clip.
    tol
        Finite tolerance satisfying ``0 <= tol <= 0.5``.
    out
        Optional writable floating-point buffer with the same shape as ``value``.

    Returns
    -------
    A newly allocated ``float64`` array or the supplied output buffer by identity.

    Raises
    ------
    ValueError
        If ``tol`` or the output shape is invalid.
    """
    result = np.empty(value.shape, dtype=np.float64) if out is None else out
    clip_prob_array_impl(value, tol, result)
    return result


@register_jitable(**JIT_OPTIONS)
def powerspace_impl(
    xmin: float,
    xmax: float,
    n: int,
    exponent: float,
) -> np.ndarray:
    """Create a power-spaced grid.

    Parameters
    ----------
    xmin
        First grid boundary.
    xmax
        Second grid boundary.
    n
        Number of points; at least one.
    exponent
        Finite, strictly positive shape exponent.

    Returns
    -------
    A ``float64`` grid with the documented boundary ordering.

    Raises
    ------
    ValueError
        If ``n`` or ``exponent`` is outside its valid range.
    """
    if n < 1:
        msg = 'n must be at least one'
        raise ValueError(msg)
    if not np.isfinite(exponent) or exponent <= 0.0:
        msg = 'exponent must be finite and strictly positive'
        raise ValueError(msg)

    zz = np.linspace(0.0, 1.0, n)
    if xmax > xmin:
        xx = xmin + (xmax - xmin) * zz**exponent
        xx[-1] = xmax
    else:
        xx = xmin - (xmin - xmax) * zz**exponent
        xx[0] = xmin
        xx = xx[::-1]
    return xx
