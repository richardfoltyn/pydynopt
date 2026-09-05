"""
Numba implementations of linear interpolation routines.

- 1D and 2D bracket location for scalars and arrays
- 1D and 2D linear interpolant evaluation for scalars and arrays
- Combined 1D and 2D interpolation kernels

NOTE: Do not add @jit decorators to functions meant to be overloaded by @overload.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence

import numpy as np

from pydynopt.numba import JIT_OPTIONS, jit, register_jitable

from .search import bsearch_impl

__all__ = [
    'interp1d_array',
    'interp1d_array_impl',
    'interp1d_eval_array',
    'interp1d_eval_array_alloc',
    'interp1d_eval_scalar',
    'interp1d_locate_array',
    'interp1d_locate_array_alloc',
    'interp1d_locate_scalar',
    'interp1d_scalar',
    'interp2d_array',
    'interp2d_eval_array',
    'interp2d_eval_scalar',
    'interp2d_locate_array',
    'interp2d_locate_scalar',
    'interp2d_locate_scalar_impl',
    'interp2d_scalar',
]


@register_jitable(**JIT_OPTIONS)
def interp1d_locate_scalar(
    x: float,
    xp: np.ndarray,
    ilb: int = 0,
    index_out: object = None,
    weight_out: object = None,
) -> tuple[int, float]:
    """
    Compute the interpolation bracketing interval and weight for a scalar value.

    Parameters
    ----------
    x
        Sample point at which to interpolate.
    xp
        Grid points representing domain over which to interpolate.
    ilb
        Initial guess for index of the bracketing interval lower bound.
    index_out
        Ignored, present for signature compatibility.
    weight_out
        Ignored, present for signature compatibility.

    Returns
    -------
    ilb
        Index of lower bound of bracketing interval.
    weight
        Weight on lower bound of bracketing interval.
    """
    ilb = bsearch_impl(x, xp, ilb)
    weight = (xp[ilb + 1] - x) / (xp[ilb + 1] - xp[ilb])

    return ilb, float(weight)


def interp1d_locate_array_alloc(
    x: np.ndarray,
    xp: np.ndarray,
    ilb: int = 0,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute interpolation bracketing intervals and weights for an array.

    Parameters
    ----------
    x
        Sample points at which to interpolate.
    xp
        Grid points representing domain over which to interpolate.
    ilb
        Initial guess for index of lower bound of bracketing interval.
    index_out
        Optional pre-allocated output array for lower bound indices.
    weight_out
        Optional pre-allocated output array for lower bound weights.

    Returns
    -------
    index_out
        Array of lower bound indices.
    weight_out
        Array of lower bound weights.
    """
    lind_out = np.empty_like(x, dtype=np.int64) if index_out is None else index_out
    lwgt_out = np.empty_like(x, dtype=x.dtype) if weight_out is None else weight_out

    return interp1d_locate_array(x, xp, ilb, lind_out, lwgt_out)


@register_jitable(**JIT_OPTIONS)
def interp1d_locate_array(
    x: np.ndarray,
    xp: np.ndarray,
    ilb: int,
    index_out: np.ndarray,
    weight_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute interpolation bracketing intervals and weights into pre-allocated arrays.

    Parameters
    ----------
    x
        Sample points at which to interpolate.
    xp
        Grid points representing domain over which to interpolate.
    ilb
        Initial guess for index of lower bound of bracketing interval.
    index_out
        Pre-allocated output array for lower bound indices.
    weight_out
        Pre-allocated output array for lower bound weights.

    Returns
    -------
    index_out
        Array of lower bound indices.
    weight_out
        Array of lower bound weights.
    """
    lind_out_flat = index_out.reshape((-1,))
    lwgt_out_flat = weight_out.reshape((-1,))

    for i, xi in enumerate(x.flat):
        ilb = bsearch_impl(xi, xp, ilb)
        wgt_lb = (xp[ilb + 1] - xi) / (xp[ilb + 1] - xp[ilb])
        lind_out_flat[i] = ilb
        lwgt_out_flat[i] = wgt_lb

    return index_out, weight_out


@register_jitable(**JIT_OPTIONS)
def interp1d_eval_scalar(
    index: int,
    weight: float,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
    out: object = None,
) -> float:
    """
    Evaluate an interpolant at a single scalar value.

    Parameters
    ----------
    index
        Index of lower bound of bracketing interval.
    weight
        Weight on lower bound of bracketing interval.
    fp
        Function values defined on original grid points.
    extrapolate
        If True, extrapolate values outside of domain.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    out
        Ignored, present for signature compatibility.

    Returns
    -------
    Interpolant evaluated at sample point.
    """
    fx = weight * fp[index] + (1.0 - weight) * fp[index + 1]

    if not extrapolate:
        if weight > 1.0:
            fx = left
        elif weight < 0.0:
            fx = right

    return float(fx)


def interp1d_eval_array_alloc(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Evaluate an interpolant at multiple sample points, allocating output if needed.

    Parameters
    ----------
    index
        Indices of lower bounds of bracketing intervals.
    weight
        Weights on lower bounds of bracketing intervals.
    fp
        Function values defined on original grid points.
    extrapolate
        If True, extrapolate values outside of domain.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    out
        Optional pre-allocated output array.

    Returns
    -------
    Interpolant evaluated at sample points.
    """
    lout = np.empty_like(weight) if out is None else out

    return interp1d_eval_array(index, weight, fp, extrapolate, left, right, lout)


@register_jitable(**JIT_OPTIONS)
def interp1d_eval_array(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool,
    left: float,
    right: float,
    out: np.ndarray,
) -> np.ndarray:
    """
    Evaluate an interpolant at multiple sample points into pre-allocated array.

    Parameters
    ----------
    index
        Indices of lower bounds of bracketing intervals.
    weight
        Weights on lower bounds of bracketing intervals.
    fp
        Function values defined on original grid points.
    extrapolate
        If True, extrapolate values outside of domain.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    out
        Pre-allocated output array.

    Returns
    -------
    Interpolant evaluated at sample points.
    """
    index_flat = index.reshape((-1,))
    weight_flat = weight.reshape((-1,))
    out_flat = out.reshape((-1,))

    for i in range(out_flat.size):
        wgt = weight_flat[i]
        ilb = index_flat[i]
        out_flat[i] = wgt * fp[ilb] + (1.0 - wgt) * fp[ilb + 1]

    if not extrapolate:
        for i in range(out_flat.size):
            wgt = weight_flat[i]
            if wgt > 1.0:
                out_flat[i] = left
            elif wgt < 0.0:
                out_flat[i] = right

    return out


def interp1d_scalar(
    x: float,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: int = 0,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
    out: object = None,
) -> float:
    """
    Locate and evaluate linear interpolant at a single sample point.

    Parameters
    ----------
    x
        Sample point at which to interpolate.
    xp
        Grid points representing domain over which to interpolate.
    fp
        Function values defined on original grid points.
    ilb
        Initial guess for index of lower bound of bracketing interval.
    extrapolate
        If True, extrapolate values outside of domain.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    out
        Ignored, present for signature compatibility.

    Returns
    -------
    Interpolant evaluated at sample point.
    """
    ilb_found, wgt = interp1d_locate_scalar(x, xp, ilb)
    fx = interp1d_eval_scalar(ilb_found, wgt, fp, extrapolate, left, right, out)

    return fx


def interp1d_array(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: int = 0,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Locate and evaluate linear interpolant at a collection of sample points.

    Parameters
    ----------
    x
        Sample points at which to interpolate.
    xp
        Grid points representing domain over which to interpolate.
    fp
        Function values defined on original grid points.
    ilb
        Initial guess for index of lower bound of bracketing interval.
    extrapolate
        If True, extrapolate values outside of domain.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    out
        Optional pre-allocated output array.

    Returns
    -------
    Interpolant evaluated at sample points.
    """
    lout = np.empty_like(x, dtype=x.dtype) if out is None else out

    interp1d_array_impl(x, xp, fp, lout, ilb, extrapolate, left, right)

    return lout


@register_jitable(**JIT_OPTIONS)
def interp1d_array_impl(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    out: np.ndarray,
    ilb: int = 0,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
) -> None:
    """
    Locate and evaluate linear interpolant into pre-allocated array.

    Parameters
    ----------
    x
        Sample points at which to interpolate.
    xp
        Grid points representing domain over which to interpolate.
    fp
        Function values defined on original grid points.
    out
        Pre-allocated output array.
    ilb
        Initial guess for index of lower bound of bracketing interval.
    extrapolate
        If True, extrapolate values outside of domain.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    """
    out_flat = out.reshape((-1, 1))

    for i, xi in enumerate(x.flat):
        ilb, wgt = interp1d_locate_scalar(xi, xp, ilb)
        fx = interp1d_eval_scalar(ilb, wgt, fp, extrapolate, left, right)

        out_flat[i] = fx


def interp2d_locate_scalar(
    x0: float,
    x1: float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate bracketing interval indices and weights for a 2D scalar point.

    Parameters
    ----------
    x0
        Sample point in the first dimension.
    x1
        Sample point in the second dimension.
    xp0
        Grid points in the first dimension.
    xp1
        Grid points in the second dimension.
    ilb
        Optional initial guess for indices in each dimension.
    index_out
        Optional pre-allocated output array for indices.
    weight_out
        Optional pre-allocated output array for weights.

    Returns
    -------
    index_out
        Lower bound indices in each dimension.
    weight_out
        Weights on lower bounds in each dimension.
    """
    lind_out = np.empty(2, dtype=np.int64) if index_out is None else index_out
    lwgt_out = np.empty(2, dtype=np.float64) if weight_out is None else weight_out

    lilb = np.zeros(2, dtype=np.int64)
    if ilb is not None:
        lilb[:] = ilb

    interp2d_locate_scalar_impl(x0, x1, xp0, xp1, lilb, lind_out, lwgt_out)

    return lind_out, lwgt_out


@register_jitable(**JIT_OPTIONS)
def interp2d_locate_scalar_impl(
    x0: float,
    x1: float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: np.ndarray,
    index_out: np.ndarray,
    weight_out: np.ndarray,
) -> None:
    """
    Locate bracketing interval indices and weights for 2D scalar into output arrays.

    Parameters
    ----------
    x0
        Sample point in the first dimension.
    x1
        Sample point in the second dimension.
    xp0
        Grid points in the first dimension.
    xp1
        Grid points in the second dimension.
    ilb
        Initial guess for indices in each dimension.
    index_out
        Output array of shape (2,) for lower bound indices.
    weight_out
        Output array of shape (2,) for lower bound weights.
    """
    ilb0, ilb1 = ilb[0], ilb[1]

    ilb0, wgt0 = interp1d_locate_scalar(x0, xp0, ilb0)
    ilb1, wgt1 = interp1d_locate_scalar(x1, xp1, ilb1)

    index_out[0], index_out[1] = ilb0, ilb1
    weight_out[0], weight_out[1] = wgt0, wgt1


def interp2d_locate_array(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate bracketing interval indices and weights for 2D array sample points.

    Parameters
    ----------
    x0
        Sample points in the first dimension.
    x1
        Sample points in the second dimension.
    xp0
        Grid points in the first dimension.
    xp1
        Grid points in the second dimension.
    ilb
        Optional initial guess for indices in each dimension.
    index_out
        Optional pre-allocated output array for indices.
    weight_out
        Optional pre-allocated output array for weights.

    Returns
    -------
    index_out
        Array of lower bound indices in each dimension.
    weight_out
        Array of weights on lower bounds in each dimension.
    """
    shp = (*tuple(x0.shape), 2)

    lind_out = np.empty(shp, dtype=np.int64) if index_out is None else index_out
    lwgt_out = np.empty(shp, dtype=x0.dtype) if weight_out is None else weight_out

    lind_out_flat = lind_out.reshape((-1, 2))
    lwgt_out_flat = lwgt_out.reshape((-1, 2))

    ilb0 = 0
    ilb1 = 0

    if ilb is not None:
        ilb0, ilb1 = ilb[0], ilb[1]

    for i, (x0i, x1i) in enumerate(zip(x0.flat, x1.flat, strict=False)):
        ilb0 = bsearch_impl(x0i, xp0, ilb0)
        ilb1 = bsearch_impl(x1i, xp1, ilb1)

        wgt0 = (xp0[ilb0 + 1] - x0i) / (xp0[ilb0 + 1] - xp0[ilb0])
        wgt1 = (xp1[ilb1 + 1] - x1i) / (xp1[ilb1 + 1] - xp1[ilb1])

        lind_out_flat[i, 0] = ilb0
        lind_out_flat[i, 1] = ilb1

        lwgt_out_flat[i, 0] = wgt0
        lwgt_out_flat[i, 1] = wgt1

    return lind_out, lwgt_out


def interp2d_eval_scalar(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: object = None,
) -> float:
    """
    Evaluate a 2D interpolant at a single sample point.

    Parameters
    ----------
    index
        Lower bound indices in each dimension.
    weight
        Weights on lower bounds in each dimension.
    fp
        Function values evaluated on grid points.
    extrapolate
        If True, extrapolate values outside of domain.
    out
        Ignored, present for signature compatibility.

    Returns
    -------
    Interpolant evaluated at sample point.
    """
    wgt0, wgt1 = weight[0], weight[1]
    ilb0, ilb1 = index[0], index[1]

    if not extrapolate and (not np.all(weight >= 0.0) or not np.all(weight <= 1.0)):
        return np.nan

    fx0_lb = wgt0 * fp[ilb0, ilb1] + (1.0 - wgt0) * fp[ilb0 + 1, ilb1]
    fx0_ub = wgt0 * fp[ilb0, ilb1 + 1] + (1.0 - wgt0) * fp[ilb0 + 1, ilb1 + 1]

    fx = wgt1 * fx0_lb + (1.0 - wgt1) * fx0_ub

    return float(fx)


interp2d_locate_scalar_jit = jit(interp2d_locate_scalar, **JIT_OPTIONS)
interp2d_eval_scalar_jit = jit(interp2d_eval_scalar, **JIT_OPTIONS)


def interp2d_eval_array(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Evaluate a 2D interpolant at multiple sample points.

    Parameters
    ----------
    index
        Lower bound indices in each dimension.
    weight
        Weights on lower bounds in each dimension.
    fp
        Function values evaluated on grid points.
    extrapolate
        If True, extrapolate values outside of domain.
    out
        Optional pre-allocated output array.

    Returns
    -------
    Interpolant evaluated at sample points.
    """
    lout = np.empty_like(weight[..., 0], dtype=fp.dtype) if out is None else out

    index_flat = index.reshape((-1, 2))
    weight_flat = weight.reshape((-1, 2))
    lout_flat = lout.reshape((-1,))

    for i in range(lout.size):
        wgt0, wgt1 = weight_flat[i, 0], weight_flat[i, 1]
        ilb0, ilb1 = index_flat[i, 0], index_flat[i, 1]

        if not extrapolate and (wgt0 < 0.0 or wgt0 > 1.0 or wgt1 < 0.0 or wgt1 > 1.0):
            lout_flat[i] = np.nan
            continue

        # Interpolate in dimension 0
        fx0_lb = wgt0 * fp[ilb0, ilb1] + (1.0 - wgt0) * fp[ilb0 + 1, ilb1]
        fx0_ub = wgt0 * fp[ilb0, ilb1 + 1] + (1.0 - wgt0) * fp[ilb0 + 1, ilb1 + 1]

        # Interpolate in dimension 1
        fx = wgt1 * fx0_lb + (1.0 - wgt1) * fx0_ub

        lout_flat[i] = fx

    return lout


def interp2d_scalar(
    x0: float,
    x1: float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
    out: object = None,
) -> float:
    """
    Locate and evaluate 2D bilinear interpolant at a single sample point.

    Parameters
    ----------
    x0
        Sample point in the first dimension.
    x1
        Sample point in the second dimension.
    xp0
        Grid points in the first dimension.
    xp1
        Grid points in the second dimension.
    fp
        Function values evaluated on grid points.
    ilb
        Optional initial guess for indices in each dimension.
    extrapolate
        If True, extrapolate values outside of domain.
    out
        Ignored, present for signature compatibility.

    Returns
    -------
    Interpolant evaluated at sample point.
    """
    index = np.empty(2, dtype=np.int64)
    weight = np.empty(2, dtype=xp0.dtype)

    interp2d_locate_scalar_jit(x0, x1, xp0, xp1, ilb, index, weight)
    fx = interp2d_eval_scalar_jit(index, weight, fp, extrapolate)

    return float(fx)


def interp2d_array(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Locate and evaluate 2D bilinear interpolant at multiple sample points.

    Parameters
    ----------
    x0
        Sample points in the first dimension.
    x1
        Sample points in the second dimension.
    xp0
        Grid points in the first dimension.
    xp1
        Grid points in the second dimension.
    fp
        Function values evaluated on grid points.
    ilb
        Optional initial guess for indices in each dimension.
    extrapolate
        If True, extrapolate values outside of domain.
    out
        Optional pre-allocated output array.

    Returns
    -------
    Interpolant evaluated at sample points.
    """
    lout = np.empty_like(x0) if out is None else out
    lout_flat = lout.reshape((-1, 1))

    lilb = np.zeros(2, dtype=np.int64)
    wgt = np.zeros(2, dtype=x0.dtype)

    if ilb is not None:
        lilb[:] = ilb

    for i, (x0i, x1i) in enumerate(zip(x0, x1, strict=False)):
        interp2d_locate_scalar_jit(x0i, x1i, xp0, xp1, lilb, lilb, wgt)
        fx = interp2d_eval_scalar_jit(lilb, wgt, fp, extrapolate)
        lout_flat[i] = fx

    return lout
