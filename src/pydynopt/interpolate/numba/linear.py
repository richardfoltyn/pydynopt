"""Provide unchecked Numba-compatible kernels for linear interpolation.

Callers must supply strictly increasing one-dimensional grids with at least two
points, valid lower-bound indices, conformable arrays, and writable output buffers
of an appropriate dtype. Allocation wrappers create int64 indices and float64
weights or interpolation results.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence

import numpy as np

from pydynopt.numba import JIT_OPTIONS, register_jitable

from .search import bsearch_impl

__all__ = [
    'interp1d_array',
    'interp1d_array_impl',
    'interp1d_eval_array',
    'interp1d_eval_array_impl',
    'interp1d_eval_scalar',
    'interp1d_locate_array',
    'interp1d_locate_array_impl',
    'interp1d_locate_scalar',
    'interp1d_scalar',
    'interp2d_array',
    'interp2d_array_impl',
    'interp2d_eval_array',
    'interp2d_eval_array_impl',
    'interp2d_eval_scalar',
    'interp2d_locate_array',
    'interp2d_locate_array_impl',
    'interp2d_locate_scalar',
    'interp2d_locate_scalar_impl',
    'interp2d_scalar',
]


@register_jitable(**JIT_OPTIONS)
def interp1d_locate_scalar(
    x: float | np.number,
    xp: np.ndarray,
    ilb: int = 0,
) -> tuple[int, float]:
    """Locate a scalar sample and return its lower-grid-point weight.

    ``xp`` must satisfy the module grid preconditions, and ``ilb`` must be in
    ``[0, len(xp) - 2]``.
    """
    index = bsearch_impl(x, xp, ilb)
    weight = (xp[index + 1] - x) / (xp[index + 1] - xp[index])
    return index, float(weight)


@register_jitable(**JIT_OPTIONS)
def interp1d_locate_array_impl(
    x: np.ndarray,
    xp: np.ndarray,
    ilb: int,
    index_out: np.ndarray,
    weight_out: np.ndarray,
) -> None:
    """Locate array samples into required output buffers.

    ``x``, ``index_out``, and ``weight_out`` must have identical shapes. Grid and
    lower-bound preconditions match :func:`interp1d_locate_scalar`.
    """
    index = ilb
    for i in range(x.size):
        index, weight = interp1d_locate_scalar(x.flat[i], xp, index)
        index_out.flat[i] = index
        weight_out.flat[i] = weight


def interp1d_locate_array(
    x: np.ndarray,
    xp: np.ndarray,
    ilb: int = 0,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate array samples, allocating omitted int64 and float64 buffers."""
    index = np.empty(x.shape, dtype=np.int64) if index_out is None else index_out
    weight = np.empty(x.shape, dtype=np.float64) if weight_out is None else weight_out
    interp1d_locate_array_impl(x, xp, ilb, index, weight)
    return index, weight


@register_jitable(**JIT_OPTIONS)
def interp1d_eval_scalar(
    index: int | np.integer,
    weight: float | np.number,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
) -> float:
    """Evaluate one located point using a valid lower-bound index and weight."""
    if not extrapolate:
        if weight > 1.0:
            return float(left)
        if weight < 0.0:
            return float(right)

    value = weight * fp[index] + (1.0 - weight) * fp[index + 1]
    return float(value)


@register_jitable(**JIT_OPTIONS)
def interp1d_eval_array_impl(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool,
    left: float,
    right: float,
    out: np.ndarray,
) -> None:
    """Evaluate located samples into an output matching index and weight shapes."""
    for i in range(index.size):
        out.flat[i] = interp1d_eval_scalar(
            index.flat[i],
            weight.flat[i],
            fp,
            extrapolate,
            left,
            right,
        )


def interp1d_eval_array(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate located array samples, allocating a float64 output if omitted."""
    result = np.empty(index.shape, dtype=np.float64) if out is None else out
    interp1d_eval_array_impl(
        index,
        weight,
        fp,
        extrapolate,
        left,
        right,
        result,
    )
    return result


@register_jitable(**JIT_OPTIONS)
def interp1d_scalar(
    x: float | np.number,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: int = 0,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
) -> float:
    """Interpolate one point on conformable one-dimensional grid and value arrays."""
    index, weight = interp1d_locate_scalar(x, xp, ilb)
    return interp1d_eval_scalar(index, weight, fp, extrapolate, left, right)


@register_jitable(**JIT_OPTIONS)
def interp1d_array_impl(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: int,
    extrapolate: bool,
    left: float,
    right: float,
    out: np.ndarray,
) -> None:
    """Interpolate array samples into an output with the same shape as ``x``."""
    index = ilb
    for i in range(x.size):
        index, weight = interp1d_locate_scalar(x.flat[i], xp, index)
        out.flat[i] = interp1d_eval_scalar(
            index,
            weight,
            fp,
            extrapolate,
            left,
            right,
        )


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
    """Interpolate array samples, allocating a float64 output if omitted."""
    result = np.empty(x.shape, dtype=np.float64) if out is None else out
    interp1d_array_impl(x, xp, fp, ilb, extrapolate, left, right, result)
    return result


@register_jitable(**JIT_OPTIONS)
def _initial_indices(
    ilb: Sequence[int] | np.ndarray | None,
) -> tuple[int, int]:
    """Return initial lower-bound indices for two dimensions."""
    if ilb is None:
        return 0, 0
    return int(ilb[0]), int(ilb[1])


@register_jitable(**JIT_OPTIONS)
def interp2d_locate_scalar_impl(
    x0: float | np.number,
    x1: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None,
    index_out: np.ndarray,
    weight_out: np.ndarray,
) -> None:
    """Locate one point into required index and weight buffers of shape ``(2,)``."""
    ilb0, ilb1 = _initial_indices(ilb)
    index0, weight0 = interp1d_locate_scalar(x0, xp0, ilb0)
    index1, weight1 = interp1d_locate_scalar(x1, xp1, ilb1)
    index_out[0] = index0
    index_out[1] = index1
    weight_out[0] = weight0
    weight_out[1] = weight1


def interp2d_locate_scalar(
    x0: float | np.number,
    x1: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate one point, allocating omitted int64 and float64 buffers."""
    index = np.empty(2, dtype=np.int64) if index_out is None else index_out
    weight = np.empty(2, dtype=np.float64) if weight_out is None else weight_out
    interp2d_locate_scalar_impl(x0, x1, xp0, xp1, ilb, index, weight)
    return index, weight


@register_jitable(**JIT_OPTIONS)
def interp2d_locate_array_impl(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None,
    index_out: np.ndarray,
    weight_out: np.ndarray,
) -> None:
    """Locate equal-shaped coordinate arrays into ``x0.shape + (2,)`` buffers."""
    ilb0, ilb1 = _initial_indices(ilb)
    for i in range(x0.size):
        ilb0, weight0 = interp1d_locate_scalar(x0.flat[i], xp0, ilb0)
        ilb1, weight1 = interp1d_locate_scalar(x1.flat[i], xp1, ilb1)
        index_out.flat[2 * i] = ilb0
        index_out.flat[2 * i + 1] = ilb1
        weight_out.flat[2 * i] = weight0
        weight_out.flat[2 * i + 1] = weight1


def interp2d_locate_array(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate equal-shaped coordinate arrays, allocating omitted output buffers."""
    shape = (*x0.shape, 2)
    index = np.empty(shape, dtype=np.int64) if index_out is None else index_out
    weight = np.empty(shape, dtype=np.float64) if weight_out is None else weight_out
    interp2d_locate_array_impl(x0, x1, xp0, xp1, ilb, index, weight)
    return index, weight


@register_jitable(**JIT_OPTIONS)
def interp2d_eval_scalar(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
) -> float:
    """Evaluate one point from valid index and weight arrays of shape ``(2,)``."""
    weight0 = weight[0]
    weight1 = weight[1]
    if not extrapolate and (
        weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0
    ):
        return np.nan

    index0 = index[0]
    index1 = index[1]
    value0 = weight0 * fp[index0, index1] + (1.0 - weight0) * fp[index0 + 1, index1]
    value1 = (
        weight0 * fp[index0, index1 + 1] + (1.0 - weight0) * fp[index0 + 1, index1 + 1]
    )
    value = weight1 * value0 + (1.0 - weight1) * value1
    return float(value)


@register_jitable(**JIT_OPTIONS)
def interp2d_eval_array_impl(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool,
    out: np.ndarray,
) -> None:
    """Evaluate located samples into an output with the corresponding sample shape."""
    for i in range(out.size):
        offset = 2 * i
        weight0 = weight.flat[offset]
        weight1 = weight.flat[offset + 1]
        if not extrapolate and (
            weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0
        ):
            out.flat[i] = np.nan
            continue

        index0 = index.flat[offset]
        index1 = index.flat[offset + 1]
        value0 = weight0 * fp[index0, index1] + (1.0 - weight0) * fp[index0 + 1, index1]
        value1 = (
            weight0 * fp[index0, index1 + 1]
            + (1.0 - weight0) * fp[index0 + 1, index1 + 1]
        )
        out.flat[i] = weight1 * value0 + (1.0 - weight1) * value1


def interp2d_eval_array(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate located samples, allocating a float64 output if omitted."""
    result = np.empty(index.shape[:-1], dtype=np.float64) if out is None else out
    interp2d_eval_array_impl(index, weight, fp, extrapolate, result)
    return result


@register_jitable(**JIT_OPTIONS)
def interp2d_scalar(
    x0: float | np.number,
    x1: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
) -> float:
    """Interpolate one point on conformable two-dimensional grids and values."""
    ilb0, ilb1 = _initial_indices(ilb)
    index0, weight0 = interp1d_locate_scalar(x0, xp0, ilb0)
    index1, weight1 = interp1d_locate_scalar(x1, xp1, ilb1)

    if not extrapolate and (
        weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0
    ):
        return np.nan

    value0 = weight0 * fp[index0, index1] + (1.0 - weight0) * fp[index0 + 1, index1]
    value1 = (
        weight0 * fp[index0, index1 + 1] + (1.0 - weight0) * fp[index0 + 1, index1 + 1]
    )
    value = weight1 * value0 + (1.0 - weight1) * value1
    return float(value)


@register_jitable(**JIT_OPTIONS)
def interp2d_array_impl(
    x0: np.ndarray,
    x1: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None,
    extrapolate: bool,
    out: np.ndarray,
) -> None:
    """Interpolate equal-shaped coordinates into an output of the same shape."""
    ilb0, ilb1 = _initial_indices(ilb)
    for i in range(x0.size):
        ilb0, weight0 = interp1d_locate_scalar(x0.flat[i], xp0, ilb0)
        ilb1, weight1 = interp1d_locate_scalar(x1.flat[i], xp1, ilb1)

        if not extrapolate and (
            weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0
        ):
            out.flat[i] = np.nan
            continue

        value0 = weight0 * fp[ilb0, ilb1] + (1.0 - weight0) * fp[ilb0 + 1, ilb1]
        value1 = weight0 * fp[ilb0, ilb1 + 1] + (1.0 - weight0) * fp[ilb0 + 1, ilb1 + 1]
        out.flat[i] = weight1 * value0 + (1.0 - weight1) * value1


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
    """Interpolate equal-shaped coordinates, allocating float64 output if omitted."""
    result = np.empty(x0.shape, dtype=np.float64) if out is None else out
    interp2d_array_impl(x0, x1, xp0, xp1, fp, ilb, extrapolate, result)
    return result
