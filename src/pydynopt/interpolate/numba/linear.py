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
from typing import Any

import numpy as np

from pydynopt.numba import (
    JIT_OPTIONS,
    JIT_OPTIONS_INLINE,
    overload as numba_overload,
    register_jitable,
)

from .search import _bsearch_range

__all__ = [
    'interp1d_array',
    'interp1d_array_impl',
    'interp1d_eval_array',
    'interp1d_eval_array_impl',
    'interp1d_eval_point',
    'interp1d_locate_array',
    'interp1d_locate_array_impl',
    'interp1d_locate_scalar',
    'interp1d_scalar',
    'interp2d_array',
    'interp2d_array_impl',
    'interp2d_eval_array',
    'interp2d_eval_array_impl',
    'interp2d_eval_point',
    'interp2d_locate_array',
    'interp2d_locate_array_impl',
    'interp2d_locate_scalar',
    'interp2d_locate_scalar_impl',
    'interp2d_scalar',
    'interp3d_array',
    'interp3d_array_impl',
    'interp3d_eval_array',
    'interp3d_eval_array_impl',
    'interp3d_eval_point',
    'interp3d_locate_array',
    'interp3d_locate_array_impl',
    'interp3d_locate_point',
    'interp3d_locate_point_impl',
    'interp3d_point',
]


@register_jitable(**JIT_OPTIONS_INLINE)
def interp1d_locate_scalar(
    x: float | np.number,
    xp: np.ndarray,
    ilb: int = 0,
) -> tuple[int, float]:
    """Locate a scalar sample and return its lower-grid-point weight.

    ``xp`` must satisfy the module grid preconditions, and ``ilb`` must be in
    ``[0, len(xp) - 2]``.
    """
    # Preserve branch-local early returns: merged endpoints produce poor Numba SSA.
    # Reuse loaded grid endpoints through weighting on the same/adjacent hot paths.
    index = ilb
    lower = xp[index]
    upper = xp[index + 1]
    if lower <= x:
        if upper > x or index == xp.shape[0] - 2:
            weight = (upper - x) / (upper - lower)
            return index, float(weight)

        next_index = index + 1
        next_upper = xp[next_index + 1]
        if next_upper > x or next_index == xp.shape[0] - 2:
            weight = (next_upper - x) / (next_upper - upper)
            return next_index, float(weight)

        # Keep distant search out of line so local callers do not absorb its loop.
        range_index = _bsearch_range(x, xp, next_index, xp.shape[0] - 1)
        range_lower = xp[range_index]
        range_upper = xp[range_index + 1]
        weight = (range_upper - x) / (range_upper - range_lower)
        return range_index, float(weight)

    if index == 0:
        weight = (upper - x) / (upper - lower)
        return index, float(weight)

    previous = xp[index - 1]
    if previous <= x:
        weight = (lower - x) / (lower - previous)
        return index - 1, float(weight)

    # Keep index, not index - 1: the comparisons above are unordered for NaN.
    range_index = _bsearch_range(x, xp, 0, index)
    range_lower = xp[range_index]
    range_upper = xp[range_index + 1]
    weight = (range_upper - x) / (range_upper - range_lower)
    return range_index, float(weight)


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


@register_jitable(**JIT_OPTIONS_INLINE)
def interp1d_eval_point(
    index: int | np.integer,
    weight: float | np.number,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
) -> float:
    """Evaluate one located point using a valid lower-bound index and weight."""
    # Inline this leaf, but keep the public eval overload out of repeated-field callers.
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
        out.flat[i] = interp1d_eval_point(
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


@register_jitable(**JIT_OPTIONS_INLINE)
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
    # Both leaves must inline into the scalar-only public overload to remove tuple calls.
    index, weight = interp1d_locate_scalar(x, xp, ilb)
    return interp1d_eval_point(index, weight, fp, extrapolate, left, right)


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
    # Version this invariant outside the loop; Numba did not reliably unswitch it.
    if extrapolate:
        for i in range(x.size):
            index, weight = interp1d_locate_scalar(x.flat[i], xp, index)
            value = weight * fp[index] + (1.0 - weight) * fp[index + 1]
            out.flat[i] = float(value)
        return

    for i in range(x.size):
        index, weight = interp1d_locate_scalar(x.flat[i], xp, index)
        out.flat[i] = interp1d_eval_point(
            index,
            weight,
            fp,
            False,
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


def _initial_indices(
    ilb: Sequence[int] | np.ndarray | None,
) -> tuple[int, int]:
    """Return initial lower-bound indices for two dimensions."""
    # The compiled overload below removes this polymorphic branch at typing time.
    if ilb is None:
        return 0, 0
    return int(ilb[0]), int(ilb[1])


@numba_overload(_initial_indices, jit_options=JIT_OPTIONS, inline='always')
def _overload_initial_indices(ilb: Any) -> Any:
    from numba import types

    # Separate implementations avoid typing the invalid dead indexing branch for None.
    if isinstance(ilb, types.NoneType):

        def impl(ilb):
            return 0, 0

    else:

        def impl(ilb):
            return int(ilb[0]), int(ilb[1])

    return impl


@register_jitable(**JIT_OPTIONS_INLINE)
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
    # Keep dimension 0 first and contiguous index stores before weight stores.
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


@register_jitable(**JIT_OPTIONS_INLINE)
def interp2d_eval_point(
    index: Sequence[int | np.integer] | np.ndarray,
    weight: Sequence[float | np.number] | np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
) -> float:
    """Evaluate one point from valid index and weight pairs of length two."""
    weight0 = weight[0]
    weight1 = weight[1]
    if not extrapolate and (
        weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0
    ):
        return np.nan

    index0 = index[0]
    index1 = index[1]
    # Build both row bases first, then load adjacent corners in lower/upper row order.
    lower_row = fp[index0]
    upper_row = fp[index0 + 1]
    lower0 = lower_row[index1]
    lower1 = lower_row[index1 + 1]
    upper0 = upper_row[index1]
    upper1 = upper_row[index1 + 1]
    upper_weight0 = 1.0 - weight0
    value0 = weight0 * lower0 + upper_weight0 * upper0
    value1 = weight0 * lower1 + upper_weight0 * upper1
    value = weight1 * value0 + (1.0 - weight1) * value1
    return float(value)


@register_jitable(**JIT_OPTIONS_INLINE)
def _interp2d_eval_point_c(
    index: Sequence[int | np.integer] | np.ndarray,
    weight: Sequence[float | np.number] | np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
) -> float:
    """Evaluate one point with C-contiguous two-dimensional values."""
    weight0 = weight[0]
    weight1 = weight[1]
    if not extrapolate and (
        weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0
    ):
        return np.nan

    # C layout needs one row-major offset instead of four multidimensional addresses.
    # Preserve adjacent row-wise corner loads and the shared complement below.
    n1 = fp.shape[1]
    offset = index[0] * n1 + index[1]
    lower0 = fp.flat[offset]
    lower1 = fp.flat[offset + 1]
    upper0 = fp.flat[offset + n1]
    upper1 = fp.flat[offset + n1 + 1]
    upper_weight0 = 1.0 - weight0
    value0 = weight0 * lower0 + upper_weight0 * upper0
    value1 = weight0 * lower1 + upper_weight0 * upper1
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


@register_jitable(**JIT_OPTIONS_INLINE)
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
    # Keep this generic leaf inline; C-contiguous values use the flat helper below.
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


@register_jitable(**JIT_OPTIONS_INLINE)
def _interp2d_scalar_c(
    x0: float | np.number,
    x1: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
) -> float:
    """Interpolate one point with C-contiguous two-dimensional values."""
    ilb0, ilb1 = _initial_indices(ilb)
    # Dimension 1 is contiguous and fast-moving; this order is faster only here.
    index1, weight1 = interp1d_locate_scalar(x1, xp1, ilb1)
    index0, weight0 = interp1d_locate_scalar(x0, xp0, ilb0)

    if not extrapolate and (
        weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0
    ):
        return np.nan

    # Keep direct flat expressions here; extra corner/complement locals add pressure.
    n1 = fp.shape[1]
    offset = index0 * n1 + index1
    value0 = weight0 * fp.flat[offset] + (1.0 - weight0) * fp.flat[offset + n1]
    value1 = weight0 * fp.flat[offset + 1] + (1.0 - weight0) * fp.flat[offset + n1 + 1]
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
    # Hoist the mode branch and keep dimension 0 search first in both loop versions.
    if extrapolate:
        for i in range(x0.size):
            ilb0, weight0 = interp1d_locate_scalar(x0.flat[i], xp0, ilb0)
            ilb1, weight1 = interp1d_locate_scalar(x1.flat[i], xp1, ilb1)

            # Keep both row bases and this lower/upper contiguous load order intact.
            lower_row = fp[ilb0]
            upper_row = fp[ilb0 + 1]
            lower0 = lower_row[ilb1]
            lower1 = lower_row[ilb1 + 1]
            upper0 = upper_row[ilb1]
            upper1 = upper_row[ilb1 + 1]
            value0 = weight0 * lower0 + (1.0 - weight0) * upper0
            value1 = weight0 * lower1 + (1.0 - weight0) * upper1
            out.flat[i] = weight1 * value0 + (1.0 - weight1) * value1
        return

    for i in range(x0.size):
        ilb0, weight0 = interp1d_locate_scalar(x0.flat[i], xp0, ilb0)
        ilb1, weight1 = interp1d_locate_scalar(x1.flat[i], xp1, ilb1)

        if weight0 < 0.0 or weight0 > 1.0 or weight1 < 0.0 or weight1 > 1.0:
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


def _initial_indices_3d(
    ilb: Sequence[int] | np.ndarray | None,
) -> tuple[int, int, int]:
    """Resolve optional lower-bound hints for unchecked 3D kernels.

    The checked Python entry point validates and clamps supplied hints. This helper
    only removes the optional representation while preserving tuple or array input
    for Numba callers.
    """
    if ilb is None:
        return 0, 0, 0
    return int(ilb[0]), int(ilb[1]), int(ilb[2])


@numba_overload(_initial_indices_3d, jit_options=JIT_OPTIONS, inline='always')
def _overload_initial_indices_3d(ilb: Any) -> Any:
    """Remove the optional-hint branch during Numba compilation.

    Separate implementations prevent Numba from typing the dead indexing path for
    ``None``. Always inlining leaves only three scalar hint values in callers.
    """
    from numba import types

    if isinstance(ilb, types.NoneType):

        def impl(ilb):
            return 0, 0, 0

    else:

        def impl(ilb):
            return int(ilb[0]), int(ilb[1]), int(ilb[2])

    return impl


@register_jitable(**JIT_OPTIONS_INLINE)
def interp3d_locate_point_impl(
    x0: float | np.number,
    x1: float | np.number,
    x2: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None,
    index_out: np.ndarray,
    weight_out: np.ndarray,
) -> None:
    """Locate one point into required length-three output buffers.

    Each axis uses the optimized 1D locator and its independent initial hint. The
    axis order and grouped index-then-weight stores keep the inlined point kernel
    compact; benchmark both scalarized and buffered callers before reordering them.
    """
    ilb0, ilb1, ilb2 = _initial_indices_3d(ilb)
    # Keep the logical axis order for standalone location; fused C evaluation uses
    # a different search order selected for its own generated code.
    index0, weight0 = interp1d_locate_scalar(x0, xp0, ilb0)
    index1, weight1 = interp1d_locate_scalar(x1, xp1, ilb1)
    index2, weight2 = interp1d_locate_scalar(x2, xp2, ilb2)
    # Group contiguous index stores before weight stores. Interleaving these stores
    # lengthens the live ranges differently after this function is inlined.
    index_out[0] = index0
    index_out[1] = index1
    index_out[2] = index2
    weight_out[0] = weight0
    weight_out[1] = weight1
    weight_out[2] = weight2


def interp3d_locate_point(
    x0: float | np.number,
    x1: float | np.number,
    x2: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate one point and allocate only output buffers that were omitted.

    Supplied buffers are forwarded unchanged and returned by identity. Keeping
    allocation in this wrapper leaves :func:`interp3d_locate_point_impl` suitable
    for allocation-free hot loops.
    """
    index = np.empty(3, dtype=np.int64) if index_out is None else index_out
    weight = np.empty(3, dtype=np.float64) if weight_out is None else weight_out
    interp3d_locate_point_impl(x0, x1, x2, xp0, xp1, xp2, ilb, index, weight)
    return index, weight


@register_jitable(**JIT_OPTIONS)
def interp3d_locate_array_impl(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None,
    index_out: np.ndarray,
    weight_out: np.ndarray,
) -> None:
    """Locate equal-shaped coordinate arrays into caller-provided buffers.

    Samples are traversed in flat C order. The interval found for each axis becomes
    that axis's next hint, making locally moving query sequences cheap while the
    1D locator retains logarithmic fallback for distant movement.
    """
    ilb0, ilb1, ilb2 = _initial_indices_3d(ilb)
    for i in range(x0.size):
        # Carry hints independently; coupling them would destroy per-axis locality.
        ilb0, weight0 = interp1d_locate_scalar(x0.flat[i], xp0, ilb0)
        ilb1, weight1 = interp1d_locate_scalar(x1.flat[i], xp1, ilb1)
        ilb2, weight2 = interp1d_locate_scalar(x2.flat[i], xp2, ilb2)
        # Components are interleaved by sample, but index stores remain grouped
        # before weight stores to match the point kernel's compact store order.
        offset = 3 * i
        index_out.flat[offset] = ilb0
        index_out.flat[offset + 1] = ilb1
        index_out.flat[offset + 2] = ilb2
        weight_out.flat[offset] = weight0
        weight_out.flat[offset + 1] = weight1
        weight_out.flat[offset + 2] = weight2


def interp3d_locate_array(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate equal-shaped arrays and allocate omitted result buffers.

    The trailing coordinate dimension stores axes 0, 1, and 2 in that order.
    Caller-provided buffers avoid allocation and are returned without replacement.
    """
    shape = (*x0.shape, 3)
    index = np.empty(shape, dtype=np.int64) if index_out is None else index_out
    weight = np.empty(shape, dtype=np.float64) if weight_out is None else weight_out
    interp3d_locate_array_impl(x0, x1, x2, xp0, xp1, xp2, ilb, index, weight)
    return index, weight


@register_jitable(**JIT_OPTIONS_INLINE)
def interp3d_eval_point(
    index: Sequence[int | np.integer] | np.ndarray,
    weight: Sequence[float | np.number] | np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
) -> float:
    """Evaluate one located point from valid index and weight triples.

    This generic implementation supports arbitrary strides. It caches neighboring
    plane and row views before loading the eight corners, then applies the fixed
    axis-0, axis-1, axis-2 arithmetic tree without relaxed floating-point rules.
    """
    weight0 = weight[0]
    weight1 = weight[1]
    weight2 = weight[2]
    if not extrapolate and (
        weight0 < 0.0
        or weight0 > 1.0
        or weight1 < 0.0
        or weight1 > 1.0
        or weight2 < 0.0
        or weight2 > 1.0
    ):
        return np.nan

    index0 = index[0]
    index1 = index[1]
    index2 = index[2]
    # Construct both planes and all four rows before corner loads. This keeps the
    # generic stride calculations shared and makes axis-2 loads adjacent.
    lower_plane = fp[index0]
    upper_plane = fp[index0 + 1]
    lower_row0 = lower_plane[index1]
    lower_row1 = lower_plane[index1 + 1]
    upper_row0 = upper_plane[index1]
    upper_row1 = upper_plane[index1 + 1]
    lower00 = lower_row0[index2]
    lower01 = lower_row0[index2 + 1]
    lower10 = lower_row1[index2]
    lower11 = lower_row1[index2 + 1]
    upper00 = upper_row0[index2]
    upper01 = upper_row0[index2 + 1]
    upper10 = upper_row1[index2]
    upper11 = upper_row1[index2 + 1]
    # Preserve the arithmetic tree: collapse axis 0, then axis 1, then axis 2.
    # Shared complements reduce repeated work without reassociating operations.
    upper_weight0 = 1.0 - weight0
    value00 = weight0 * lower00 + upper_weight0 * upper00
    value01 = weight0 * lower01 + upper_weight0 * upper01
    value10 = weight0 * lower10 + upper_weight0 * upper10
    value11 = weight0 * lower11 + upper_weight0 * upper11
    upper_weight1 = 1.0 - weight1
    value0 = weight1 * value00 + upper_weight1 * value10
    value1 = weight1 * value01 + upper_weight1 * value11
    value = weight2 * value0 + (1.0 - weight2) * value1
    return float(value)


@register_jitable(**JIT_OPTIONS_INLINE)
def _interp3d_eval_point_c(
    index: Sequence[int | np.integer] | np.ndarray,
    weight: Sequence[float | np.number] | np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
) -> float:
    """Evaluate one point using C-contiguous flat corner addressing.

    A single row-major base offset addresses all eight corners. Corners are loaded
    as adjacent axis-2 pairs from neighboring rows and planes before applying the
    same arithmetic tree as the arbitrary-strided evaluator.
    """
    weight0 = weight[0]
    weight1 = weight[1]
    weight2 = weight[2]
    if not extrapolate and (
        weight0 < 0.0
        or weight0 > 1.0
        or weight1 < 0.0
        or weight1 > 1.0
        or weight2 < 0.0
        or weight2 > 1.0
    ):
        return np.nan

    # Compute multidimensional addressing once. Keep loads in plane/row order so
    # every pair touches adjacent values along the contiguous final axis.
    n1 = fp.shape[1]
    n2 = fp.shape[2]
    plane_size = n1 * n2
    offset = index[0] * plane_size + index[1] * n2 + index[2]
    lower00 = fp.flat[offset]
    lower01 = fp.flat[offset + 1]
    lower10 = fp.flat[offset + n2]
    lower11 = fp.flat[offset + n2 + 1]
    upper00 = fp.flat[offset + plane_size]
    upper01 = fp.flat[offset + plane_size + 1]
    upper10 = fp.flat[offset + plane_size + n2]
    upper11 = fp.flat[offset + plane_size + n2 + 1]
    # Match the generic evaluator exactly; changing association changes results and
    # can also alter register pressure in repeated-field callers.
    upper_weight0 = 1.0 - weight0
    value00 = weight0 * lower00 + upper_weight0 * upper00
    value01 = weight0 * lower01 + upper_weight0 * upper01
    value10 = weight0 * lower10 + upper_weight0 * upper10
    value11 = weight0 * lower11 + upper_weight0 * upper11
    upper_weight1 = 1.0 - weight1
    value0 = weight1 * value00 + upper_weight1 * value10
    value1 = weight1 * value01 + upper_weight1 * value11
    value = weight2 * value0 + (1.0 - weight2) * value1
    return float(value)


@register_jitable(**JIT_OPTIONS)
def interp3d_eval_array_impl(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool,
    out: np.ndarray,
) -> None:
    """Evaluate located arrays into a required sample-shaped output.

    Index and weight buffers use an interleaved trailing coordinate dimension.
    This loop stays out of line to avoid copying corner evaluation into every
    allocating or output-reusing caller.
    """
    for i in range(out.size):
        # Flatten only the compact coordinate buffers; ``fp`` may have any strides.
        offset = 3 * i
        weight0 = weight.flat[offset]
        weight1 = weight.flat[offset + 1]
        weight2 = weight.flat[offset + 2]
        if not extrapolate and (
            weight0 < 0.0
            or weight0 > 1.0
            or weight1 < 0.0
            or weight1 > 1.0
            or weight2 < 0.0
            or weight2 > 1.0
        ):
            out.flat[i] = np.nan
            continue

        # Reject extrapolated samples before loading any function values. Ordered
        # comparisons intentionally leave NaN weights to propagate arithmetically.
        index0 = index.flat[offset]
        index1 = index.flat[offset + 1]
        index2 = index.flat[offset + 2]
        # Collapse the eight corners along axes 0, 1, and 2 in that order.
        value00 = (
            weight0 * fp[index0, index1, index2]
            + (1.0 - weight0) * fp[index0 + 1, index1, index2]
        )
        value01 = (
            weight0 * fp[index0, index1, index2 + 1]
            + (1.0 - weight0) * fp[index0 + 1, index1, index2 + 1]
        )
        value10 = (
            weight0 * fp[index0, index1 + 1, index2]
            + (1.0 - weight0) * fp[index0 + 1, index1 + 1, index2]
        )
        value11 = (
            weight0 * fp[index0, index1 + 1, index2 + 1]
            + (1.0 - weight0) * fp[index0 + 1, index1 + 1, index2 + 1]
        )
        value0 = weight1 * value00 + (1.0 - weight1) * value10
        value1 = weight1 * value01 + (1.0 - weight1) * value11
        out.flat[i] = weight2 * value0 + (1.0 - weight2) * value1


def interp3d_eval_array(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate located arrays and allocate the output only when needed.

    The output drops the trailing length-three coordinate dimension. A supplied
    buffer is forwarded to the out-of-line implementation and returned by identity.
    """
    result = np.empty(index.shape[:-1], dtype=np.float64) if out is None else out
    interp3d_eval_array_impl(index, weight, fp, extrapolate, result)
    return result


@register_jitable(**JIT_OPTIONS_INLINE)
def interp3d_point(
    x0: float | np.number,
    x1: float | np.number,
    x2: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
) -> float:
    """Locate and interpolate one point for arbitrary-strided values.

    The three searches use independent hints in logical axis order. Generic
    multidimensional corner addressing then preserves the same arithmetic tree as
    :func:`interp3d_eval_point` without allocating index or weight containers.
    """
    ilb0, ilb1, ilb2 = _initial_indices_3d(ilb)
    # Keep search order separate from the C-specialized kernel. No one ordering is
    # best for standalone location, generic values, and flat C addressing.
    index0, weight0 = interp1d_locate_scalar(x0, xp0, ilb0)
    index1, weight1 = interp1d_locate_scalar(x1, xp1, ilb1)
    index2, weight2 = interp1d_locate_scalar(x2, xp2, ilb2)

    if not extrapolate and (
        weight0 < 0.0
        or weight0 > 1.0
        or weight1 < 0.0
        or weight1 > 1.0
        or weight2 < 0.0
        or weight2 > 1.0
    ):
        return np.nan

    # Evaluate axis 0 at four corner pairs before reducing axes 1 and 2. Keep this
    # order aligned with the separate evaluation kernels.
    value00 = (
        weight0 * fp[index0, index1, index2]
        + (1.0 - weight0) * fp[index0 + 1, index1, index2]
    )
    value01 = (
        weight0 * fp[index0, index1, index2 + 1]
        + (1.0 - weight0) * fp[index0 + 1, index1, index2 + 1]
    )
    value10 = (
        weight0 * fp[index0, index1 + 1, index2]
        + (1.0 - weight0) * fp[index0 + 1, index1 + 1, index2]
    )
    value11 = (
        weight0 * fp[index0, index1 + 1, index2 + 1]
        + (1.0 - weight0) * fp[index0 + 1, index1 + 1, index2 + 1]
    )
    value0 = weight1 * value00 + (1.0 - weight1) * value10
    value1 = weight1 * value01 + (1.0 - weight1) * value11
    value = weight2 * value0 + (1.0 - weight2) * value1
    return float(value)


@register_jitable(**JIT_OPTIONS_INLINE)
def _interp3d_point_c(
    x0: float | np.number,
    x1: float | np.number,
    x2: float | np.number,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
) -> float:
    """Locate and interpolate one point for C-contiguous values.

    The fastest-moving axis is located first for this fused layout-specialized
    caller. One flattened base offset then addresses all eight corners, avoiding
    repeated multidimensional address calculations.
    """
    ilb0, ilb1, ilb2 = _initial_indices_3d(ilb)
    # Search the contiguous, fast-moving axis first here. This ordering is specific
    # to the fused C path and should only be changed with scalar benchmark evidence.
    index2, weight2 = interp1d_locate_scalar(x2, xp2, ilb2)
    index1, weight1 = interp1d_locate_scalar(x1, xp1, ilb1)
    index0, weight0 = interp1d_locate_scalar(x0, xp0, ilb0)

    if not extrapolate and (
        weight0 < 0.0
        or weight0 > 1.0
        or weight1 < 0.0
        or weight1 > 1.0
        or weight2 < 0.0
        or weight2 > 1.0
    ):
        return np.nan

    # Form one row-major base and keep axis-2 corner accesses adjacent. Direct flat
    # expressions avoid extra address temporaries in this search-heavy fused leaf.
    n1 = fp.shape[1]
    n2 = fp.shape[2]
    plane_size = n1 * n2
    offset = index0 * plane_size + index1 * n2 + index2
    value00 = weight0 * fp.flat[offset] + (1.0 - weight0) * fp.flat[offset + plane_size]
    value01 = (
        weight0 * fp.flat[offset + 1]
        + (1.0 - weight0) * fp.flat[offset + plane_size + 1]
    )
    value10 = (
        weight0 * fp.flat[offset + n2]
        + (1.0 - weight0) * fp.flat[offset + plane_size + n2]
    )
    value11 = (
        weight0 * fp.flat[offset + n2 + 1]
        + (1.0 - weight0) * fp.flat[offset + plane_size + n2 + 1]
    )
    # Preserve the generic arithmetic association after the specialized loads.
    value0 = weight1 * value00 + (1.0 - weight1) * value10
    value1 = weight1 * value01 + (1.0 - weight1) * value11
    value = weight2 * value0 + (1.0 - weight2) * value1
    return float(value)


@register_jitable(**JIT_OPTIONS)
def interp3d_array_impl(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None,
    extrapolate: bool,
    out: np.ndarray,
) -> None:
    """Interpolate equal-shaped coordinate arrays into a required output.

    Each located interval is reused as the next hint on its axis. Separate
    extrapolating and bounded loops hoist the invariant mode branch, while direct
    corner indexing keeps this out-of-line kernel correct for arbitrary strides.
    """
    ilb0, ilb1, ilb2 = _initial_indices_3d(ilb)
    # Version the loop explicitly; relying on Numba to unswitch this branch caused
    # larger generated loops in the established lower-dimensional kernels.
    if extrapolate:
        for i in range(x0.size):
            # Preserve independent stateful hints in logical axis order.
            ilb0, weight0 = interp1d_locate_scalar(x0.flat[i], xp0, ilb0)
            ilb1, weight1 = interp1d_locate_scalar(x1.flat[i], xp1, ilb1)
            ilb2, weight2 = interp1d_locate_scalar(x2.flat[i], xp2, ilb2)

            # Collapse axis 0 first, then axes 1 and 2; do not reassociate this tree.
            value00 = (
                weight0 * fp[ilb0, ilb1, ilb2]
                + (1.0 - weight0) * fp[ilb0 + 1, ilb1, ilb2]
            )
            value01 = (
                weight0 * fp[ilb0, ilb1, ilb2 + 1]
                + (1.0 - weight0) * fp[ilb0 + 1, ilb1, ilb2 + 1]
            )
            value10 = (
                weight0 * fp[ilb0, ilb1 + 1, ilb2]
                + (1.0 - weight0) * fp[ilb0 + 1, ilb1 + 1, ilb2]
            )
            value11 = (
                weight0 * fp[ilb0, ilb1 + 1, ilb2 + 1]
                + (1.0 - weight0) * fp[ilb0 + 1, ilb1 + 1, ilb2 + 1]
            )
            value0 = weight1 * value00 + (1.0 - weight1) * value10
            value1 = weight1 * value01 + (1.0 - weight1) * value11
            out.flat[i] = weight2 * value0 + (1.0 - weight2) * value1
        return

    # The bounded version checks weights before touching any of the eight corners.
    for i in range(x0.size):
        ilb0, weight0 = interp1d_locate_scalar(x0.flat[i], xp0, ilb0)
        ilb1, weight1 = interp1d_locate_scalar(x1.flat[i], xp1, ilb1)
        ilb2, weight2 = interp1d_locate_scalar(x2.flat[i], xp2, ilb2)

        if (
            weight0 < 0.0
            or weight0 > 1.0
            or weight1 < 0.0
            or weight1 > 1.0
            or weight2 < 0.0
            or weight2 > 1.0
        ):
            out.flat[i] = np.nan
            continue

        value00 = (
            weight0 * fp[ilb0, ilb1, ilb2] + (1.0 - weight0) * fp[ilb0 + 1, ilb1, ilb2]
        )
        value01 = (
            weight0 * fp[ilb0, ilb1, ilb2 + 1]
            + (1.0 - weight0) * fp[ilb0 + 1, ilb1, ilb2 + 1]
        )
        value10 = (
            weight0 * fp[ilb0, ilb1 + 1, ilb2]
            + (1.0 - weight0) * fp[ilb0 + 1, ilb1 + 1, ilb2]
        )
        value11 = (
            weight0 * fp[ilb0, ilb1 + 1, ilb2 + 1]
            + (1.0 - weight0) * fp[ilb0 + 1, ilb1 + 1, ilb2 + 1]
        )
        value0 = weight1 * value00 + (1.0 - weight1) * value10
        value1 = weight1 * value01 + (1.0 - weight1) * value11
        out.flat[i] = weight2 * value0 + (1.0 - weight2) * value1


def interp3d_array(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    xp0: np.ndarray,
    xp1: np.ndarray,
    xp2: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate equal-shaped arrays and allocate the output if omitted.

    Allocation remains outside the large numerical loop. Passing ``out`` therefore
    provides an allocation-free wrapper suitable for repeated compiled calls.
    """
    result = np.empty(x0.shape, dtype=np.float64) if out is None else out
    interp3d_array_impl(x0, x1, x2, xp0, xp1, xp2, fp, ilb, extrapolate, result)
    return result
