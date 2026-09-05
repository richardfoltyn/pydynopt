"""Provide checked array creation and probability manipulation functions.

- Create power-spaced and logarithmically spaced one-dimensional grids.
- Clip scalar, sequence, and array probabilities through shared Numba kernels.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence
from math import log
from typing import Any, overload as typing_overload

import numpy as np
from numpy.typing import NDArray

from pydynopt.numba import JIT_OPTIONS, jit, overload as numba_overload

from .numba.arrays import (
    clip_prob_array,
    clip_prob_array_impl,
    clip_prob_scalar,
    powerspace_impl,
)

__all__ = [
    'clip_prob',
    'logspace',
    'powerspace',
]

type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type RealArrayInput = Sequence[RealScalar] | np.ndarray
type FloatArray = NDArray[np.float64]

_clip_prob_scalar_jit = jit(clip_prob_scalar, **JIT_OPTIONS)
_powerspace_jit = jit(powerspace_impl, **JIT_OPTIONS)


def _normalize_real_scalar(value: object, name: str) -> float:
    """Normalize a real scalar to a Python ``float``."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'iuf':
        msg = f'{name} must be a real scalar'
        raise TypeError(msg)
    return float(array.item())


def _normalize_count(value: object, name: str, minimum: int) -> int:
    """Normalize a non-boolean integer count and enforce its lower bound."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'iu':
        msg = f'{name} must be an integer'
        raise TypeError(msg)
    count = int(array.item())
    if count < minimum:
        msg = f'{name} must be at least {minimum}'
        raise ValueError(msg)
    return count


def _validate_float_output(out: np.ndarray, shape: tuple[int, ...]) -> None:
    """Validate a writable output that safely represents ``float64`` values."""
    if not isinstance(out, np.ndarray):
        msg = 'out must be a NumPy array'
        raise TypeError(msg)
    if out.shape != shape:
        msg = f'out must have shape {shape}, got {out.shape}'
        raise ValueError(msg)
    if out.dtype.kind != 'f':
        msg = 'out must have a floating dtype'
        raise TypeError(msg)
    if not np.can_cast(np.dtype(np.float64), out.dtype, casting='safe'):
        msg = f'out dtype {out.dtype} cannot safely represent float64'
        raise TypeError(msg)
    if not out.flags.writeable:
        msg = 'out must be writable'
        raise ValueError(msg)


@typing_overload
def clip_prob(
    value: RealScalar,
    tol: RealScalar,
    out: None = None,
) -> float: ...


@typing_overload
def clip_prob(
    value: RealArrayInput,
    tol: RealScalar,
    out: FloatArray | None = None,
) -> FloatArray: ...


def clip_prob(
    value: RealScalar | RealArrayInput,
    tol: RealScalar,
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Clip probabilities close to zero or one.

    Parameters
    ----------
    value
        Value or values to clip. Scalars follow the scalar return path; sequences
        and arrays follow the array return path.
    tol
        Finite tolerance satisfying ``0 <= tol <= 0.5``. Values strictly below
        ``tol`` become zero and values strictly above ``1 - tol`` become one.
    out
        Optional output for the array path. It must match the input shape, be
        writable, and safely represent ``float64`` values. The scalar path does
        not support an output buffer.

    Returns
    -------
    A built-in ``float`` for scalar input, a newly allocated ``float64`` array for
    array input, or the supplied output buffer by identity.

    Raises
    ------
    TypeError
        If an input is not real or the scalar path receives ``out``.
    ValueError
        If ``tol`` or an output buffer is invalid.
    """
    tolerance = _normalize_real_scalar(tol, 'tol')
    if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance > 0.5:
        msg = 'tol must satisfy 0 <= tol <= 0.5'
        raise ValueError(msg)

    scalar = np.isscalar(value) and not isinstance(value, np.ndarray)
    if scalar:
        if out is not None:
            msg = 'scalar clip_prob calls do not accept an output buffer'
            raise TypeError(msg)
        item = _normalize_real_scalar(value, 'value')
        return float(_clip_prob_scalar_jit(item, tolerance))

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        msg = 'value must contain real numbers'
        raise TypeError(msg) from exc
    if array.dtype.kind not in 'iuf':
        msg = 'value must contain real numbers'
        raise TypeError(msg)

    work = np.asarray(array, dtype=np.float64)
    if work.ndim > 0:
        work = np.ascontiguousarray(work)
    if out is None:
        result = np.empty(array.shape, dtype=np.float64)
    else:
        _validate_float_output(out, array.shape)
        result = out
    clip_prob_array_impl(work, tolerance, result)
    return result


def powerspace(
    xmin: RealScalar,
    xmax: RealScalar,
    n: int,
    exponent: RealScalar,
) -> FloatArray:
    """Create a power-spaced one-dimensional grid.

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
    A ``float64`` grid. Increasing boundaries preserve their argument order;
    decreasing boundaries preserve the historical flipped, increasing output.

    Raises
    ------
    TypeError
        If ``n`` is not integral or another argument is not a real scalar.
    ValueError
        If ``n`` or ``exponent`` is outside its valid range.
    """
    lower = _normalize_real_scalar(xmin, 'xmin')
    upper = _normalize_real_scalar(xmax, 'xmax')
    count = _normalize_count(n, 'n', 1)
    power = _normalize_real_scalar(exponent, 'exponent')
    if not np.isfinite(power) or power <= 0.0:
        msg = 'exponent must be finite and strictly positive'
        raise ValueError(msg)
    return _powerspace_jit(lower, upper, count, power)


def logspace(
    start: float,
    stop: float,
    num: int,
    log_shift: float = 0.0,
    x0: float | None = None,
    frac_at_x0: float | None = None,
    insert_vals: Sequence[float] | np.ndarray | float | None = None,
) -> FloatArray:
    """Create a Python-only grid that is uniform in shifted logarithms.

    Parameters
    ----------
    start
        Finite lower endpoint.
    stop
        Finite upper endpoint, strictly greater than ``start``.
    num
        Total number of returned points, including inserted values.
    log_shift
        Finite shift for which ``start + log_shift`` is strictly positive.
    x0
        Optional reference point strictly inside the domain. It is used when
        ``frac_at_x0`` is supplied and defaults to the domain midpoint.
    frac_at_x0
        Fraction in ``(0, 1)`` used to solve for ``log_shift``.
    insert_vals
        Finite interior values to insert in sorted order. At least two generated
        points must remain for the endpoints.

    Returns
    -------
    A ``float64`` array of length ``num`` with exact requested endpoints.

    Raises
    ------
    TypeError
        If an argument cannot be interpreted as the annotated scalar or sequence.
    ValueError
        If the domain, spacing controls, or insertion count is invalid.

    Notes
    -----
    This function uses SciPy root finding when ``frac_at_x0`` is specified and is
    intentionally unavailable from Numba-compiled code.
    """
    from scipy.optimize import brentq

    lower = _normalize_real_scalar(start, 'start')
    upper = _normalize_real_scalar(stop, 'stop')
    count = _normalize_count(num, 'num', 2)
    shift = _normalize_real_scalar(log_shift, 'log_shift')
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        msg = 'start and stop must be finite and satisfy start < stop'
        raise ValueError(msg)
    if not np.isfinite(shift):
        msg = 'log_shift must be finite'
        raise ValueError(msg)

    inserted: np.ndarray | None = None
    if insert_vals is not None:
        try:
            inserted_raw = np.asarray(insert_vals)
        except (TypeError, ValueError) as exc:
            msg = 'insert_vals must contain real values'
            raise TypeError(msg) from exc
        if inserted_raw.ndim > 1 or inserted_raw.dtype.kind not in 'iuf':
            msg = 'insert_vals must be a scalar or one-dimensional real sequence'
            raise TypeError(msg)
        inserted = np.atleast_1d(inserted_raw).astype(np.float64, copy=False)
        if not np.all(np.isfinite(inserted)):
            msg = 'insert_vals must contain only finite values'
            raise ValueError(msg)
        if inserted.size > count - 2:
            msg = 'insert_vals leaves fewer than two generated grid points'
            raise ValueError(msg)
        if np.any(inserted <= lower) or np.any(inserted >= upper):
            msg = 'insert_vals must lie strictly between start and stop'
            raise ValueError(msg)
        inserted = np.sort(inserted)

    if frac_at_x0 is not None:
        frac = _normalize_real_scalar(frac_at_x0, 'frac_at_x0')
        if not np.isfinite(frac) or frac <= 0.0 or frac >= 1.0:
            msg = 'frac_at_x0 must lie strictly between zero and one'
            raise ValueError(msg)

        reference = (
            (upper + lower) / 2.0 if x0 is None else _normalize_real_scalar(x0, 'x0')
        )
        if not np.isfinite(reference) or reference <= lower or reference >= upper:
            msg = 'x0 must lie strictly between start and stop'
            raise ValueError(msg)

        def fobj(candidate: float) -> float:
            dist = np.log(upper + candidate) - np.log(lower + candidate)
            value = (
                np.log(reference + candidate) - np.log(lower + candidate) - frac * dist
            )
            return float(value)

        lb = -lower + 1.0e-12
        ub = upper - lower
        for _ in range(10):
            if fobj(ub) < 0.0:
                break
            ub *= 10.0
        else:
            msg = (
                f'cannot find grid spacing for x0={reference:g} and frac_at_x0={frac:g}'
            )
            raise ValueError(msg)
        shift = float(brentq(fobj, lb, ub))

    if lower + shift <= 0.0 or upper + shift <= 0.0:
        msg = 'start + log_shift and stop + log_shift must be strictly positive'
        raise ValueError(msg)

    generated = count if inserted is None else count - inserted.size
    lstart = log(lower + shift)
    lstop = log(upper + shift)
    grid = np.exp(np.linspace(lstart, lstop, generated)) - shift

    if inserted is not None and inserted.size > 0:
        idx = np.searchsorted(grid, inserted)
        grid = np.insert(grid, idx, inserted)

    grid[0] = lower
    grid[-1] = upper
    return grid


def _numba_none(value: Any) -> bool:
    """Return whether a Numba overload argument represents ``None``."""
    from numba import types

    return value is None or isinstance(value, (types.NoneType, types.Omitted))


def _numba_real_scalar(value: Any) -> bool:
    """Return whether a Numba type is a real scalar."""
    from numba import types

    return value in types.integer_domain or value in types.real_domain


def _numba_real_array(value: Any) -> bool:
    """Return whether a Numba type is a real numeric array."""
    from numba import types

    return isinstance(value, types.Array) and (
        value.dtype in types.integer_domain or value.dtype in types.real_domain
    )


def _numba_float_output(value: Any, ndim: int) -> bool:
    """Return whether a Numba output is omitted or writable ``float64``."""
    from numba import types

    if _numba_none(value):
        return True
    return (
        isinstance(value, types.Array)
        and value.dtype == types.float64
        and value.ndim == ndim
        and value.mutable
    )


@numba_overload(clip_prob, jit_options=JIT_OPTIONS)
def _overload_clip_prob(
    value: Any,
    tol: Any,
    out: Any = None,
) -> Any:
    """Select the scalar or array clipping implementation for Numba."""
    if not _numba_real_scalar(tol):
        return None
    if _numba_real_scalar(value):
        if not _numba_none(out):
            return None

        def impl(value, tol, out=None):
            return clip_prob_scalar(value, tol)

        return impl
    if _numba_real_array(value):
        if not _numba_float_output(out, value.ndim):
            return None
        return clip_prob_array
    return None


@numba_overload(powerspace, jit_options=JIT_OPTIONS)
def _overload_powerspace(
    xmin: Any,
    xmax: Any,
    n: Any,
    exponent: Any,
) -> Any:
    """Select the power-grid implementation for supported Numba scalars."""
    from numba import types

    if (
        _numba_real_scalar(xmin)
        and _numba_real_scalar(xmax)
        and n in types.integer_domain
        and _numba_real_scalar(exponent)
    ):
        return powerspace_impl
    return None
