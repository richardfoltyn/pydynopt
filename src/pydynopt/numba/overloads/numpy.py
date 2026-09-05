"""Provide Numba overloads for partially supported NumPy functions.

- Add axis-aware cumulative sums for two-dimensional arrays.
- Add one-dimensional value insertion for scalar and array indices.

Author: Richard Foltyn

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy import cumsum, insert, typing as npt

from pydynopt.numba import JIT_OPTIONS, jit, overload

__all__ = ['cumsum', 'insert']


def cumsum_dispatch(
    x: npt.NDArray[Any], axis: int | None = None
) -> Callable[..., npt.NDArray[Any]] | None:
    """Select an axis-aware implementation of ``numpy.cumsum``.

    Parameters
    ----------
    x
        Numba array type used to select an implementation.
    axis
        Axis along which to compute the cumulative sum.

    Returns
    -------
    Matching implementation, or ``None`` for unsupported arguments.
    """
    if axis is None:
        return np.cumsum

    if x.ndim == 2:

        def _impl(x: npt.NDArray[Any], axis: int | None = None) -> npt.NDArray[Any]:
            """Compute a cumulative sum along one axis of a 2D array."""
            xout = np.empty_like(x)
            if axis == 0:
                xout[0] = x[0]
                for i in range(1, x.shape[0]):
                    xout[i] = xout[-1] + x[i]
            else:
                # The dispatcher returns this implementation only when an axis
                # is supplied.
                xout[:, 0] = x[:, 0]
                for j in range(1, x.shape[1]):
                    xout[:, j] = xout[:, j - 1] + x[:, j]
            return xout

        return _impl

    return None


def overload_cumsum(jit_options: Mapping[str, Any] | None = None) -> None:
    """Register an axis-aware ``np.cumsum`` overload when needed.

    Parameters
    ----------
    jit_options
        JIT options passed to Numba's ``overload`` decorator.
    """
    try:

        def f(x: npt.NDArray[Any], axis: int) -> npt.NDArray[Any]:
            """Call ``np.cumsum`` with an explicit axis."""
            result = np.cumsum(x, axis=axis)
            return result

        kw: dict[str, Any] = dict(jit_options) if jit_options else JIT_OPTIONS
        kw['nopython'] = True

        fjit = jit(f, **kw)
        fjit(np.zeros((2, 2)), axis=1)
        # A successful compilation means Numba supports the axis argument.
        return
    except Exception:  # noqa: BLE001
        # Any failure means the installed Numba lacks usable axis support.
        # Register the fallback after the feature probe fails.
        overload(np.cumsum, jit_options=jit_options)(cumsum_dispatch)


def _insert(
    arr: npt.NDArray[Any],
    obj: Any,
    values: Any,
    axis: int | None = None,
) -> npt.NDArray[Any]:
    """Insert values before selected indices in a one-dimensional array.

    This mostly NumPy-compatible implementation can be compiled by Numba.

    Parameters
    ----------
    arr
        Input array.
    obj
        Integer index or integer-valued array of indices.
    values
        Values to insert into ``arr``.
    axis
        Ignored and accepted only for API compatibility.

    Returns
    -------
    Copy of ``arr`` with the values inserted.

    Notes
    -----
    - The ``axis`` argument is ignored.
    - Only integer-valued index arrays are supported.
    """
    lobj = np.asarray(obj)
    lvalues = np.asarray(values)

    if lobj.ndim > 1:
        raise ValueError('Unsupported array dimension')

    if lobj.ndim != lvalues.ndim or lobj.size != lvalues.size:
        raise ValueError('Array dimension or shape mismatch')

    N = arr.shape[0]
    Nnew = lobj.size
    Nout = N + Nnew
    out = np.empty(Nout, dtype=arr.dtype)

    indices = np.empty(Nnew, dtype=np.int64)
    indices[:] = lobj
    indices[indices < 0] += N

    # Stable sorting preserves the input order of identical indices.
    iorder = np.argsort(indices, kind='mergesort')
    indices[iorder] += np.arange(Nnew)

    mask_old = np.ones(Nout, dtype=np.bool_)
    mask_old[indices] = False

    out[mask_old] = arr
    out[indices] = lvalues

    return out


@overload(insert, jit_options=JIT_OPTIONS)
def insert_generic(
    arr: Any,
    obj: Any,
    values: Any,
    axis: Any = None,
) -> Callable[..., npt.NDArray[Any]] | None:
    """Select an ``insert`` implementation for Numba argument types.

    Parameters
    ----------
    arr
        Numba type describing the input array.
    obj
        Numba type describing insertion indices.
    values
        Numba type describing values to insert.
    axis
        Numba type describing the axis argument.

    Returns
    -------
    Matching overload implementation, or ``None`` for unsupported types.
    """
    from numba import types

    f = None
    if (isinstance(obj, types.Integer) and isinstance(values, types.Number)) or (  # type: ignore
        isinstance(obj, types.Array)
        and isinstance(values, types.Array)
        and obj.ndim <= 1
    ):
        f = _insert

    return f
