"""Provide checked and unchecked bracketing searches for interpolation.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import JIT_OPTIONS, jit

__all__ = [
    'bsearch',
    'bsearch_impl',
]


@jit(**JIT_OPTIONS)
def bsearch(needle: float | np.number, haystack: np.ndarray, ilb: int = 0) -> int:
    """
    Locate the lower bound index of the bracketing interval containing needle.

    Returns the index ``ilb`` such that
    ``haystack[ilb] <= needle < haystack[ilb + 1]``.
    If ``needle < haystack[0]``, ``0`` is returned.
    If ``needle >= haystack[-1]``, ``len(haystack) - 2`` is returned.
    The grid is checked before entering the minimal search implementation.

    Parameters
    ----------
    needle
        Value to locate in the sorted array.
    haystack
        One-dimensional monotonically increasing array.
    ilb
        Initial guess for the lower bound index of the bracketing interval.

    Returns
    -------
    Index of the lower bound of the bracketing interval.
    """
    if haystack.ndim != 1:
        raise ValueError('haystack must be one-dimensional')

    n = haystack.shape[0]
    if n < 2:
        raise ValueError('haystack must contain at least two values')

    for i in range(n):
        if not np.isfinite(haystack[i]):
            raise ValueError('haystack must contain only finite values')
        if i > 0 and haystack[i] <= haystack[i - 1]:
            raise ValueError('haystack must be strictly increasing')

    ilb_start = max(0, min(ilb, n - 2))
    return bsearch_impl(needle, haystack, ilb_start)


@jit(inline='always', **JIT_OPTIONS)
def bsearch_impl(needle: float | np.number, haystack: np.ndarray, ilb: int = 0) -> int:
    """Locate an interval without validating inputs.

    ``haystack`` must be strictly increasing with at least two values, and ``ilb``
    must be in ``[0, len(haystack) - 2]``.
    """
    n = haystack.shape[0]
    iub = n - 1

    if haystack[ilb] <= needle:
        if haystack[ilb + 1] > needle or ilb == (n - 2):
            return ilb
    else:
        ilb, iub = 0, ilb

    while iub > (ilb + 1):
        imid = (iub + ilb) // 2
        if haystack[imid] > needle:
            iub = imid
        else:
            ilb = imid

    return ilb
