"""Provide checked array utilities for ordinary Python and Numba.

- ``clip_prob``, ``powerspace``, ``ind2sub``, and ``sub2ind`` share low-level
  kernels between Python and Numba-compiled callers.
- ``logspace`` remains Python-only because it uses SciPy root finding and dynamic
  insertion.
- Index conversion follows C order and uses dimension-first coordinates.

Low-level kernels remain in ``pydynopt.arrays.numba`` implementation modules.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from .base import clip_prob, logspace, powerspace
from .indexing import ind2sub, sub2ind

__all__ = [
    'clip_prob',
    'ind2sub',
    'logspace',
    'powerspace',
    'sub2ind',
]
