"""Expose low-level Numba-compatible optimization kernels."""

from .common import nderiv_array, nderiv_scalar
from .zeros import (
    newton_bisect_callable_full,
    newton_bisect_callable_simple,
    newton_bisect_full,
    newton_bisect_impl,
    newton_bisect_simple,
)

__all__ = [
    'nderiv_array',
    'nderiv_scalar',
    'newton_bisect_callable_full',
    'newton_bisect_callable_simple',
    'newton_bisect_full',
    'newton_bisect_impl',
    'newton_bisect_simple',
]
