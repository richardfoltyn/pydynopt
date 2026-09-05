"""
Routines for creating, transforming, and indexing multidimensional arrays.

- Grid generation functions (power-spaced, log-spaced)
- Coordinate and linear index conversion (ind2sub, sub2ind)
- Probability clipping routines

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
