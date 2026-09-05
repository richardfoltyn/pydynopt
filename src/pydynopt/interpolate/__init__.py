"""Provide checked linear interpolation functions for Python and Numba.

- One-dimensional interpolation requires equal-shaped, one-dimensional ``xp``
  and ``fp`` arrays.
- Scalar 1D queries return floats; array and sequence queries return arrays and
  support validated output buffers.
- Two-dimensional coordinates follow NumPy broadcasting in Python, and ``fp``
  must have shape ``(len(xp0), len(xp1))``.
- Locate and evaluate operations are available separately from combined
  interpolation.

The same six functions can be called from ordinary Python and Numba-compiled code.
Low-level kernels live in the ``pydynopt.interpolate.numba`` submodules.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from .linear import (
    interp1d,
    interp1d_eval,
    interp1d_locate,
    interp2d,
    interp2d_eval,
    interp2d_locate,
)

__all__ = [
    'interp1d',
    'interp1d_eval',
    'interp1d_locate',
    'interp2d',
    'interp2d_eval',
    'interp2d_locate',
]
