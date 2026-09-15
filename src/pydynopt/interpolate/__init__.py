"""Provide checked linear interpolation functions for Python and Numba.

- One-dimensional interpolation requires equal-shaped, one-dimensional ``xp``
  and ``fp`` arrays.
- Scalar 1D queries return floats; array and sequence queries return arrays and
  support validated output buffers.
- Multi-dimensional coordinates follow NumPy broadcasting in Python, and ``fp``
  must conform to the corresponding interpolation grids.
- Locate and evaluate operations are available separately from combined
  interpolation.
- Length-two and length-three tuple inputs to ``interp2d_eval`` and
  ``interp3d_eval`` avoid temporary index and weight arrays when a Numba kernel
  evaluates several fields at the same coordinates.

The same public functions can be called from ordinary Python and Numba-compiled code.
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
    interp3d,
    interp3d_eval,
    interp3d_locate,
)

__all__ = [
    'interp1d',
    'interp1d_eval',
    'interp1d_locate',
    'interp2d',
    'interp2d_eval',
    'interp2d_locate',
    'interp3d',
    'interp3d_eval',
    'interp3d_locate',
]
