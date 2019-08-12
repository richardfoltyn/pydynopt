__author__ = 'Richard Foltyn'

import numpy as np

from pydynopt.numba import overload
from .linear import interp1d, interp1d_locate, interp1d_eval
from .linear import interp2d, interp2d_locate, interp2d_eval

from .numba.linear import interp1d_locate_scalar, interp1d_locate_array
from .numba.linear import interp1d_eval_scalar, interp1d_eval_array
from .numba.linear import interp1d_scalar, interp1d_array

from .numba.linear import interp2d_locate_scalar, interp2d_locate_array
from .numba.linear import interp2d_eval_scalar, interp2d_eval_array
from .numba.linear import interp2d_scalar, interp2d_array


__all__ = ['interp1d', 'interp1d_locate', 'interp1d_eval',
           'interp2d', 'interp2d_locate', 'interp2d_eval']


@overload(interp1d, jit_options={'parallel': False})
def _interp1d_generic(x, xp, fp, extrapolate=True, left=np.nan,
                     right=np.nan, out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array
    f = None

    if isinstance(x, Number):
        f = interp1d_scalar
    elif isinstance(x, Array):
        f = interp1d_array

    return f


@overload(interp1d_locate, jit_options={'parallel': False})
def _interp1d_locate_generic(x, xp, ilb=0, index_out=None, weight_out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array
    f = None

    if isinstance(x, Number):
        f = interp1d_locate_scalar
    elif isinstance(x, Array):
        f = interp1d_locate_array

    return f


@overload(interp1d_eval, jit_options={'parallel': False})
def _interp1d_eval_generic(index, weight, fp, extrapolate=True,
                          left=np.nan, right=np.nan, out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array

    f = None
    if isinstance(index, Number):
        f = interp1d_eval_scalar
    elif isinstance(index, Array):
        f = interp1d_eval_array

    return f


@overload(interp2d, jit_options={'parallel': False})
def _interp2d_generic(x0, x1, xp0, xp1, fp, extrapolate=True, out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array
    f = None

    if isinstance(x0, Number):
        f = interp2d_scalar
    elif isinstance(x0, Array):
        f = interp2d_array

    return f


@overload(interp2d_locate, jit_options={'parallel': False})
def _interp2d_locate_generic(x0, x1, xp0, xp1, ilb=None, index_out=None,
                            weight_out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array
    f = None

    if isinstance(x0, Number):
        f = interp2d_locate_scalar
    elif isinstance(x0, Array):
        f = interp2d_locate_array

    return f


@overload(interp2d_eval, jit_options={'parallel': False})
def _interp2d_eval_generic(index, weight, fp, extrapolate=True, out=None):
    from numba.types.npytypes import Array
    from numba.types import Optional

    f = None

    # For whatever reason, index might be inferred as optionl type, so
    # first recover underlying type if necessary.
    if isinstance(index, Optional):
        index = index.type

    if isinstance(index, Array) and index.ndim == 1:
        f = interp2d_eval_scalar
    elif isinstance(index, Array):
        f = interp2d_eval_array

    return f
