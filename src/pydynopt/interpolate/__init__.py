__author__ = 'Richard Foltyn'

import numpy as np

from pydynopt.numba import overload
from .linear import interp1d, interp1d_locate, interp1d_eval

from .numba.linear import interp1d_locate_scalar, interp1d_locate_array
from .numba.linear import interp1d_eval_scalar, interp1d_eval_array
from .numba.linear import interp1d_scalar, interp1d_array


@overload(interp1d)
def interp1d_generic(x, xp, fp, extrapolate=True, left=np.nan,
                     right=np.nan, out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array
    f = None

    if isinstance(x, Number):
        f = interp1d_scalar
    elif isinstance(x, Array):
        f = interp1d_array

    return f


@overload(interp1d_locate)
def interp1d_locate_generic(x, xp, ilb=0, index_out=None, weight_out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array
    f = None

    if isinstance(x, Number):
        f = interp1d_locate_scalar
    elif isinstance(x, Array):
        f = interp1d_locate_array

    return f


@overload(interp1d_eval)
def interp1d_eval_generic(index, weight, fp, extrapolate=True,
                          left=np.nan, right=np.nan, out=None):
    from numba.types.scalars import Number
    from numba.types.npytypes import Array

    f = None
    if isinstance(index, Number):
        f = interp1d_eval_scalar
    elif isinstance(index, Array):
        f = interp1d_eval_array

    return f
