__author__ = 'Richard Foltyn'

import numpy as np

from pydynopt.numba import overload

from .linear import interp1d, interp1d_locate, interp1d_eval
from .linear import interp2d, interp2d_locate, interp2d_eval

from .numba.search import bsearch

from .numba.linear import interp1d_locate_scalar, interp1d_locate_array
from .numba.linear import interp1d_eval_scalar, interp1d_eval_array
from .numba.linear import interp1d_scalar, interp1d_array

from .numba.linear import interp2d_locate_scalar, interp2d_locate_array
from .numba.linear import interp2d_eval_scalar, interp2d_eval_array
from .numba.linear import interp2d_scalar, interp2d_array


__all__ = ['bsearch',
           'interp1d', 'interp1d_locate', 'interp1d_eval',
           'interp2d', 'interp2d_locate', 'interp2d_eval']


JIT_OPTIONS = {'parallel': False, 'nogil': True, 'cache': True}


@overload(interp1d, jit_options=JIT_OPTIONS)
def _interp1d_generic(x, xp, fp, ilb=0, extrapolate=True, left=np.nan,
                     right=np.nan, out=None):
    from numba import types
    f = None

    if isinstance(x, types.Number):
        f = interp1d_scalar
    elif isinstance(x, types.Array):
        f = interp1d_array

    return f


@overload(interp1d, jit_options=JIT_OPTIONS)
def _interp1d_impl_generic(x, xp, fp, out, ilb=0, extrapolate=True, left=np.nan,
                           right=np.nan):
    from numba import types
    
    f = None

    from .numba.linear import interp1d_array_impl

    if isinstance(x, types.Number):
        pass
    elif isinstance(x, types.Array):
        f = interp1d_array_impl

    return f


@overload(interp1d_locate, jit_options=JIT_OPTIONS)
def _interp1d_locate_generic(x, xp, ilb=0, index_out=None, weight_out=None):
    from numba import types
    
    f = None

    if isinstance(x, types.Number):
        f = interp1d_locate_scalar
    elif isinstance(x, types.Array):
        f = interp1d_locate_array

    return f


@overload(interp1d_eval, jit_options=JIT_OPTIONS)
def _interp1d_eval_generic(index, weight, fp, extrapolate=True,
                          left=np.nan, right=np.nan, out=None):
    from numba import types
    

    f = None
    if isinstance(index, types.Number):
        f = interp1d_eval_scalar
    elif isinstance(index, types.Array):
        f = interp1d_eval_array

    return f


@overload(interp2d, jit_options=JIT_OPTIONS)
def _interp2d_generic(x0, x1, xp0, xp1, fp, ilb=None, extrapolate=True, out=None):
    from numba import types
    
    f = None

    if isinstance(x0, types.Number):
        f = interp2d_scalar
    elif isinstance(x0, types.Array):
        f = interp2d_array

    return f


@overload(interp2d_locate, jit_options=JIT_OPTIONS)
def _interp2d_locate_generic(x0, x1, xp0, xp1, ilb=None, index_out=None,
                            weight_out=None):
    from numba import types
    

    f = None

    if isinstance(x0, types.Number):
        if ilb is None or index_out is None or weight_out is None:
            f = interp2d_locate_scalar
    elif isinstance(x0, types.Array):
        f = interp2d_locate_array

    return f


@overload(interp2d_locate, jit_options=JIT_OPTIONS)
def _interp2d_locate_impl_generic(x0, x1, xp0, xp1, ilb, index_out, weight_out):
    from numba import types

    from .numba.linear import interp2d_locate_scalar_impl

    f = None

    if isinstance(x0, types.Number):
        f = interp2d_locate_scalar_impl

    return f


@overload(interp2d_eval, jit_options=JIT_OPTIONS)
def _interp2d_eval_generic(index, weight, fp, extrapolate=True, out=None):
    
    from numba import types

    f = None

    # For whatever reason, index might be inferred as optional type, so
    # first recover underlying type if necessary.
    if isinstance(index, types.Optional):
        index = index.type

    if isinstance(index, types.Array) and index.ndim == 1:
        f = interp2d_eval_scalar
    elif isinstance(index, types.Array):
        f = interp2d_eval_array

    return f
