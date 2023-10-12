"""
Overloads for NumPy functions not (fully) supported by Numba. This includes functions
which are only supported for a sub-set of arguments.

Author: Richard Foltyn

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/
"""
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np


def cumsum_dispatch(x: np.ndarray, axis: Optional[int] = None) -> Union[callable, None]:
    """
    Overload for numpy.cumsum() with second argument (axis) given, which is not
    supported by Numba.

    Parameters
    ----------
    x : np.ndarray
    axis : int, optional

    Returns
    -------
    callable or None
    """

    if axis is None:
        return np.cumsum
    elif x.ndim == 2 and axis is not None:

        def _impl(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
            xout = np.empty_like(x)
            if axis == 0:
                xout[0] = x[0]
                for i in range(1, x.shape[0]):
                    xout[i] = xout[-1] + x[i]
            else:
                # Note that function is guaranteed to be called only of axis is not
                # None, otherwise the dispatcher does not return it for the given
                # arguments.
                xout[:, 0] = x[:, 0]
                for j in range(1, x.shape[1]):
                    xout[:, j] = xout[:, j - 1] + x[:, j]
            return xout

        return _impl


def overload_cumsum(jit_options: Optional[Mapping] = None):
    """
    Overload np.cumsum() for 2d arrays if current version of Numba does not support
    axis argument.

    Parameters
    ----------
    jit_options : mapping
        JIT options passed to Numba's overload()

    """
    from numba import jit
    from numba.extending import overload

    try:

        def f(x, axis):
            return np.cumsum(x, axis=axis)

        kw = dict(jit_options) if jit_options else dict()
        kw["nopython"] = True

        fjit = jit(f, **kw)
        fjit(np.zeros((2, 2)), axis=1)
        # If this compiles and runs, current Numba version supports cumsum with axis
        # argument. Nothing else needs to be done
        return
    except:
        # Ignore error, return custom implementation
        pass

    overload(np.cumsum, jit_options=jit_options)(cumsum_dispatch)
