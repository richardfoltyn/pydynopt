"""
Author: Richard Foltyn
"""

from functools import wraps
from pydynopt import use_numba

import numpy as np

def jit(signature_or_function=None, *jit_args, **jit_kwargs):
    """
    Default implementation of Numba's @jit decorator when Numba is not
    available or not desired.
    """
    if signature_or_function is None:
        pyfunc = None
    elif isinstance(signature_or_function, list):
        pyfunc = None
    else:
        pyfunc = signature_or_function

    if pyfunc is not None:
        @wraps(pyfunc)
        def wrapper(*args, **kwargs):
            return pyfunc(*args, **kwargs)
        return wrapper
    else:
        def decorate(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorate


def jitclass(spec):
    """
    Default implementation of Numba's @jitclass decorator when Numba is
    not available or not desired.
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorate


def overload(func, *overload_args, **overload_kwargs):
    """
    Default implementation of Numba's @overload decorator when Numba is
    not available or not desired.
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorate


class SubscriptableType(np.int64):
    """
    Dummy type that servers as default drop-in for Numba's data types.
    Note: Needs __getitem__ method so that statements such as float64[::1]
    are valid.
    """
    def __getitem__(self, item):
        return self


int64 = SubscriptableType()
float64 = SubscriptableType()

if use_numba:
    try:
        from numba import jit, jitclass
        from numba.extending import overload
        from numba.types import int64, float64
    except ImportError:
        # Nothing to do, use the default decorators defined above
        pass

