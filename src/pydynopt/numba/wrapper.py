"""
Author: Richard Foltyn
"""

from functools import wraps
from pydynopt import use_numba

import numpy as np

__all__ = ['jit', 'jitclass', 'overload', 'register_jitable',
           'float64', 'int64', 'boolean',
           'prange']


def jit_dummy(signature_or_function=None, *jit_args, **jit_kwargs):
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
        return pyfunc
    else:
        def decorate(func):
            return func

        return decorate


def jitclass_dummy(spec):
    """
    Default implementation of Numba's @jitclass decorator when Numba is
    not available or not desired.
    """
    def decorate(func):
        return func

    return decorate


def overload_dummy(func, jit_options={}, strict=True):
    """
    Default implementation of Numba's @overload decorator when Numba is
    not available or not desired.
    """
    def decorate(func):
        return func

    return decorate


def register_jitable_dummy(*args, **kwargs):
    """
    Register a regular python function that can be executed by the python
    interpreter and can be compiled into a nopython function when referenced
    by other jit'ed functions.  Can be used as::
        @register_jitable
        def foo(x, y):
            return x + y
    Or, with compiler options::
        @register_jitable(_nrt=False) # disable runtime allocation
        def foo(x, y):
            return x + y
    """

    def wrap(fn):
        # It is just a wrapper for @overload
        @overload(fn, jit_options=kwargs, strict=False)
        def ov_wrap(*args, **kwargs):
            return fn

        return fn

    if kwargs:
        return wrap
    else:
        return wrap(*args)


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
boolean = SubscriptableType()

jit = jit_dummy
jitclass = jitclass_dummy
overload = overload_dummy
register_jitable = register_jitable_dummy
prange = range

if use_numba:
    try:
        from numba import jit, jitclass
        from numba.extending import overload, register_jitable
        from numba.types import int64, float64, boolean, string
        from numba import prange
    except ImportError:
        # Nothing to do, use the default decorators defined above
        pass

