"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import numpy as np

__all__ = [
    'jit',
    'jitclass',
    'overload',
    'register_jitable',
    'float32',
    'float64',
    'int8',
    'int16',
    'int32',
    'int64',
    'boolean',
    'string',
    'prange',
    'from_dtype',
]


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

    return wrap


def from_dtype(obj):
    return obj


class SubscriptableType:
    """
    Dummy type that serves as the default drop-in for Numba's data types.
    Note: Needs __getitem__ method so that statements such as float64[::1]
    are valid.
    """

    def __getitem__(self, item):
        return self


int8 = SubscriptableType()
int16 = SubscriptableType()
int32 = SubscriptableType()
int64 = SubscriptableType()
float32 = SubscriptableType()
float64 = SubscriptableType()
boolean = SubscriptableType()
string = str

jit = jit_dummy
jitclass = jitclass_dummy
overload = overload_dummy
register_jitable = register_jitable_dummy
prange = range
