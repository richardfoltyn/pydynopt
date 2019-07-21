"""
Author: Richard Foltyn
"""

from functools import wraps

try:
    from numba import jit, jitclass
    from numba.extending import overload
    from numba.types import int64, float64
except ImportError:
    # Create a custom decorator that does nothing
    def overload(*args, **kwargs):
        def decorate(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorate

    def jit(*args, **kwargs):
        def decorate(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorate

    def jitclass(*args, **kwargs):
        def decorate(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorate

    int64 = int
    float64 = float

