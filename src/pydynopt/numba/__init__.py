"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from .dummy import (
    boolean,
    float32,
    float64,
    from_dtype,
    int16,
    int32,
    int64,
    int8,
    jit,
    jitclass,
    overload,
    prange,
    register_jitable,
    string,
)

has_numba = False

from pydynopt import use_numba

if use_numba:
    try:
        from numba import jit

        try:
            # Move to numba.experimental in newer Numba releases, try this first
            # to avoid warnings
            from numba.experimental import jitclass
        except ImportError:
            from numba import jitclass
        from numba.extending import overload, register_jitable
        from numba.types import int8, int16, int32, int64
        from numba.types import float32, float64
        from numba.types import boolean, string
        from numba import prange
        from numba import from_dtype

        has_numba = True

        JIT_OPTIONS = {"nopython": True, "nogil": True, "parallel": False}
        JIT_OPTIONS_INLINE = {
            "nopython": True,
            "nogil": True,
            "parallel": False,
            'inline': 'always',
        }

    except ImportError:
        # Nothing to do, use the default decorators defined above
        has_numba = False

        JIT_OPTIONS = {}
        JIT_OPTIONS_INLINE = {}
