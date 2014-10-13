__author__ = 'Richard Foltyn'

try:
    import numba as nb
    from numba import float64 as f8, int32 as i4, void

    JIT_DISABLED = False
except ImportError:
    JIT_DISABLED = True


def njit(*args, **kwargs):
    if not JIT_DISABLED:
        return nb.njit(*args, **kwargs)
    else:
        return lambda func: func