"""
    Common type definitions.
"""

ctypedef fused int_real_t:
    short
    int
    long
    float
    double

ctypedef int_real_t[:, ::1] numc2d_t
ctypedef int_real_t[::1] numc1d_t