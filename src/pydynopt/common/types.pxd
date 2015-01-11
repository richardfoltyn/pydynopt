"""
    Common type definitions.
"""

ctypedef fused int_real_t:
    short
    unsigned short
    int
    unsigned int
    long
    unsigned long
    float
    double

ctypedef int_real_t[:, ::1] num_arr2d_t
ctypedef int_real_t[::1] num_arr1d_t