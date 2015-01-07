from cython cimport numeric

ctypedef numeric[:,::1] numc2d_t
ctypedef numeric[::1] numc1d_t

ctypedef fused int_real_t:
    short
    int
    long
    float
    double