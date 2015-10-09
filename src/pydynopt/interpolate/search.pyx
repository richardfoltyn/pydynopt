

cdef struct interp_accel:
    pass

cdef inline size_t \
        c_interp_bsearch(double *xp, double x, size_t lb, size_t ub) nogil:
    cdef size_t midx

    while ub > (lb + 1):
        midx = (ub + lb) // 2
        if xp[midx] > x:
            ub = midx
        else:
            lb = midx

    return lb

cdef inline size_t \
    c_interp_find(double *xp, double x, size_t length, interp_accel *acc) nogil:

    cdef size_t index

    if acc == NULL:
        if x <= xp[0]:
            index = 0
        elif x >= xp[length - 2]:
            index = length - 2
        else:
            index = c_interp_bsearch(xp, x, 0, length - 1)
        return index
    else:
        if x <= xp[0]:
            acc.index = 0
        elif x >= xp[length - 2]:
            acc.index = length - 2
        else:
            index = acc.index
            if x < xp[index]:
                acc.index = c_interp_bsearch(xp, x, 0, index)
            elif x >= xp[index + 1]:
                acc.index = c_interp_bsearch(xp, x, index, length - 1)
        return acc.index
