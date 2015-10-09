from pydynopt.interpolate.search cimport interp_accel, c_interp_find
from pydynopt.interpolate.linear cimport cy_find_lb

import numpy as np
from time import perf_counter

def bench_linear(size_t times = int(1e6), size_t length = 10,
                 size_t key_length = 10):

    cdef double[::1] xp = np.sort(np.random.rand(length))
    cdef double[::1] keys = np.sort(np.random.choice(xp, key_length))

    cdef interp_accel acc
    cdef size_t[::1] res1, res2

    res1 = np.empty(key_length, dtype=np.uint)
    res2 = np.empty(key_length, dtype=np.uint)

    cdef size_t i, j

    t0 = perf_counter()
    for i in range(times):
        for j in range(key_length):
            res1[j] = cy_find_lb(&xp[0], keys[j], length)

    t1 = perf_counter()
    print('Time elapsed: {:e}'.format(t1-t0))

    t0 = perf_counter()
    for i in range(times):
        # re-initialize accelerator
        # acc.index = length // 2
        for j in range(key_length):
            res2[j] = c_interp_find(&xp[0], keys[j], length, NULL)

    t1 = perf_counter()
    print('Time elapsed: {:e}'.format(t1 - t0))

    diff = np.max(np.abs(np.asarray(res1) - np.asarray(res2)))
    print('Sup. norm: {:e}'.format(diff))


