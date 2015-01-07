__author__ = 'Richard Foltyn'

import numpy as np


def fpbspl(x, t, k):

    h = np.empty((k+1, ))
    hh = np.empty((k, ))

    l = np.sum(t <= x) - 1

    h[0] = 1.0
    for j in range(k):
        for i in range(j):
            hh[i] = h[i]
        h[0] = 0.0
        for i in range(j):
            li = l + i + 1
            lj = li - j
            if t[li] != t[lj]:
                f = hh[i] / (t[li] - t[lj])
                h[i] = h[i] + f * (t[li] - x)
                h[i+1] = f * (x - t[lj])
            else:
                h[i+1] = 0

    return h


def my_fpbspl(x, t, k):
    # bik = np.zeros((2, k+1), dtype=np.double)
    x = np.atleast_1d(x).ravel()
    t = np.atleast_1d(t).ravel()

    out = np.zeros((x.shape[0], t.shape[0]))

    l = np.sum(t <= x.reshape((-1, 1)), axis=1) - 1

    b_old = np.zeros((t.shape[0], ), dtype=np.double)
    b_new = np.zeros_like(b_old)

    for m in range(x.shape[0]):
        b_old[...] = 0
        b_old[l[m]] = 1
        xi = x[m]
        for ik in range(1, k+1):
            for i in range(t.shape[0] - (k+1)):
                a = (xi-t[i]) / (t[i+ik] - t[i]) if t[i+ik] != t[i] else 0
                b = (t[i+ik-1] - xi)/(t[i+ik+1] - t[i+1]) \
                    if t[i+ik+1] != t[i+1] else 0

                b_new[i] = a * b_old[i] + b * b_old[i+1]

        out[m, :] = b_new

    return out

def my_spevl(x, t, c, k):
    bspl = my_fpbspl(x, t, k)
    print(bspl)
    return bspl.dot(c)