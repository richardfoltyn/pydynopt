from __future__ import absolute_import, division, print_function

__author__ = 'richard'

import numpy as np


def normprob(vec):
    assert vec.ndim == 1
    # ensure that the input vector is of type float, otherwise this will not
    # work for integer-type arrays
    vec = np.asarray(vec, dtype=np.float)
    mu = vec / np.sum(vec)
    assert np.amin(mu) > -1e-12
    mu = np.maximum(mu, 0)
    mu = mu/np.sum(mu)

    return mu
