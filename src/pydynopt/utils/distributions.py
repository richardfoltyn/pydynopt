__author__ = 'richard'

import numpy as np

def normprob(vec):
    assert vec.ndim == 1
    # ensure that the input vector is of type float, otherwise this will not
    # work for integer-type arrays
    vec = np.array(vec, dtype=np.float_)
    mu = vec / np.sum(vec)
    assert np.amin(mu) > -1e-12
    mu = np.maximum(mu, 0)
    mu = mu/np.sum(mu)

    return mu
