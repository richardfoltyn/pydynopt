import numpy as np
from numpy import pad

from ..common import ConvergenceError


def rouwenhorst(n, mu, rho, sigma):
    """
    Code to approximate an AR(1) process using the Rouwenhorst method as in
    Kopecky & Suen, Review of Economic Dynamics (2010), Vol 13, pp. 701-714
    Adapted from Matlab code by Martin Floden.
    """

    p = (1+rho)/2
    Pi = np.array([[p, 1-p], [1-p, p]])

    for i in range(Pi.shape[0], n):
        tmp = pad(Pi, 1, mode='constant', constant_values=0)
        Pi = p * tmp[1:, 1:] + (1-p) * tmp[1:, :-1] + \
             (1-p) * tmp[:-1, 1:] + p * tmp[:-1, :-1]
        Pi[1:-1, :] /= 2

    fi = np.sqrt(n-1) * sigma / np.sqrt(1 - rho ** 2)
    z = np.linspace(-fi, fi, n) + mu

    return z, Pi


def markov_ergodic_dist(transm, tol=1e-12, maxiter=10000, transpose=True, mu0=None):

    # This function should also work for sparse matrices from scipy.sparse,
    # so do not use .T to get the transpose.
    if transpose:
        transm = transm.transpose()

    assert np.all(np.abs(transm.sum(axis=0) - 1) < 1e-12)

    if mu0 is None:
        # start out with uniform distribution
        mu0 = np.ones((transm.shape[0], ), dtype=np.float64)/transm.shape[0]

    for it in range(maxiter):
        mu1 = transm.dot(mu0)

        dv = np.max(np.abs(mu0 - mu1))
        if dv < tol:
            return mu1/(np.sum(mu1))
        else:
            mu0 = mu1
    else:
        print('Failed to converge after %d iterations (delta = %e)' %
              (it, dv))
        raise ConvergenceError(it, dv)


def markov_moments(states, transm, ergodic_dist=None):

    if ergodic_dist is None:
        ergodic_dist = markov_ergodic_dist(transm)

    x = states

    mean_uncond = np.dot(ergodic_dist, x)
    var_uncond = np.dot(np.power(x - mean_uncond, 2), ergodic_dist)
    x_demeaned = x - mean_uncond
    x_m1 = np.outer(x_demeaned, x_demeaned)
    wgt = transm * ergodic_dist.reshape((-1, 1))

    # Autocovariance
    autocov = np.sum(np.sum(x_m1 * wgt))

    # implied autocorrelation and variance of error term of discretized
    # process
    autocorr = autocov / var_uncond
    sigma_e = np.sqrt((1-autocorr**2) * var_uncond)
    sigma_x = np.sqrt(var_uncond)

    return autocorr, sigma_x, sigma_e
