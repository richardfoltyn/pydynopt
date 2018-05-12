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


def markov_ergodic_dist(transm, tol=1e-12, maxiter=10000, transpose=True,
                        mu0=None, inverse=False):
    """
    Compute the ergodic distribution implied by a given Makrov chain transition
    matrix.

    Parameters
    ----------
    transm : numpy.ndarray
        Markov chain transition matrix
    tol : float
        Terminal tolerance on consecutive changes in the ergodic distribution
        if computing via the iterative method (`inverse` = False)
    maxiter : int
        Maximum number of iterations for iterative method.
    transpose : bool
        If true, the transition matrix `transm` is provided in transposed form.
    mu0 : numpy.ndarray
        Optional initial guess for the ergodic distribution if the iterative
        method is used (default: uniform distribution).
    inverse : bool
        If True, compute the erdogic distribution using the "inverse" method

    Returns
    -------
    mu : numpy.ndarray
        Ergodic distribution
    """

    # This function should also work for sparse matrices from scipy.sparse,
    # so do not use .T to get the transpose.
    if transpose:
        transm = transm.transpose()

    assert np.all(np.abs(transm.sum(axis=0) - 1) < 1e-12)

    if not inverse:
        if mu0 is None:
            # start out with uniform distribution
            mu0 = np.ones((transm.shape[0], ), dtype=np.float64)/transm.shape[0]

        for it in range(maxiter):
            mu1 = transm.dot(mu0)

            dv = np.max(np.abs(mu0 - mu1))
            if dv < tol:
                mu = mu1 / np.sum(mu1)
                break
            else:
                mu0 = mu1
        else:
            msg = 'Failed to converge (delta = {:.e})'.format(dv)
            print(msg)
            raise ConvergenceError(it, dv)
    else:
        m = transm - np.identity(transm.shape[0])
        m[-1] = 1
        m = np.linalg.inv(m)
        mu = np.ascontiguousarray(m[:, -1])
        assert np.abs(np.sum(mu) - 1) < 1e-9
        mu /= np.sum(mu)

    return mu


def markov_moments(states, transm, ergodic_dist=None, moments=False,
                   *args, **kwargs):
    """
    Computes (exact) moments implied my a given Markov process, including
    the autocorrelation and the unconditional and conditional standard
    deviation.

    Parameters
    ----------
    states : array_like
        State space of the Markov chain
    transm : numpy.ndarray
        Transition matrix of the Markov chain
    ergodic_dist : array_like or None
        Optional ergodic distribution implied by the transition matrix. If
        None this will be computed.
    moments : bool
        If True, additionally return array containing the usual first four
        unconditional (standardized) moments.

    Returns
    -------
    autocorr : float
        First-order autocorrelation
    sigma_x : float
        Unconditional standard deviation
    sigma_e : float
        Conditional standard deviation
    mom : numpy.ndarray
        If `moments` is True, returns array containing the following
        (unconditional) moments: [mean, variance, skewness, kurtosis]
    """

    if ergodic_dist is None:
        ergodic_dist = markov_ergodic_dist(transm, *args, **kwargs)
    else:
        ergodic_dist = np.atleast_1d(ergodic_dist)

    x = np.atleast_1d(states)

    # unconditional centered moments
    # include the 0-th moment so that m[i] gives the i-th moment
    m = np.zeros(5)
    m[1] = np.dot(ergodic_dist, x)
    m[2] = np.dot(np.power(x - m[1], 2), ergodic_dist)
    # Compute skewness and kurtosis only if requested
    if moments:
        for k in range(3, 5):
            m[k] = np.dot(np.power(x - m[1], k), ergodic_dist) / m[2] ** (k/2.0)

    x_demeaned = x - m[1]
    x_m1 = np.outer(x_demeaned, x_demeaned)
    wgt = transm * ergodic_dist.reshape((-1, 1))

    # Autocovariance
    autocov = np.sum(np.sum(x_m1 * wgt))

    # implied autocorrelation and variance of error term of discretized
    # process
    autocorr = autocov / m[2]
    sigma_e = np.sqrt((1-autocorr**2) * m[2])
    sigma_x = np.sqrt(m[2])

    if moments:
        # do not return the 0-th moment
        return autocorr, sigma_x, sigma_e, m[1:]
    else:
        return autocorr, sigma_x, sigma_e
