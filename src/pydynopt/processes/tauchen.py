__author__ = 'Richard Foltyn'

from pydynopt.processes import markov_ergodic_dist
from pydynopt.processes import markov_moments

import numpy as np
from scipy.stats import norm


def tauchen(rho, sigma, n, m=3, sigma_cond=True, full_output=False):
    """
    Implements AR(1) approximation by a Markov chain as proposed in
    Tauchen (1986) in Economic Letters.

    Parameters
    ----------
    rho : float
        AR(1) autocorrelation parameter (lambda in Tauchen 1986)
    sigma : float
        Conditional or unconditional std. deviation or AR(1)
        In the Tauchen (1986) notation this corresponds to sigma_epsilon
        and sigma_y, respectively.
    n : int
        Number of elements on discretized state space.
    m : int
        Determines range of state space, which has equidistant grid
        points on +/- m * sigma_y. Tauchen uses m=3, Floden (2008) sets
        m = 1.2 * log(n) * sigma_y
    sigma_cond : bool
        If true, sigma is taken to be the conditional std. deviation,
        i.e. the standard deviation of the error term.
    full_output : bool
        If true, return ergodic distribution, implied autocorrelation and
        std. deviation of discretized process.

    Returns
    -------
    z : array
        Discretized state space (1-d array)
    transm : array
        Transition matrix
    ergodic_dist : array
        Ergodic distribution of discretized process (1-d array)
    rho_impl : float
        Implied autocorrelation of discretized process
    sigma_z_impl : float
        Implied unconditional std. deviation of discretized process
    sigma_e_impl : float
        Implied conditional std. deviation corresponding of AR(1)

    """

    if sigma_cond:
        sigma_z = np.sqrt(sigma ** 2 / (1 - rho ** 2))
        sigma_e = sigma
    else:
        sigma_z = sigma
        sigma_e = sigma_z * np.sqrt(1 - rho**2)

    if n > 1:
        z = np.linspace(-sigma_z * m, sigma_z * m, n)
        w2 = (z[1] - z[0])/2

        transm = np.empty((n, n), dtype=np.float64)

        for j in range(n):
            for k in range(1, n-1):
                zjk = z[k] - rho * z[j]
                transm[j, k] = norm.cdf((zjk + w2) / sigma_e) - \
                               norm.cdf((zjk - w2) / sigma_e)

            zj1 = z[0] - rho * z[j]
            zjn = z[n-1] - rho * z[j]
            transm[j, 0] = norm.cdf((zj1 + w2) / sigma_e)
            transm[j, n-1] = 1 - norm.cdf((zjn - w2) / sigma_e)
    elif n == 1:
        transm = np.ones((1, 1))
        z = np.zeros(1)
    else:
        raise ValueError('Invalid number of states: n={:d}'.format(n))

    assert np.all(np.abs(np.sum(transm, axis=1) - 1) < 1e-12)

    if full_output:
        # Compute some other stuff such as the ergodic distribution
        # and implied autocorrelation / variance of the discretized process
        ergodic_dist = markov_ergodic_dist(transm)

        rho_impl, sigma_z_impl, sigma_e_impl = \
            markov_moments(z, transm, ergodic_dist)

        return z, transm, ergodic_dist, rho_impl, sigma_z_impl, sigma_e_impl
    else:
        return z, transm
