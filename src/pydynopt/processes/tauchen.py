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
    rho :   AR(1) autocorrelation parameter (lambda in Tauchen 1986)
    sigma : Conditional or unconditional std. deviation or AR(1)
            In the Tauchen (1986) notation this corresponds to sigma_epsilon
            and sigma_y, respectively.
    n :     Number of elements on discretized state space.
    m :     Determines range of state space, which has equidistant grid
            points on +/- m * sigma_y. Tauchen uses m=3, Floden (2008) sets
            m = 1.2 * log(n) * sigma_y
    sigma_cond : If true, sigma is taken to be the conditional std.
                deviation, i.e. the standard deviation of the error term.
    full_output : If true, return ergodic distribution, implied
                  autocorrelation and std. deviation of discretized process.

    Returns
    -------

    """

    if sigma_cond:
        sigma_z = np.sqrt(sigma ** 2 / (1 - rho ** 2))
        sigma_e = sigma
    else:
        sigma_z = sigma
        sigma_e = sigma_z * np.sqrt(1 - rho**2)

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
