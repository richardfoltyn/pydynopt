__author__ = 'Richard Foltyn'

from pydynopt.processes import markov_ergodic_dist

import numpy as np
from scipy.stats import norm


def tauchen(rho, sigma, n, m=3, sigma_cond=True, full_output=False):
    """
    Implements AR(1) approximation by a Markov chain as proposed in
    Tauchen (1986) in Economic Letters.

    Arguments:
    rho     AR(1) autocorrelation parameter (lambda in Tauchen 1986)
    sigma   Conditional or unconditional std. deviation or AR(1)
            In the Tauchen (1986) notation this corresponds to sigma_epsilon
            and sigma_y, respectively.
    n       Number of elements on discretized state space.
    m       Determines range of state space, which has equidistant grid
            points on +/- m * sigma_y. Tauchen uses m=3, Floden (2008) sets
            m = 1.2 * log(n) * sigma_y
    sigma_cond  If true, sigma equals sigma_epsilon, i.e. the conditional
                std. deviation.
    full_output If true, return ergodic distribution and implied
                autocorrelation and std. deviation of discretized process.
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

        z_mean_uc = np.dot(ergodic_dist, z)
        z_var_uc = np.dot(np.power(z - z_mean_uc, 2), ergodic_dist)
        z_demeaned = z - z_mean_uc
        z_m1 = np.outer(z_demeaned, z_demeaned)
        wgt = transm * ergodic_dist.reshape((-1, 1))

        z_acov = np.sum(np.sum(z_m1 * wgt))

        # implied autocorrelation and variance of error term of discretized
        # process
        rho_impl = z_acov / z_var_uc
        sigma_e_impl = np.sqrt((1-rho_impl**2) * z_var_uc)
        sigma_z_impl = np.sqrt(z_var_uc)

        return z, transm, ergodic_dist, rho_impl, sigma_z_impl, sigma_e_impl
    else:
        return z, transm


if __name__ == '__main__':

    # Illustrate with the parametrization / results given in Tauchen (1986),
    # Table 1
    params = [[0.1, 0.101, 9],
              [0.8, 0.167, 9],
              [0.9, 0.229, 9],
              [0.9, 0.229, 5]]

    # Results from Table 1 for implied lambda and sigma_y of discretized process
    tbl1_res = [[0.100, 0.103],
                [0.798, 0.176],
                [0.898, 0.253],
                [0.932, 0.291]]

    results = list()

    for args in params:
        allargs = args + [3, False, True]
        results.append(tauchen(*allargs))

    header = '{:>2s}{:>10s}{:>10s}{:>15s}{:>10s}'.format('N', 'lambda',
                                                         'sigma_y',
                                                         'lambda_bar',
                                                         'sigma_yab')
    print(header)
    print('-' * len(header))
    for i, res in enumerate(results):
        print('{:2d}{:-10.2f}{:-10.3f}{:-15.3f}{:10.3f}'.format(params[i][2],
              params[i][0], params[i][1], res[3], res[4]))

        assert np.abs(tbl1_res[i][0] - res[3]) < 1e-3
        assert np.abs(tbl1_res[i][1] - res[4]) < 1e-3

