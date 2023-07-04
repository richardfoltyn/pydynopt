
import numpy as np

from ..common import ConvergenceError


def rouwenhorst(n, mu, rho, sigma):
    """
    Code to approximate an AR(1) process using the Rouwenhorst method as in
    Kopecky & Suen, Review of Economic Dynamics (2010), Vol 13, pp. 701-714
    Adapted from Matlab code by Martin Floden.

    Parameters
    ----------
    n : int
        Number of states for discretized Markov process
    mu : float
        Unconditional mean or AR(1) process
    rho : float
        Autocorrelation of AR(1) process
    sigma : float
        Conditional standard deviation of AR(1) innovations

    Returns
    -------
    z : numpy.ndarray
        Discretized state space
    Pi : numpy.ndarray
        Transition matrix of discretized process where
            Pi[i,j] = Prob[z'=z_j | z=z_i]
    """

    if n < 1:
        msg = 'Invalid number of states'
        raise ValueError(msg)
    if sigma < 0.0:
        msg = 'Argument sigma must be non-negative'
        raise ValueError(msg)
    if abs(rho) >= 1.0:
        msg = 'Cannot create stationary process with abs(rho) >= 1.0'
        raise ValueError(msg)

    if n == 1:
        # Degenerate process on a single state: disregard variance and
        # autocorrelation
        z = np.array([mu])
        Pi = np.ones((1, 1))
        return z, Pi

    p = (1+rho)/2
    Pi = np.array([[p, 1-p], [1-p, p]])

    for i in range(Pi.shape[0], n):
        tmp = np.pad(Pi, 1, mode='constant', constant_values=0)
        Pi = p * tmp[1:, 1:] + (1-p) * tmp[1:, :-1] + \
             (1-p) * tmp[:-1, 1:] + p * tmp[:-1, :-1]
        Pi[1:-1, :] /= 2

    fi = np.sqrt(n-1) * sigma / np.sqrt(1 - rho ** 2)
    z = np.linspace(-fi, fi, n) + mu

    return z, Pi


def markov_ergodic_dist(transm, tol=1e-12, maxiter=10000, transpose=True,
                        mu0=None, inverse=False):
    """
    Compute the ergodic distribution implied by a given Markov chain transition
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

    n = len(states)

    # unconditional centered moments
    # include the 0-th moment so that m[i] gives the i-th moment
    m = np.zeros(5)

    if n > 1:
        if ergodic_dist is None:
            ergodic_dist = markov_ergodic_dist(transm, *args, **kwargs)
        else:
            ergodic_dist = np.atleast_1d(ergodic_dist)

        x = np.atleast_1d(states)

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
    elif n == 1:
        autocorr = 1.0
        sigma_e = 0.0
        sigma_x = 0.0
    else:
        raise ValueError('Invalid number of states: {:d}'.format(n))

    if moments:
        # do not return the 0-th moment
        return autocorr, sigma_x, sigma_e, m[1:]
    else:
        return autocorr, sigma_x, sigma_e


def markov_simulate(transm, size, dtype=int, init=None):
    """
    Simulate a sequence of draws from a Markov chain with given transition
    matrix.

    Parameters
    ----------
    transm : ndarray
        Transition matrix of Markov process
    size : int
        Number of draws to be simulated
    dtype : object
        Optional integer dtype of return value
    init : int or None
        Optional initial value of the simulated series. If None, a random
        initial value will be drawn from the ergodic distribution.

    Returns
    -------
    isim : ndarray
        Array of simulated draws.
    """

    if transm.shape[0] != transm.shape[1] or transm.ndim != 2:
        raise ValueError('Invalid transition matrix shape')

    x = np.sum(transm, axis=1)
    if np.any(abs(x-1.0) > 1.0e-12) or np.any(transm < 0.0):
        raise ValueError('Invalid values in transition matrix')

    if init is not None:
        init = int(init)
        if init >= transm.shape[0] or init < 0:
            raise ValueError('Invalid initial value %d'.format(init))

    m = transm.shape[0]
    n = size

    # "cumulative" transition matrix
    tm_cum = np.cumsum(transm, axis=1)
    tm_cum = np.hstack((np.zeros((m, 1)), tm_cum))

    isim = np.empty(n, dtype=dtype)

    if init is None:
        # Compute ergodic distribution for initial draw
        edist = markov_ergodic_dist(transm, inverse=True)
        init = int(np.random.choice(np.arange(m), 1, p=edist))

    isim[0] = init

    eps = np.random.rand(n-1)

    for i in range(1, n):
        ifrom = isim[i-1]
        ito = np.sum(tm_cum[ifrom] < eps[i-1]) - 1
        isim[i] = ito

    return isim


def istransm(m, transposed=False, tol=1e-12):
    """
    Check that matrix m satisfies properties of transition matrix

    Parameters
    ----------
    m : np.ndarray
        Transition matrix with element (i,j) containing the probability
        Prob[x' = j | x = i].
    transposed : bool
        If true, assume that transition matrix `tm` is transposed.
    tol : float
        Tolerance on rows sums being close to 1.0 (column sums, if transposed)

    Returns
    -------
    valid : bool
        True if `tm` is a valid Markov transition matrix.
    """

    m = np.atleast_2d(m)
    if (m.shape[0] != m.shape[1]) or (m.ndim != 2):
        msg = 'Square matrix argument required'
        raise ValueError(msg)

    axis = 1 - int(transposed)
    valid = np.all(m >= 0) and np.all(m <= 1)
    valid = valid and np.all(np.abs(np.sum(m, axis=axis) - 1 < tol))

    return valid


def discretize_markov(nobs: int,
                      tm: np.ndarray,
                      verbose: bool = False,
                      logger=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a discrete approximation for a Markov process for a given number
    of cross-sectional units.

    This function (slightly) perturbs transition probabilities as needed such
    that in a finite-sample cross-section the simulated cross-section
    is still ergodic.

    Parameters
    ----------
    nobs : int
        Size of the simulated cross-section
    tm : np.ndarray
        Markov chain transition matrix
    verbose : bool, optional
    logger
        If not None, use logger instance for status messages.

    Returns
    -------
    inv_dist_approx : np.ndarray
        Invariant distribution in terms of observations
    tm_approx : np.ndarray
        Transition "histogram" in terms of observations
    """

    nstates = tm.shape[0]
    tm_approx = np.empty_like(tm, dtype=int)

    inv_dist = markov_ergodic_dist(tm, inverse=True)
    inv_dist_approx = pmf_to_histogram(nobs, inv_dist, verbose, logger)

    for i in range(nstates):
        tm_approx[i] = pmf_to_histogram(inv_dist_approx[i], tm[i],
                                        verbose, logger)

    inv_to = np.sum(tm_approx, axis=0)
    while not np.all(inv_dist_approx == inv_to):
        i = np.argmax(inv_dist_approx - inv_to)
        j = np.argmin(inv_dist_approx - inv_to)

        cidx = np.array([[i, j]], dtype=int)

        # need to flip columns i and j in one row. As candidate rows we
        # consider only those rows where original transition matrix has no
        # exact zeros in columns i, j, as we do not want to artificially
        # introduce transitions that did not exist with non-zero probability
        # in the original process.
        ridx = (tm[:, i] > 0.0)
        # Do not subtract from elements in approx. that are already zero!
        ridx &= (tm_approx[:, j] > 0)
        # Resulting matrix for all candidate rows
        tm_try = np.copy(tm_approx)
        tm_try[:, i] += 1
        tm_try[:, j] -= 1
        # Remove all rows that would lead to absorbing state, i.e. rows
        # with only one non-zero element.
        absorb = np.any(np.sum(tm_try, axis=1, keepdims=True) == tm_try, axis=1)
        ridx &= ~absorb

        if not np.any(ridx):
            raise ValueError('Cannot discretize transition matrix')

        ridx = np.arange(nstates)[ridx].reshape((-1, 1))

        m = tm_approx[ridx, cidx] + np.array([[1.0, -1.0]])
        m /= inv_dist_approx[ridx]
        diff = m - tm[ridx, cidx]

        eps = np.sum(np.abs(diff), axis=1)
        # row with least distortions
        k = np.argmin(eps)
        tm_approx[ridx[k], cidx] += np.array([[1, -1]])
        inv_to = np.sum(tm_approx, axis=0)

    # Perform some consistency checks and diagnostics
    # Implied transition matrix of sample-size-adjusted process
    tm_impl = tm_approx / inv_to[:, None]
    # this should hold by construction
    assert np.max(np.abs(np.sum(tm_impl, axis=1) - 1)) < 1e-12

    inv_dist_impl = markov_ergodic_dist(tm_impl, inverse=True)
    inv_dist_impl_smpl = pmf_to_histogram(nobs, inv_dist_impl, verbose=False)
    assert np.all(inv_dist_impl_smpl == inv_dist_approx)

    tm_smpl_impl = np.empty_like(tm_approx)
    for i in range(nstates):
        tm_smpl_impl[i] = pmf_to_histogram(inv_dist_approx[i], tm_impl[i],
                                           verbose=False)

    assert np.all(tm_smpl_impl == tm_approx)

    eps = np.max(np.abs(tm_impl - tm))
    msg = f'MARKOV_SIM: trans. mat. max delta: {eps:.3e}'

    if logger is not None:
        logger.debug(msg)
    elif verbose:
        print(msg)

    tm_approx = np.array(tm_approx, dtype=int)

    return inv_dist_approx, tm_approx


def pmf_to_histogram(nobs: int,
                     pmf: np.ndarray,
                     verbose: bool = False,
                     logger=None) -> np.ndarray:
    """
    Convert a PMF to an "optimal" histogram for a given number of observations.

    Parameters
    ----------
    nobs
    pmf
    verbose
    logger

    Returns
    -------
    np.ndarray
    """

    assert np.all(pmf >= 0)
    idx = np.where(pmf > 0)[0]

    arr = np.zeros_like(pmf)
    arr[idx] = np.around(nobs * pmf[idx])

    while np.sum(arr) > nobs:
        i = np.argmax(arr[idx] / nobs - pmf[idx])
        arr[idx[i]] -= 1
    while np.sum(arr) < nobs:
        i = np.argmax(pmf[idx] - arr[idx] / nobs)
        arr[idx[i]] += 1

    eps = np.max(np.abs(arr / nobs - pmf))
    msg = f'PMF_TO_HISTOGRAM: max. deviation: {eps:.3e}'

    if logger is not None:
        logger.debug(msg)
    elif verbose:
        print(msg)

    arr = np.array(arr, dtype=int)
    return arr
