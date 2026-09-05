"""Unit tests for Tauchen discretization.

Author: Richard Foltyn
"""

import numpy as np
import pytest

from pydynopt.processes.tauchen import tauchen


@pytest.mark.parametrize(
    ('rho', 'sigma_y', 'n', 'exp_lambda_bar', 'exp_sigma_y'),
    [
        (0.1, 0.101, 9, 0.100, 0.103),
        (0.8, 0.167, 9, 0.798, 0.176),
        (0.9, 0.229, 9, 0.898, 0.253),
        (0.9, 0.229, 5, 0.932, 0.291),
    ],
)
def test_tauchen_table1(
    rho: float,
    sigma_y: float,
    n: int,
    exp_lambda_bar: float,
    exp_sigma_y: float,
) -> None:
    """Verify implied moments match Tauchen (1986) Table 1."""
    z, p, dist, rho_impl, sigma_z_impl, _ = tauchen(
        rho, sigma_y, n, m=3, sigma_cond=False, full_output=True
    )
    assert abs(exp_lambda_bar - rho_impl) < 1e-3
    assert abs(exp_sigma_y - sigma_z_impl) < 1e-3
    assert z.shape == (n,)
    assert p.shape == (n, n)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0.0)
    assert np.allclose(dist.sum(), 1.0)


def test_tauchen_basic_output() -> None:
    """Verify default output format and stochastic matrix properties."""
    z, p = tauchen(0.8, 0.1, 5)
    assert z.shape == (5,)
    assert p.shape == (5, 5)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0.0)


def test_tauchen_conditional_vs_unconditional() -> None:
    """Verify consistency between conditional and unconditional sigma specifications."""
    rho = 0.8
    sigma_e = 0.1
    sigma_z = np.sqrt(sigma_e**2 / (1.0 - rho**2))

    z_cond, p_cond = tauchen(rho, sigma_e, 7, sigma_cond=True)
    z_uncond, p_uncond = tauchen(rho, sigma_z, 7, sigma_cond=False)

    assert np.allclose(z_cond, z_uncond)
    assert np.allclose(p_cond, p_uncond)


def test_tauchen_single_state() -> None:
    """Verify degenerate state space when n=1."""
    z, p = tauchen(0.8, 0.1, 1)
    assert np.array_equal(z, np.zeros(1))
    assert np.array_equal(p, np.ones((1, 1)))


def test_tauchen_invalid_n() -> None:
    """Verify ValueError is raised for non-positive number of states."""
    with pytest.raises(ValueError):
        tauchen(0.8, 0.1, 0)

    with pytest.raises(ValueError):
        tauchen(0.8, 0.1, -3)
