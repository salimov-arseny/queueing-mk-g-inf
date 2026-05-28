"""Unit tests for the M^k/G/infinity module.

These checks verify (a) limit values of g_{{1}}(t) as t -> infty,
(b) the asymptotic full correlation rho_{12}(t) -> 1, and (c) agreement
between the closed-form covariances and a Monte-Carlo simulation of the
queue.
"""

import math

import numpy as np
import pytest

from src.queue_mkginf import (
    FGMExponential,
    correlation_dd_fgm,
    cov_dd_diag_fgm,
    cov_dd_fgm,
    cov_n_d_fgm,
    fgm_service_sampler,
    g_single_fgm,
    g_single_fgm_limit,
    simulate_mk_g_inf,
)


class TestFGMDistribution:
    @pytest.mark.parametrize("alpha", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_marginal_exponentiality(self, alpha):
        rng = np.random.default_rng(0)
        sample = FGMExponential(alpha).sample(20000, rng)
        assert abs(sample[:, 0].mean() - 1.0) < 0.05
        assert abs(sample[:, 1].mean() - 1.0) < 0.05

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 1.0])
    def test_covariance_matches_gumbel(self, alpha):
        rng = np.random.default_rng(0)
        sample = FGMExponential(alpha).sample(40000, rng)
        empirical = np.cov(sample.T)[0, 1]
        assert abs(empirical - alpha / 4.0) < 0.03

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError):
            FGMExponential(1.5)


class TestGFunction:
    @pytest.mark.parametrize("alpha", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_limit_at_infinity(self, alpha):
        value = float(g_single_fgm(np.array(50.0), alpha))
        assert value == pytest.approx(g_single_fgm_limit(alpha), abs=1e-6)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 1.0])
    def test_nonnegative_on_grid(self, alpha):
        grid = np.linspace(0.0, 10.0, 200)
        values = g_single_fgm(grid, alpha)
        assert (values >= -1e-12).all()


class TestCovariances:
    def test_cov_nd_diagonal_zero(self):
        m = cov_n_d_fgm(t=5.0, alpha=0.5)
        assert m[0, 0] == 0.0 and m[1, 1] == 0.0

    def test_cov_nd_offdiagonal_positive(self):
        m = cov_n_d_fgm(t=5.0, alpha=0.0)
        assert m[0, 1] > 0.0 and m[1, 0] > 0.0

    def test_cov_dd_diag_linear_asymptote(self):
        t = 50.0
        assert cov_dd_diag_fgm(t, t) == pytest.approx(t - 1.0, abs=1e-6)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 1.0])
    def test_rho_to_one(self, alpha):
        rho = correlation_dd_fgm(t=80.0, alpha=alpha)
        assert rho == pytest.approx(1.0, abs=1e-3)


class TestSimulationVsAnalytic:
    def test_diagonal_variance(self):
        rng = np.random.default_rng(42)
        grid = np.array([1.0, 2.0, 4.0])
        result = simulate_mk_g_inf(
            lam=1.0,
            horizon=4.5,
            grid=grid,
            service_sampler=fgm_service_sampler(alpha=0.0),
            n_replicas=4000,
            rng=rng,
        )
        for gi, t in enumerate(grid):
            empirical = np.var(result.D_paths[:, gi, 0])
            theoretical = cov_dd_diag_fgm(t, t, lam=1.0)
            assert empirical == pytest.approx(theoretical, rel=0.1, abs=0.1)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 1.0])
    def test_offdiagonal_covariance(self, alpha):
        rng = np.random.default_rng(123)
        grid = np.array([3.0])
        result = simulate_mk_g_inf(
            lam=1.0,
            horizon=3.5,
            grid=grid,
            service_sampler=fgm_service_sampler(alpha),
            n_replicas=4000,
            rng=rng,
        )
        empirical = np.cov(result.D_paths[:, 0, 0], result.D_paths[:, 0, 1])[0, 1]
        theoretical = cov_dd_fgm(3.0, 3.0, alpha)[0, 1]
        assert empirical == pytest.approx(theoretical, rel=0.2, abs=0.15)
