"""Departure process of the M^k/G/infinity batch-arrival queue
with heterogeneous dependent demands.

Reference implementation accompanying the 4th-year course paper of A.E. Salimov
(MSU, Faculty of Mechanics and Mathematics, Department of Probability
Theory, 2026) supervised by Prof. G.I. Falin.

The module provides:

* The bivariate Farlie-Gumbel-Morgenstern (FGM) distribution with
  exponential marginals used as a tractable model of within-batch
  dependence between service times.
* The functions g_y(t) describing the rate of batches whose outcome
  set equals y in [0, t].
* Closed-form covariance kernels Cov(N_i(t), D_j(t)) and
  Cov(D_i(t_1), D_j(t_2)) for the two-dimensional FGM case derived
  in Sections 4-6 of the thesis.
* A direct Monte-Carlo simulator for the M^k/G/infinity queue that
  serves as a sanity check for the analytic formulas.

References
----------
Falin, G. (1994). The M^k/G/infty batch arrival queue by heterogeneous
dependent demands. Journal of Applied Probability, 31(3), 841-846.

Daw, A., Fralix, B., Pender, J. (2022). Non-stationary queues with
batch arrivals. arXiv:2008.00625.

Gumbel, E. J. (1960). Bivariate exponential distributions. JASA, 55(292).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class FGMExponential:
    """Bivariate Farlie-Gumbel-Morgenstern distribution with Exp(1) marginals.

    The joint CDF is

        F(t1, t2) = (1 - e^{-t1})(1 - e^{-t2}) * (1 + alpha * e^{-t1 - t2}),

    and the dependence parameter ``alpha`` is restricted to ``[-1, 1]``.
    """

    alpha: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.alpha <= 1.0:
            raise ValueError("FGM parameter alpha must lie in [-1, 1].")

    def cdf(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        return (
            (1.0 - np.exp(-t1))
            * (1.0 - np.exp(-t2))
            * (1.0 + self.alpha * np.exp(-t1 - t2))
        )

    def pdf(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        return np.exp(-t1 - t2) * (
            1.0
            + self.alpha
            * (2.0 * np.exp(-t1) - 1.0)
            * (2.0 * np.exp(-t2) - 1.0)
        )

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        u = rng.uniform(size=size)
        v = rng.uniform(size=size)
        a = self.alpha * (2.0 * u - 1.0)
        b = -(1.0 + a)
        c = a
        d = np.sqrt(b * b - 4.0 * a * v)
        with np.errstate(invalid="ignore", divide="ignore"):
            v_corr = np.where(
                np.isclose(a, 0.0),
                v,
                2.0 * c / (-b + d) if False else v,
            )
            v_corr = np.where(np.isclose(a, 0.0), v, (-b - d) / (2.0 * a))
        v_corr = np.clip(v_corr, 1e-12, 1.0 - 1e-12)
        t1 = -np.log1p(-u)
        t2 = -np.log1p(-v_corr)
        return np.column_stack([t1, t2])


def g_single_fgm(t: np.ndarray, alpha: float) -> np.ndarray:
    """Analytic expression for ``g_{{1}}(t)`` in the FGM case.

    From the thesis (Section 4, Example 4.1):

        g_{{1}}(t) = (1 - e^{-t})
                   - (alpha + 1) / 2 * (1 - e^{-2t})
                   + 2 alpha / 3 * (1 - e^{-3t})
                   - alpha / 4 * (1 - e^{-4t}).
    """
    t = np.asarray(t, dtype=float)
    return (
        (1.0 - np.exp(-t))
        - 0.5 * (alpha + 1.0) * (1.0 - np.exp(-2.0 * t))
        + (2.0 * alpha / 3.0) * (1.0 - np.exp(-3.0 * t))
        - (alpha / 4.0) * (1.0 - np.exp(-4.0 * t))
    )


def g_single_fgm_limit(alpha: float) -> float:
    """Stationary limit ``lim_{t -> infty} g_{{1}}(t) = 1/2 - alpha/12``."""
    return 0.5 - alpha / 12.0


def cov_n_d_fgm(t: float, alpha: float, lam: float = 1.0) -> np.ndarray:
    """Covariance matrix ``Cov(N(t), D(t))`` for the 2D FGM model.

    By Theorem 1 the diagonal entries vanish (independence of
    ``N_i(t)`` and ``D_i(t)``) and the off-diagonal entries equal
    ``lambda * g_{{1}}(t)``.
    """
    g = float(g_single_fgm(np.array(t), alpha))
    return np.array([[0.0, lam * g], [lam * g, 0.0]])


def cov_dd_diag_fgm(t1: float, t2: float, lam: float = 1.0) -> float:
    """Diagonal entry ``Cov(D_i(t1), D_i(t2)) = lambda * (m - 1 + e^{-m})``
    with ``m = min(t1, t2)`` (independent of alpha by symmetry of the
    marginals).
    """
    m = min(float(t1), float(t2))
    return lam * (m - 1.0 + np.exp(-m))


def cov_dd_offdiag_fgm(t1: float, t2: float, alpha: float, lam: float = 1.0) -> float:
    """Off-diagonal entry ``Cov(D_1(t1), D_2(t2))`` for the 2D FGM model.

    Implements formula (\\ref{eq:cov_matrix_offdiag}) of the thesis.
    Assumes ``t1 <= t2``; the symmetric case is handled by the public
    wrapper ``cov_dd_fgm``.
    """
    a, b = float(t1), float(t2)
    base = (
        a
        - 1.0
        + np.exp(-a)
        + np.exp(-b)
        - 0.5 * np.exp(a - b)
        - 0.5 * np.exp(-a - b)
    )
    correction = (
        (1.0 / 6.0) * np.exp(a - b)
        - (1.0 / 12.0) * np.exp(2.0 * (a - b))
        - 0.5 * np.exp(-a - b)
        + (1.0 / 3.0) * np.exp(-2.0 * a - b)
        + (1.0 / 3.0) * np.exp(-a - 2.0 * b)
        - 0.25 * np.exp(-2.0 * a - 2.0 * b)
    )
    return lam * (base + alpha * correction)


def cov_dd_fgm(
    t1: float, t2: float, alpha: float, lam: float = 1.0
) -> np.ndarray:
    """Full 2x2 covariance matrix ``Cov(D(t1), D(t2))`` for the FGM model."""
    a, b = (t1, t2) if t1 <= t2 else (t2, t1)
    diag = cov_dd_diag_fgm(a, b, lam)
    off = cov_dd_offdiag_fgm(a, b, alpha, lam)
    return np.array([[diag, off], [off, diag]])


def correlation_dd_fgm(t: float, alpha: float, lam: float = 1.0) -> float:
    """Equal-time correlation ``rho_{12}(t)`` between ``D_1`` and ``D_2``."""
    m = cov_dd_fgm(t, t, alpha, lam)
    denom = np.sqrt(m[0, 0] * m[1, 1])
    if denom == 0.0:
        return float("nan")
    return float(m[0, 1] / denom)


@dataclass
class SimulationResult:
    """Container for outputs of the Monte-Carlo simulator."""

    times: np.ndarray
    N_paths: np.ndarray
    D_paths: np.ndarray

    @property
    def n_replicas(self) -> int:
        return self.N_paths.shape[0]

    @property
    def k(self) -> int:
        return self.N_paths.shape[2]


def simulate_mk_g_inf(
    lam: float,
    horizon: float,
    grid: np.ndarray,
    service_sampler: Callable[[int, np.random.Generator], np.ndarray],
    n_replicas: int,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Discrete-event simulator for the ``M^k/G/infinity`` queue.

    Parameters
    ----------
    lam : float
        Batch arrival rate.
    horizon : float
        End of the observation interval (must contain ``grid``).
    grid : ndarray
        Time points at which ``N(t)`` and ``D(t)`` are recorded.
    service_sampler : callable
        ``service_sampler(n_batches, rng)`` returns an array of shape
        ``(n_batches, k)`` of service times for each batch.
    n_replicas : int
        Number of independent replications.

    Returns
    -------
    SimulationResult
        Trajectories of ``N(t)`` and ``D(t)`` of shape
        ``(n_replicas, len(grid), k)``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    probe = service_sampler(1, rng)
    k = int(probe.shape[1])
    N_paths = np.zeros((n_replicas, len(grid), k), dtype=np.int64)
    D_paths = np.zeros_like(N_paths)
    for rep in range(n_replicas):
        n_batches = rng.poisson(lam * horizon)
        if n_batches == 0:
            continue
        arrivals = np.sort(rng.uniform(0.0, horizon, size=n_batches))
        services = service_sampler(n_batches, rng)
        departures = arrivals[:, None] + services
        for gi, t in enumerate(grid):
            arrived = arrivals <= t
            departed = (departures <= t) & arrived[:, None]
            N_paths[rep, gi] = arrived[:, None].sum(axis=0) - departed.sum(axis=0)
            D_paths[rep, gi] = departed.sum(axis=0)
    return SimulationResult(times=np.asarray(grid), N_paths=N_paths, D_paths=D_paths)


def fgm_service_sampler(alpha: float) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Service-time sampler for the 2D FGM model with Exp(1) marginals."""
    dist = FGMExponential(alpha)

    def _sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return dist.sample(n, rng)

    return _sampler


__all__ = [
    "FGMExponential",
    "g_single_fgm",
    "g_single_fgm_limit",
    "cov_n_d_fgm",
    "cov_dd_diag_fgm",
    "cov_dd_offdiag_fgm",
    "cov_dd_fgm",
    "correlation_dd_fgm",
    "SimulationResult",
    "simulate_mk_g_inf",
    "fgm_service_sampler",
]
