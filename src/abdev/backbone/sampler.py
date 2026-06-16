"""CTMC sampler for discrete flow matching (GOALS.md §5, proposal §3).

Simulates the continuous-time Markov chain t: 0->1. At each step the denoiser predicts the
clean-token distribution f_theta(x_1 | x_t, t); the marginal rate matrix R_hat_t (Eq. 6) is
formed and Euler steps advance the chain. A `guidance` hook reweights the predicted
distribution in token space *before* forming the rate (posterior tempering, Eq. 8), so
guidance is a drop-in modification of the denoiser output — no architectural change.

Positions are treated conditionally independent given (x_t, t) (standard factorized CTMC);
epistasis is an acknowledged approximation (proposal §3.5).

TODO(phase1): implement Euler CTMC simulation with the optional guidance callback.
"""
from __future__ import annotations

from collections.abc import Callable


def sample(backbone, n_steps: int = 128, guidance: Callable | None = None):
    """Generate sequences by simulating the CTMC. `guidance` is a posterior-tempering hook."""
    raise NotImplementedError("Phase 1: CTMC Euler sampler.")
