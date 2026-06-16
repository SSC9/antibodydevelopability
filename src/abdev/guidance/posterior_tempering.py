"""Posterior tempering — the core guidance mechanism (GOALS.md §5, proposal §4.1).

Reweight the denoiser's predicted clean-token distribution by the scorer:

    p_tilde(x_1 | x_t, t)  ∝  f_theta(x_1 | x_t, t) · exp(lambda · s(x_1))      (Eq. 8)

then feed p_tilde back into the rate matrix (Eq. 6). Properties vs. SVDD (proposal §4.2):
  * no differentiability needed — s is any black-box Scorer;
  * no look-ahead sampling — one scorer call/step, not M rollouts;
  * lambda tunes guidance strength.

This is implemented as a callback the CTMC sampler invokes each step.

TODO(phase1): implement the reweighting given clean-token logits + a Scorer.
"""
from __future__ import annotations

from abdev.scorers.base import Scorer


def make_guidance(scorer: Scorer, lam: float):
    """Return a sampler-compatible guidance callback that applies Eq. 8 with strength `lam`."""
    raise NotImplementedError("Phase 1: posterior-tempering reweighting.")
