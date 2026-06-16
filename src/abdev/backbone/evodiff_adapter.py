"""EvoDiff backbone adapter (GOALS.md §4, Phase 1).

EvoDiff (Alamdari et al. 2023) is a public discrete diffusion model for protein sequences
(OADM objective, UniRef50). We fine-tune it on OAS antibody sequences rather than retrain
from scratch — the single-GPU budget choice. The denoiser f_theta(x_t, t) predicts the clean
sequence x_1; trained with cross-entropy (Eq. 7) and plugged into the CTMC sampler at inference.

Requires the `backbone` extra.

TODO(phase1): load EvoDiff, OAS fine-tuning loop (confirm single-GPU feasibility first),
expose `predict_x1(x_t, t) -> per-position token logits` for the sampler/guidance.
"""
from __future__ import annotations


class EvoDiffBackbone:
    """Wraps a (fine-tuned) EvoDiff denoiser as f_theta(x_t, t) -> clean-token logits."""

    def predict_x1_logits(self, x_t, t):
        raise NotImplementedError("Phase 1: EvoDiff denoiser forward pass.")
