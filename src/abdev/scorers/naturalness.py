"""L4 scorer — AbLang-2 pseudo-log-likelihood naturalness regularizer (GOALS.md §6).

PLL = sum of per-token log-probs under masking; measures how natural a sequence looks
relative to the immune repertoire. Used as the `alpha * log p_AbLang` term in the composite
scorer (Eq. 9). Sweeping alpha characterizes the naturalness-developability tradeoff — a
clean, backbone-agnostic publishable result.

    import ablang2
    model = ablang2.pretrained(model_to_use='ablang2-paired')
    pll = model.pseudo_log_likelihood([(heavy, light)], mode='paired')

TODO(phase2): wrap ablang2 PLL; batch + cache.
"""
from __future__ import annotations

import numpy as np

from abdev.scorers.base import AntibodySeq, Scorer


class NaturalnessScorer(Scorer):
    name = "L4:ablang2-pll"
    inline = True

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        raise NotImplementedError("Phase 2: AbLang-2 PLL.")
