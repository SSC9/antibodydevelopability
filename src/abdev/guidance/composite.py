"""Composite scorer — naturalness-developability tradeoff (GOALS.md §6 L4, proposal Eq. 9).

    s_composite(x_1) = s_dev(x_1) + alpha · log p_AbLang(x_1)                  (Eq. 9)

alpha = 0 is pure developability; alpha -> inf preserves naturalness. Sweeping alpha
empirically characterizes the tradeoff (a secondary publishable result).

TODO(phase2): combine a developability Scorer with the L4 NaturalnessScorer.
"""
from __future__ import annotations

import numpy as np

from abdev.scorers.base import AntibodySeq, Scorer


class CompositeScorer(Scorer):
    name = "composite:dev+alpha*pll"
    inline = True

    def __init__(self, dev: Scorer, naturalness: Scorer, alpha: float):
        self.dev = dev
        self.naturalness = naturalness
        self.alpha = alpha

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        return self.dev(seqs) + self.alpha * self.naturalness(seqs)
