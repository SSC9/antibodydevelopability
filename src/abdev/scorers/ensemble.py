"""L3 scorer — deep ensemble + uncertainty-aware (LCB) guidance (GOALS.md §6, stretch).

Train N=5 regressors (differing seed/feature subset); guide on the lower confidence bound
    s_LCB(x) = mu(x) - beta * sigma(x)                                      (Eq. 10)
to avoid exploiting high-variance extrapolation regions. Foundation for active learning and
for an uncertainty-aware Pareto front (proposal §7.3). Wraps any base Scorer.

TODO(phase3): implement ensemble training + LCB aggregation.
"""
from __future__ import annotations

import numpy as np

from abdev.scorers.base import AntibodySeq, Scorer


class EnsembleScorer(Scorer):
    name = "L3:ensemble-lcb"
    inline = True

    def __init__(self, members: list[Scorer], beta: float = 1.0):
        self.members = members
        self.beta = beta

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        raise NotImplementedError("Phase 3: ensemble LCB.")
