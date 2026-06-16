"""L0 scorer — frozen ESM-2 embeddings + shallow regressor on GDPa1 (baseline).

The direct comparison point with Zhao et al.: any improvement must be attributable to the
framework or scorer design, not extra data. Mean-pool the ESM-2 representation of the
concatenated heavy+light sequence; fit one shallow regressor per developability property
on the 246 GDPa1 labels. Cache embeddings to disk (extraction is the only GPU cost).

TODO(phase2): implement embed() (transformers, ESM2_MODEL_ID) + fit/score on GDPa1.
"""
from __future__ import annotations

import numpy as np

from abdev.scorers.base import AntibodySeq, Scorer


class ESM2Scorer(Scorer):
    name = "L0:esm2-gdpa"
    inline = True

    def __init__(self, target: str):
        self.target = target  # which developability property this instance predicts

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        raise NotImplementedError("Phase 2: ESM-2 embed + regressor.")
