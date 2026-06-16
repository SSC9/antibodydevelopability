"""L1 scorer — physics-informed sequence features (GOALS.md §6, ablation).

Two feature sets used as auxiliary inputs to the GDPa1 regressor:
  * PROPERMAB (iFeatureOmega): sequence-derived descriptors incl. SAP aggregation score.
    Near-instant -> inline-capable.
  * TAP (ImmuneBuilder/ABodyBuilder2): 5 metrics (CDR length, PSH, PPC, PNC, SFvCSP).
    ~3 s/seq structure prediction -> terminal evaluator only (`inline = False`).

Requires the `features` extra (PROPERMAB) and/or the `physics` extra (TAP).

TODO(phase3): implement PROPERMAB feature extraction and the TAP terminal scorer.
"""
from __future__ import annotations

import numpy as np

from abdev.scorers.base import AntibodySeq, Scorer


class PropermabScorer(Scorer):
    name = "L1:propermab"
    inline = True

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        raise NotImplementedError("Phase 3: PROPERMAB features.")


class TAPScorer(Scorer):
    name = "L1:tap"
    inline = False  # ~3 s structure prediction; terminal evaluation only

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        raise NotImplementedError("Phase 3: TAP via ImmuneBuilder.")
