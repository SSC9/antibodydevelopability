"""L2 scorer — SSH2.0 OAS-scale weak supervision (CORE NOVELTY, GOALS.md §6).

Pipeline:
  1. Retrain SSH2.0 SVM ensemble locally on its published 131-seq set using CKSAAGP
     features (sub-ms query time; the public web server is rate-limited/fragile).
  2. Pseudo-label a ~100k OAS paired subset (soft probabilities).
  3. Pretrain an ESM-2 regressor on the pseudo-labels (BCE).
  4. Swap the head and fine-tune (MSE) on the 246 GDPa1 continuous labels.

This is the weak-supervision answer to the 246-label bottleneck — the lift over L0 is the
headline ablation. Requires the `oas` + `features` extras.

TODO(phase2): (a) local SSH2.0 retrain, (b) pseudo-label, (c) pretrain, (d) fine-tune.
"""
from __future__ import annotations

import numpy as np

from abdev.scorers.base import AntibodySeq, Scorer


class SSH2Scorer(Scorer):
    name = "L2:ssh2-oas-gdpa"
    inline = True

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        raise NotImplementedError("Phase 2: SSH2.0 weak-supervision pipeline.")
