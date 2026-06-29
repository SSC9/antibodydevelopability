"""Embedding-based HIC oracles (train on GDPa1 → predict GDPa3).

Implemented here (no OAS needed):
  * GoldOnly-Ridge   — Ridge on embeddings (the benchmark-comparable bar)
  * SSH2-feature     — Ridge on [embeddings ++ SSH2.0 P_positive] (no OAS; isolates the raw signal)

WeakSup (OAS pretrain → fine-tune) and the arch-matched GoldOnly-MLP live in `weaksup.py` (built
when OAS embeddings + pseudo-labels exist), so the MLP architecture is shared for a clean delta.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ridge_predict(X_tr, y_tr, X_te, alpha: float = 1.0, standardize: bool = True):
    """Ridge regression. `standardize=False, alpha=1.0` reproduces the abdev-benchmark recipe."""
    model = (
        make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        if standardize
        else Ridge(alpha=alpha)
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def with_ssh2_feature(X, ids, seqs_by_id):
    """Append SSH2.0 P_positive as one extra (standardized-at-fit) feature column.

    `seqs_by_id` maps id -> (heavy, light). Returns X augmented with the SSH2.0 score per row.
    """
    from abdev.scorers.ssh2 import ssh2_p_positive

    heavy = [seqs_by_id[i][0] for i in ids]
    light = [seqs_by_id[i][1] for i in ids]
    pp = ssh2_p_positive(heavy, light, names=ids).reshape(-1, 1)
    return np.hstack([X, pp])
