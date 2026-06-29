"""Evaluation metrics for the HIC oracle (GOALS.md §8): Spearman, top-k recall, bootstrap CIs.

Scores are compared to true HIC by **rank** (Spearman), so a predictor on any scale works (e.g.
SSH2.0's probability). Convention: a higher score means higher predicted HIC. The headline number
is Spearman ρ on the external GDPa3 held-out, with a bootstrap CI (n=79 is small).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _clean(pred, true):
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    m = ~(np.isnan(pred) | np.isnan(true))
    return pred[m], true[m]


def spearman(pred, true) -> float:
    pred, true = _clean(pred, true)
    return float(spearmanr(pred, true)[0])


def top_k_recall(pred, true, k: float = 0.1) -> float:
    """Of the worst-k% antibodies by true HIC, the fraction also in the highest-predicted k%.

    Assumes higher score = higher HIC. (Benchmark's secondary metric.)
    """
    pred, true = _clean(pred, true)
    n = len(true)
    k_n = max(1, int(round(k * n)))
    top_true = set(np.argsort(-true)[:k_n].tolist())
    top_pred = set(np.argsort(-pred)[:k_n].tolist())
    return len(top_true & top_pred) / k_n


def bootstrap_ci(pred, true, metric=spearman, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Return (point_estimate, lo, hi) for `metric` via case resampling (fixed seed)."""
    pred, true = _clean(pred, true)
    rng = np.random.default_rng(seed)
    n = len(true)
    vals = [metric(pred[idx], true[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(pred, true), float(lo), float(hi)
