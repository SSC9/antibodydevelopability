"""Metric sanity tests (no model weights needed)."""
import numpy as np

from abdev.eval.metrics import bootstrap_ci, spearman, top_k_recall


def test_spearman_perfect_and_inverse():
    x = np.arange(20.0)
    assert spearman(x, x) == 1.0
    assert spearman(x, -x) == -1.0


def test_spearman_ignores_nans():
    pred = np.array([1.0, 2.0, np.nan, 4.0])
    true = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(pred, true) == 1.0


def test_top_k_recall_perfect():
    true = np.arange(100.0)
    assert top_k_recall(true, true, k=0.1) == 1.0  # same ranking → full overlap


def test_bootstrap_ci_orders():
    rng = np.random.default_rng(0)
    true = rng.normal(size=200)
    pred = true + rng.normal(scale=0.3, size=200)
    point, lo, hi = bootstrap_ci(pred, true, n_boot=200, seed=1)
    assert lo <= point <= hi and 0.5 < point <= 1.0
