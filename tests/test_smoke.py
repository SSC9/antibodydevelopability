"""Smoke tests: the package imports and the Scorer contract composes correctly.

These run without any heavy model weights (no torch/ESM downloads) so `uv run pytest`
works immediately after `uv sync`.
"""
import numpy as np

from abdev.scorers.base import AntibodySeq, Scorer
from abdev.guidance.composite import CompositeScorer


class _ConstScorer(Scorer):
    def __init__(self, value):
        self.value = value

    def score(self, seqs):
        return np.full(len(seqs), self.value, dtype=float)


def test_version():
    import abdev
    assert abdev.__version__


def test_composite_combines_dev_and_naturalness():
    seqs = [AntibodySeq("EVQL", "DIQM"), AntibodySeq("QVQL", "EIVL")]
    comp = CompositeScorer(dev=_ConstScorer(2.0), naturalness=_ConstScorer(0.5), alpha=3.0)
    out = comp(seqs)
    assert out.shape == (2,)
    # s_dev + alpha * pll = 2.0 + 3.0 * 0.5 = 3.5
    assert np.allclose(out, 3.5)
