"""SSH2-direct scorer + OAS pseudo-labeling — wraps the validated `ssh2cli` (original SSH2.0).

`ssh2cli` is the in-repo pure-Python SSH2.0 (original pretrained 3-SVM weights; validated on
Jain-131: sens 1.00 / spec 0.755 / acc 0.824). We use its **mean P(positive)** over the 3 SVMs —
the monotonic hydrophobic-interaction risk score — both as the SSH2-direct floor and as the OAS
soft pseudo-label for WeakSup. No retraining (GOALS.md §8, MEMORY.md "Current experiment").
"""
from __future__ import annotations

import sys

import numpy as np

from abdev.config import REPO_ROOT
from abdev.scorers.base import AntibodySeq, Scorer

_SSH2_DIR = REPO_ROOT / "ssh2cli"
_DATA_DIR = str(_SSH2_DIR / "data")


def _core():
    if str(_SSH2_DIR) not in sys.path:
        sys.path.insert(0, str(_SSH2_DIR))
    import ssh2_core  # noqa: E402

    return ssh2_core


def ssh2_p_positive(heavy, light, names=None) -> np.ndarray:
    """Mean P(positive) over the 3 SVMs for each (heavy, light) pair (HC+LC concatenated)."""
    core = _core()
    names = names if names is not None else [str(i) for i in range(len(heavy))]
    merged = [
        (str(n), (str(h).strip() + str(l).strip()).upper())
        for n, h, l in zip(names, heavy, light)
    ]
    res = core.predict(merged, _DATA_DIR)
    return np.array([r["P_positive"] for r in res], dtype=float)


class SSH2Direct(Scorer):
    """Floor model: the raw weak labeler's risk score (no GDPa1 training, no OAS)."""

    name = "SSH2-direct"
    inline = True

    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        return ssh2_p_positive([s.heavy for s in seqs], [s.light for s in seqs])
