"""The Scorer contract: sequence-in, score-out.

Every developability scorer (L0-L4 in GOALS.md §6) implements this interface so it is
drop-in compatible with posterior tempering (Eq. 8). Higher score = more developable.
Black-box scorers are fine — posterior tempering requires no differentiability.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AntibodySeq:
    """A paired antibody sequence (heavy + light variable domains)."""
    heavy: str
    light: str


class Scorer(ABC):
    """Maps antibody sequences to scalar developability scores (higher = better).

    Implementations must be batched and side-effect free. `inline` advertises whether
    the scorer is fast enough (<100 ms/seq) for guidance at every sampling step, vs.
    terminal-only evaluation (e.g. TAP's ~3 s structure prediction).
    """

    #: human-readable name, e.g. "L0:esm2-gdpa"
    name: str = "scorer"
    #: True if usable for inline guidance at every CTMC step
    inline: bool = True

    @abstractmethod
    def score(self, seqs: list[AntibodySeq]) -> np.ndarray:
        """Return shape-(N,) array of developability scores for N sequences."""
        raise NotImplementedError

    def __call__(self, seqs: list[AntibodySeq]) -> np.ndarray:
        return self.score(seqs)
