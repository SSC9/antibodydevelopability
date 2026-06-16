"""Developability scorers (GOALS.md §6). All implement abdev.scorers.base.Scorer.

    L0  esm2        ESM-2 embeddings + shallow regressor on GDPa1   (baseline)
    L1  physics     TAP / PROPERMAB physics-informed features
    L2  ssh2        SSH2.0 OAS-scale weak supervision               (core novelty)
    L3  ensemble    Deep ensemble + LCB guidance                    (uncertainty)
    L4  naturalness AbLang-2 pseudo-log-likelihood                  (naturalness regularizer)
"""
from abdev.scorers.base import AntibodySeq, Scorer

__all__ = ["Scorer", "AntibodySeq"]
