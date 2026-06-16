"""Developability evaluation metrics (GOALS.md §8).

Generated candidates are scored on: oracle HIC / AC-SINS, CamSol solubility, SAP aggregation,
AbLang-2 naturalness (PLL), and distributional metrics (sequence recovery, germline identity).
These reproduce/extend the legacy pipeline's eval (see legacy/developability_model/) so our
DFM results are directly comparable to the reported baseline numbers.

TODO(phase0): port oracle + CamSol + SAP scoring from legacy/; expose a single evaluate()
that takes generated sequences and returns the GOALS.md §8 metric table.
"""
from __future__ import annotations


def evaluate(generated, parents):
    raise NotImplementedError("Phase 0: port eval harness from legacy/.")
