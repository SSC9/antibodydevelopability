"""Leakage-safe cross-validation for scorers on GDPa1 (GOALS.md §8).

At N=246, generalization estimates are fragile. Always use the provided 5-fold
hierarchical-cluster, isotype-stratified splits (or leave-one-out / repeated 5-fold); never
a naive random split. Report Spearman rho per target with confidence intervals.

TODO(phase2): implement CV runner returning per-fold + aggregate Spearman with CIs.
"""
from __future__ import annotations


def spearman_cv(scorer, df, target: str):
    raise NotImplementedError("Phase 2: leakage-safe CV evaluation.")
