"""GDPa1 / PROPHET-Ab loader (Arsiwala et al., mAbs 2025).

246 paired antibodies with 11 experimental biophysical features and provided 5-fold
hierarchical-cluster, IgG-isotype-stratified CV splits. This is our only labeled data —
all scorer CV must respect the provided splits to avoid leakage (GOALS.md §8).

TODO(phase0): confirm exact column names in GDPa1_v1.2_20250814.csv (sequence cols,
the 11 feature cols, and the split/fold column) and fill in the mappings below.
"""
from __future__ import annotations

import pandas as pd

from abdev.config import GDPA1_CSV

# TODO(phase0): verify against the actual CSV header.
HEAVY_COL = "VH"
LIGHT_COL = "VL"
FOLD_COL = "cv_fold"
FEATURE_COLS = (
    "HIC", "SMAC", "HAC", "PR_Ova", "PR_CHO",
    "SEC_%Monomer", "AC-SINS_pH6.0", "AC-SINS_pH7.4", "Tm1", "Tm2", "Titer",
)


def load_gdpa1() -> pd.DataFrame:
    """Load the raw GDPa1 table."""
    return pd.read_csv(GDPA1_CSV)


def load_splits():
    """Yield (train_df, test_df) per provided CV fold. TODO(phase0): implement."""
    raise NotImplementedError("Wire up FOLD_COL once the CSV schema is confirmed.")
