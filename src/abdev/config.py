"""Project paths and shared constants.

Paths are resolved relative to the repo root so the package works regardless of the
caller's CWD. Heavy artifacts (OAS subsets, model weights) live under DATA_DIR / CACHE_DIR,
which are gitignored and should NOT be Drive-synced.
"""
from __future__ import annotations

import os
from pathlib import Path

# repo root = two levels up from this file (src/abdev/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
# Default model/embedding cache; override with ABDEV_CACHE to point outside Drive.
CACHE_DIR = Path(os.environ.get("ABDEV_CACHE", REPO_ROOT / "data" / "cache"))

# Canonical labeled dataset (see GOALS.md §8). 246 paired antibodies (train).
GDPA1_CSV = DATA_DIR / "GDPa1_v1.2_20250814.csv"
# External held-out test (Ginkgo benchmark), 80 abs; disjoint from GDPa1 and SSH2.0/Jain.
GDPA3_XLSX = DATA_DIR / "GDPa3_20260106_full.xlsx"          # raw source (multi-sheet)
GDPA3_CSV = DATA_DIR / "GDPa3_20260106.csv"                # flattened, committed (scripts/convert_gdpa3_to_csv.py)
# Isotype-stratified CV fold — used ONLY for hyperparameter selection (headline test = GDPa3).
GDPA1_FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"

# Amino-acid alphabet (|A| = 20). Used by the discrete flow matching backbone.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Developability targets we optimize / evaluate (subset of the 11 GDPa1 features).
PRIMARY_TARGETS = ("HIC", "AC-SINS_pH7.4")

# Pretrained model identifiers.
ESM2_650M = "facebook/esm2_t33_650M_UR50D"   # primary / downstream backbone (1280-d)
ESM2_8M = "facebook/esm2_t6_8M_UR50D"        # exact abdev-benchmark anchor (320-d)
ESM2_MODEL_ID = ESM2_650M                    # default
ABLANG2_MODEL = "ablang2-paired"             # antibody-specific backbone + naturalness (PLL)
