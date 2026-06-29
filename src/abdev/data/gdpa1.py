"""GDPa1 loader → normalized schema: columns [id, vh, vl, hic, fold].

GDPa1 / PROPHET-Ab (Arsiwala et al., mAbs 2025): 246 paired antibodies with experimental
developability assays + the benchmark's CV folds. This is our **training** set.

We expose variable-domain VH/VL (the developability/CDR signal lives there, and it matches
GDPa3/OAS) and the isotype-stratified fold. The fold is for **hyperparameter selection only** —
the headline generalization claim is on the external GDPa3 held-out (GOALS.md §8).
"""
from __future__ import annotations

import pandas as pd

from abdev.config import GDPA1_CSV, GDPA1_FOLD_COL


def load_gdpa1(target: str = "HIC", dropna_target: bool = True) -> pd.DataFrame:
    """Return GDPa1 as [id, vh, vl, hic, fold]. `target` is a GDPa1 assay column (default HIC)."""
    df = pd.read_csv(GDPA1_CSV)
    out = pd.DataFrame(
        {
            "id": df["antibody_id"].astype(str),
            "vh": df["vh_protein_sequence"].astype(str).str.strip().str.upper(),
            "vl": df["vl_protein_sequence"].astype(str).str.strip().str.upper(),
            "hic": pd.to_numeric(df[target], errors="coerce"),
            "fold": df[GDPA1_FOLD_COL].astype(int),
        }
    )
    out = out[out.vh.str.len() > 0]
    if dropna_target:
        out = out.dropna(subset=["hic"])
    return out.reset_index(drop=True)
