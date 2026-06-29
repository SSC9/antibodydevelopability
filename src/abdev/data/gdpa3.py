"""GDPa3 loader → normalized schema: columns [id, vh, vl, hic].

GDPa3 is the Ginkgo benchmark's external **held-out test** set (80 abs), verified disjoint from
GDPa1 and from SSH2.0's Jain131 training set — a clean external test for SSH2.0-derived models
(GOALS.md §8, MEMORY.md).

We read the flattened CSV `data/GDPa3_20260106.csv` (produced by `scripts/convert_gdpa3_to_csv.py`
from the raw multi-sheet xlsx). VH/VL are variable domains; the original light column
(`lc_protein_sequence`, ~108 aa, no constant region) was renamed `vl_protein_sequence` during
conversion. HIC = `hic_rt_avg` (replicate-averaged, provided). One antibody lacks HIC → 79 usable.
"""
from __future__ import annotations

import pandas as pd

from abdev.config import GDPA3_CSV


def load_gdpa3(target: str = "hic_rt_avg") -> pd.DataFrame:
    """Return GDPa3 as [id, vh, vl, hic]. `target` is an averaged assay column (default hic_rt_avg)."""
    df = pd.read_csv(GDPA3_CSV)
    out = pd.DataFrame(
        {
            "id": df["antibody_id"].astype(str),
            "vh": df["vh_protein_sequence"].astype(str).str.strip().str.upper(),
            "vl": df["vl_protein_sequence"].astype(str).str.strip().str.upper(),
            "hic": pd.to_numeric(df[target], errors="coerce"),
        }
    )
    return out.dropna(subset=["vh", "vl", "hic"]).reset_index(drop=True)
