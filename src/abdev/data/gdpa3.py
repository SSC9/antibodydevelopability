"""GDPa3 loader → normalized schema: columns [id, vh, vl, hic].

GDPa3 (`GDPa3_20260106_full.xlsx`) is the Ginkgo benchmark's external **held-out test** set
(80 abs), verified disjoint from GDPa1 and from SSH2.0's Jain131 training set — so it's a clean
external test for SSH2.0-derived models (GOALS.md §8, MEMORY.md).

Two sheets are joined on `antibody_id`:
  * "Sequences" — variable VH (`vh_protein_sequence`) + variable VL. NOTE: the light column is
    named `lc_protein_sequence` but is the **variable** domain (~108 aa, no constant region —
    verified by length), i.e. equivalent to GDPa1's `vl_protein_sequence`.
  * "Assay Data - average" — replicate-averaged `hic_rt_avg` (use directly; no manual averaging).

One antibody is missing HIC → 79 usable.
"""
from __future__ import annotations

import pandas as pd

from abdev.config import GDPA3_XLSX

SEQ_SHEET = "Sequences"
AVG_SHEET = "Assay Data - average"


def load_gdpa3(target: str = "hic_rt_avg") -> pd.DataFrame:
    """Return GDPa3 as [id, vh, vl, hic]. `target` is an averaged assay column (default hic_rt_avg)."""
    seq = pd.read_excel(GDPA3_XLSX, SEQ_SHEET)
    avg = pd.read_excel(GDPA3_XLSX, AVG_SHEET)

    out = pd.DataFrame(
        {
            "id": seq["antibody_id"].astype(str),
            "vh": seq["vh_protein_sequence"].astype(str).str.strip().str.upper(),
            "vl": seq["lc_protein_sequence"].astype(str).str.strip().str.upper(),  # variable light
        }
    )
    hic = avg[["antibody_id", target]].rename(columns={"antibody_id": "id", target: "hic"})
    hic["id"] = hic["id"].astype(str)
    out = out.merge(hic, on="id", how="left")
    return out.dropna(subset=["vh", "vl", "hic"]).reset_index(drop=True)
