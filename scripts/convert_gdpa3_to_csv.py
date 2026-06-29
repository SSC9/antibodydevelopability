"""Flatten the GDPa3 held-out xlsx into one directly-usable CSV (GDPa1-style layout).

Joins the "Sequences" and "Assay Data - average" sheets on antibody_id, renames the (already
variable-domain) light column to `vl_protein_sequence` for consistency with GDPa1, and keeps the
replicate-averaged assays (`*_avg`) + their std (`*_stddev`).

Run:  uv run python scripts/convert_gdpa3_to_csv.py
Out:  data/GDPa3_20260106.csv
"""
from __future__ import annotations

import pandas as pd

from abdev.config import DATA_DIR, GDPA3_XLSX

OUT = DATA_DIR / "GDPa3_20260106.csv"


def main() -> None:
    seq = pd.read_excel(GDPA3_XLSX, "Sequences")
    avg = pd.read_excel(GDPA3_XLSX, "Assay Data - average")

    seq = seq.rename(columns={"lc_protein_sequence": "vl_protein_sequence"})  # it's the variable domain
    id_seq_cols = ["antibody_id", "hc_subtype", "lc_subtype",
                   "vh_protein_sequence", "vl_protein_sequence"]
    assay_cols = [c for c in avg.columns if c.endswith("_avg") or c.endswith("_stddev")]

    df = seq[id_seq_cols].merge(avg[["antibody_id", *assay_cols]], on="antibody_id", how="left")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    print(f"  HIC (hic_rt_avg) present for {df['hic_rt_avg'].notna().sum()} antibodies")


if __name__ == "__main__":
    main()
