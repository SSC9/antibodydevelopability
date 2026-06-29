"""Pseudo-label an OAS paired subset with SSH2.0 (via ssh2cli) → soft labels for WeakSup pretraining.

Consumes the collaborator's OAS subset (Track B; columns id/heavy/light) and writes SSH2.0's
mean P(positive) per sequence. Checkpointed: writes per-shard parquet and resumes by skipping
shards that already exist (long jobs must be resumable — MEMORY.md "Working preferences").

Run:
  uv run python scripts/pseudolabel_oas.py \
      --input data/oas/oas_paired_human_100k.parquet \
      --out   data/pseudolabels/ssh2_oas \
      --shard-size 5000

Requires the `oas` extra for parquet I/O:  uv sync --extra oas
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from abdev.scorers.ssh2 import ssh2_p_positive


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="OAS subset parquet with id/heavy/light columns.")
    ap.add_argument("--out", default="data/pseudolabels/ssh2_oas", help="Output shard directory.")
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--heavy-col", default="heavy")
    ap.add_argument("--light-col", default="light")
    a = ap.parse_args()

    df = pd.read_parquet(a.input)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(df)
    n_shards = (n + a.shard_size - 1) // a.shard_size
    for s in range(n_shards):
        shard = out / f"shard_{s:05d}.parquet"
        if shard.exists():
            print(f"skip {shard.name} (exists)")
            continue
        chunk = df.iloc[s * a.shard_size : (s + 1) * a.shard_size]
        pp = ssh2_p_positive(
            chunk[a.heavy_col].tolist(),
            chunk[a.light_col].tolist(),
            chunk[a.id_col].astype(str).tolist(),
        )
        pd.DataFrame(
            {"id": chunk[a.id_col].astype(str).values, "ssh2_p_positive": pp}
        ).to_parquet(shard, index=False)
        print(f"wrote {shard.name} ({len(chunk)} seqs)  [{s + 1}/{n_shards}]")
    print(f"done → {out}")


if __name__ == "__main__":
    main()
