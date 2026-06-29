"""Load cached chain embeddings (Track B output) → per-antibody feature matrices.

Shard schema (the agreed contract): columns [id, chain, pool, emb], emb a fixed-length vector.
  * ESM-2 backbones: chains {H, L} × pools {mean, cls}
  * AbLang2:         chains {H, L, P} × {mean}   (P = native paired representation)

A per-antibody feature = concatenation of the requested chains for one pool, in a fixed order.
Default chains: ESM-2 → (H, L); AbLang2 → (P,) i.e. its native paired vector.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abdev.config import DATA_DIR

EMB_DIR = DATA_DIR / "embeddings"


def _load_shards(backbone: str, dataset: str) -> pd.DataFrame:
    d = EMB_DIR / backbone / dataset
    shards = sorted(d.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no embedding shards in {d}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def load_features(backbone: str, dataset: str, pool: str = "mean", chains=None):
    """Return (ids: list[str], X: np.ndarray [n, d]) — per-antibody concatenated chain embeddings."""
    df = _load_shards(backbone, dataset)
    present = set(df.chain.unique())
    if chains is None:
        chains = ("P",) if "P" in present else ("H", "L")
    sub = df[df.pool == pool]
    vec = {(r.id, r.chain): np.asarray(r.emb, dtype=np.float32)
           for r in sub.itertuples(index=False)}
    ids, X = [], []
    for i in sub.id.drop_duplicates():
        try:
            X.append(np.concatenate([vec[(i, c)] for c in chains]))
            ids.append(str(i))
        except KeyError:
            continue  # antibody missing a required chain for this pool
    return ids, np.vstack(X)


def align_xy(ids, X, label_df, target: str = "hic"):
    """Align feature rows to a label DataFrame (columns id, <target>); return (X, y) in label order."""
    pos = {i: k for k, i in enumerate(ids)}
    rows, y = [], []
    for r in label_df.itertuples(index=False):
        if str(r.id) in pos:
            rows.append(X[pos[str(r.id)]])
            y.append(getattr(r, target))
    return np.vstack(rows), np.asarray(y, dtype=float)
