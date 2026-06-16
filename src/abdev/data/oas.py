"""OAS (Observed Antibody Space) access — Olsen et al., Protein Science 2022.

Two uses (GOALS.md §6-7):
  * backbone fine-tuning: unpaired heavy/light sequences (hundreds of millions available).
  * L2 SSH2.0 pseudo-labeling: a ~100k *paired* subset (compute-budget choice).

Requires the `oas` extra. A 100k paired subset is ~2-5 GB after preprocessing and lands
in a gitignored, non-Drive-synced cache.

TODO(phase2): implement structured subset download (by species/study/pairing) and a
streaming reader. Confirm the `oas_utils` package name/source first.
"""
from __future__ import annotations

DEFAULT_PAIRED_SUBSET_SIZE = 100_000


def download_paired_subset(n: int = DEFAULT_PAIRED_SUBSET_SIZE):
    raise NotImplementedError("Phase 2: OAS paired subset download.")
