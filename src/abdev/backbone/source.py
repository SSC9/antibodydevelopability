"""Source distribution p_0 for the discrete flow path (GOALS.md §4, proposal §3.4).

Default = germline-absorbing: intermediate noisy states are valid antibody sequences at
varying somatic-hypermutation levels, keeping the trajectory on the data manifold even under
aggressive guidance. We also support masked-absorbing (recovers masked diffusion) for ablation.

TODO(phase1): germline lookup (per V/J gene) + the conditional path p_t(x|x_1) (Eq. 4).
"""
from __future__ import annotations

from enum import Enum


class SourceKind(str, Enum):
    GERMLINE_ABSORBING = "germline"   # recommended default
    MASKED_ABSORBING = "masked"       # ablation: == masked diffusion
    UNIFORM = "uniform"


def sample_source(kind: SourceKind, x1):
    raise NotImplementedError("Phase 1: source distribution sampling.")
