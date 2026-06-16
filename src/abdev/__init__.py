"""abdev — Antibody developability via discrete flow matching.

A discrete flow matching (DFM) generative framework that samples antibody sequences
from the natural distribution while biasing toward developability via posterior-tempered
classifier guidance. See GOALS.md for the full project ground truth.

Package map:
    abdev.data       — GDPa1 + OAS loaders and CV splits
    abdev.backbone   — EvoDiff adapter, germline-absorbing source, CTMC sampler
    abdev.guidance   — posterior tempering (Eq. 8) and the composite scorer (Eq. 9)
    abdev.scorers    — L0 ESM-2, L1 physics, L2 SSH2.0, L3 ensemble, L4 AbLang2
    abdev.eval       — developability metrics (HIC/AC-SINS oracle, CamSol, SAP, PLL)
"""

__version__ = "0.1.0"
