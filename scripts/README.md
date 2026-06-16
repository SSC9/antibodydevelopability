# scripts/

CLI entry points for the pipeline (added per phase). Planned:

- `finetune_backbone.py` — fine-tune EvoDiff on OAS (Phase 1)
- `pseudolabel_oas.py` — SSH2.0 pseudo-labeling of the OAS subset (Phase 2, L2)
- `train_scorer.py` — train/evaluate L0–L4 scorers on GDPa1
- `generate.py` — guided CTMC sampling with posterior tempering
- `evaluate.py` — score generated candidates → GOALS.md §8 metric table

Run with `uv run python scripts/<name>.py ...`.
