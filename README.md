# Antibody Developability via Discrete Flow Matching

> Generating therapeutic antibody sequences biased toward developability using **discrete flow
> matching (DFM)** with **posterior-tempered classifier guidance**, and a scorer bootstrapped
> from **OAS-scale weak supervision** to overcome the 246-label data bottleneck.

🎯 **Target:** NeurIPS 2026 workshops (SPIGM / AI for Science / FPI).

📄 **Read first:**
- [`GOALS.md`](GOALS.md) — project ground truth: objective, locked decisions, scorer ladder, roadmap.
- [`MEMORY.md`](MEMORY.md) — living progress log (update at the end of each work session).
- [`docs/antibody_dfm_working_doc.pdf`](docs/antibody_dfm_working_doc.pdf) — theory & method working document.

---

## Status

Active redevelopment. We are pivoting from the original BMI702 project (continuous conditional
flow matching in a PLS latent — see **Background** below) to **genuine discrete flow matching on
amino-acid tokens**. The original notebooks are preserved under [`legacy/`](legacy/) as a baseline.

Current phase: **Phase 0 — infra & baseline reproduction** (see `MEMORY.md` for the live checklist).

## Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/). **Important:** because this repo lives inside a
Google Drive folder, keep the virtualenv *outside* Drive so a multi-GB torch env doesn't sync:

```bash
# one-time per machine: point uv's venv outside Drive (add to your shell profile)
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/abdev"

cd antibodydevelopability
uv sync                       # core deps (scorers L0/L4, baselines, eval)
uv run pytest                 # smoke tests should pass

# per-phase extras, installed on demand:
uv sync --extra backbone      # EvoDiff DFM backbone (Phase 1)
uv sync --extra oas --extra features   # SSH2.0 / OAS pipeline (Phase 2)
uv sync --extra physics       # TAP / ImmuneBuilder (Phase 3, heavy)
```

> If `git` errors with an Xcode-license message, it's PATH ordering — use Homebrew's git:
> `export PATH=/opt/homebrew/bin:$PATH`.

## Layout

```
src/abdev/        # the package
  data/           #   GDPa1 + OAS loaders, CV splits
  backbone/       #   EvoDiff adapter, germline-absorbing source, CTMC sampler
  guidance/       #   posterior tempering (Eq. 8), composite scorer (Eq. 9)
  scorers/        #   L0 ESM-2 · L1 physics · L2 SSH2.0 · L3 ensemble · L4 AbLang2
  eval/           #   developability metrics + leakage-safe CV
scripts/          # CLI entry points (per phase)
notebooks/        # exploration only
data/             # GDPa1 CSV (tracked); caches/weights/OAS gitignored
legacy/           # original BMI702 notebooks (baseline)
docs/             # theory doc + course reports
```

---

## Background (BMI702 v1, now the baseline)

The original project (Sidhant Puntambekar & Jack Hwang, Dept. of Biomedical Informatics, HMS)
asked whether sequence-based protein language models can predict developability and enable
guided CDR3 optimization. Headline results (now our comparison anchors):

- **Prediction:** XGBoost on 11 biophysical features → approval AUPRC 0.665; Ridge on p-IgGen /
  AbLang2 embeddings for per-metric regression.
- **Generation:** oracle → PLS latent (480d→10d) → continuous CFM → AbLang2 reweighted decoding.
  Guided ≈ unguided (AbLang2's prior dominated at this data scale) — the gap we aim to close.

Full numbers and report PDFs: [`legacy/`](legacy/) and [`docs/course-reports/`](docs/course-reports/).

**Data:** GDPa1 / PROPHET-Ab (246 paired antibodies; Arsiwala et al., mAbs 2025); OAS
(Olsen et al., Protein Science 2022).

---

## AI assistance

AI coding assistants (Claude) were used to help with project scaffolding, refactoring, and
documentation. All research direction, methodological decisions, and final review are the
authors'.
