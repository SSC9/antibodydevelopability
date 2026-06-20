# Project Memory / Progress Log

> **Purpose:** Living log of status, decisions, and next actions. Update this at the end of
> each work session. Stable goals/strategy live in [`GOALS.md`](./GOALS.md) — keep those there.
> Newest session entry at the top of the Changelog.

**Project:** Antibody Developability via Discrete Flow Matching
**Target:** NeurIPS 2026 workshops (SPIGM / AI for Science / FPI). See `GOALS.md`.

> The checklist below groups work by component for tracking. It is **not a fixed sequence** —
> the team works different parts in parallel and picks what makes sense to pursue.

---

## Current status (as of 2026-06-15)

**Phase 0 — Infra & baseline reproduction: IN PROGRESS. Scaffolding DONE & pushed to `main`.**

Done: alignment; `GOALS.md`/`MEMORY.md`; fork reorganized (legacy/ + docs/); uv env
(`pyproject.toml` + `uv.lock`, Python 3.11) synced to `~/.venvs/abdev`; `src/abdev/` package
skeleton with documented seams; smoke tests pass; committed + pushed (commit `8fabb5a`).
Remaining in Phase 0: confirm GDPa1 CSV schema + wire loaders/splits, port the eval harness
from `legacy/`, reproduce the existing baseline numbers.

## Key findings

- **2026-06-15 — SSH2.0↔GDPa1 contamination is total at the fold level.** SSH2.0's training set =
  Jain et al. 2017 (137 mAbs); **135/137 are in GDPa1** (→ 135/246 = 55% contaminated, 111 clean
  individuals). **No clean fold in any of the 3 CV schemes** (hierarchical_cluster, isotype-strat,
  random — all 0/5). GDPa1 HIC vs Jain HIC are rank-consistent (Spearman 0.97) but different scale
  (Pearson 0.70). HIC distribution of the SSH2.0 subset ≈ full GDPa1 (no target-range shift).
  See `notebooks/01_ssh2_gdpa1_contamination.ipynb`. → standard cluster-CV can't give an
  SSH2.0-naive test fold; must choose an evaluation path (individual-level holdout / custom germline
  regrouping / reframed comparison) before building the HIC oracle.

## Decisions log

- **2026-06-15** — Full pivot to discrete flow matching (drop continuous-CFM/PLS core; keep it as baseline). See `GOALS.md §4`.
- **2026-06-15** — Headline = DFM + posterior tempering **and** SSH2.0 weak supervision (two coupled claims).
- **2026-06-15** — Backbone = fine-tune EvoDiff (single-GPU budget); OAS subset ~100k paired.
- **2026-06-15** — Source distribution = germline-absorbing (default), compare vs masked-absorbing.
- **2026-06-15** — New work in clean `src/abdev/` package; legacy BMI702 notebooks quarantined under `legacy/`.
- **2026-06-15** — Build directly in the `SSC9/antibodydevelopability` fork (already ours), cloned into this Drive folder. GOALS/MEMORY docs + PDF move into the repo.
- **2026-06-15** — Co-dev infra: GitHub for collaboration; `.venv`, `data/`, model weights gitignored (never Drive-synced); commit `uv.lock`.

## Roadmap checklist

### Phase 0 — Infra & baseline reproduction
- [x] Set up uv environment (`pyproject.toml` + `uv.lock`, venv at `~/.venvs/abdev`)
- [x] Create `src/abdev/` package skeleton + `notebooks/`, `scripts/`, smoke tests
- [ ] Confirm GDPa1 CSV schema; wire `data/gdpa1.py` loaders + provided CV splits
- [ ] Port eval harness (oracle, CamSol, SAP) from `legacy/developability_model/`
- [ ] Reproduce existing baseline numbers (sanity check)

### Phase 1 — DFM backbone (headline a)
- [ ] EvoDiff adapter + OAS fine-tuning spike (confirm single-GPU feasibility)
- [ ] Germline-absorbing source distribution
- [ ] CTMC sampler with posterior-tempering hook
- [ ] Replicate Zhao-style guided generation with L0 scorer under DFM

### Phase 2 — Core scorers (headline b + secondary)
- [ ] L0: ESM-2 (650M) embeddings + shallow regressor on GDPa1
- [ ] L2: retrain SSH2.0 SVM locally (CKSAAGP, 131 seqs)
- [ ] L2: pseudo-label ~100k OAS paired subset
- [ ] L2: pretrain ESM-2 regressor (BCE) → fine-tune (MSE) on GDPa1; ablate vs L0
- [ ] L4: AbLang2 PLL naturalness regularizer + α sweep

### Phase 3 — Uncertainty / physics (stretch)
- [ ] L1: PROPERMAB sequence features (incl. SAP)
- [ ] L3: deep ensemble (N=5) + LCB guidance

### Phase 4 — Writeup
- [ ] Ablation tables + figures
- [ ] Paper draft

### Deferred
- [ ] Pareto-front optimization (HIC vs AC-SINS), proposal §7 — 2nd paper

## Now / Next / Blocked

- **Now:** Alignment complete; `GOALS.md` + `MEMORY.md` written; infra decisions locked.
- **Next (blocked on git):** Re-clone the `SSC9/antibodydevelopability` fork properly with git,
  move docs+PDF in, reorganize legacy notebooks, add `pyproject.toml` + `.gitignore` + `src/abdev/`
  skeleton, `uv sync`, commit, push.
- **Not blocked.** The earlier "git broken" was just PATH ordering: `/usr/bin/git` (Apple shim,
  needs Xcode license) preceded Homebrew's git in the non-interactive shell. Use `/opt/homebrew/bin/git`
  (export `PATH=/opt/homebrew/bin:$PATH`). Verified: git 2.51.0, `gh` authed as `SSC9` (repo+workflow
  scopes, can push to fork), uv 0.8.22. No Xcode license needed.

## Open questions (mirrors GOALS §12)

- EvoDiff↔OAS fine-tuning feasible on a single GPU? (early spike)
- SSH2.0 local reproduction details (131-seq set, CKSAAGP via iFeatureOmega/pfeature).
- OAS download path via `oas_utils` for ~100k paired subset.
- Standardize on GDPa1 `v1.2_20250814`?
- Can our scorer beat the AbLang2 prior (existing repo's guided≈unguided null result)?

## Environment notes

- Working repo = the `SSC9/antibodydevelopability` fork, cloned into the Drive folder at
  `.../BIONICs Lab.../antibody/antibodydevelopability/`. We commit + push to `main`.
- Legacy BMI702 notebooks → `legacy/`; course report PDFs → `docs/course-reports/`.
- Proposal/theory doc: `docs/antibody_dfm_working_doc.pdf`. Ground truth: `GOALS.md`.
- **uv venv lives OUTSIDE Drive** at `~/.venvs/abdev` (set `UV_PROJECT_ENVIRONMENT`), so a
  multi-GB torch venv never Drive-syncs. Collaborators must export the same var (see README).
- git: use Homebrew's (`export PATH=/opt/homebrew/bin:$PATH`); `/usr/bin/git` errors on Xcode license.

## Changelog

### 2026-06-15 — Session 1 (alignment + Phase 0 scaffolding)
- Read proposal PDF (`antibody_dfm_working_doc.pdf`) and cloned/reviewed original repo.
- Identified the gap: repo = continuous CFM in PLS latent; proposal = discrete flow matching on tokens.
- Aligned on the strategic decisions (see Decisions log / `GOALS.md §4`).
- Wrote `GOALS.md` (ground truth) and `MEMORY.md` (this log).
- Reorganized the fork (legacy/, docs/course-reports/), added uv env + `src/abdev/` skeleton +
  smoke tests; committed `8fabb5a` and pushed to `main`.
- **Next session:** confirm GDPa1 schema → wire `data/gdpa1.py`; port eval harness from `legacy/`;
  reproduce baseline numbers. Then Phase 1 (EvoDiff fine-tuning spike).

### 2026-06-15 — Session 2 (SSH2.0↔GDPa1 contamination audit)
- Reframed GOALS §11 as unordered workstreams; dropped internal-deadline framing (commit 764135a).
- Discussed DFM↔oracle intuition and the weak-supervision experiment design (leakage, CV, baselines).
- Built + executed `notebooks/01_ssh2_gdpa1_contamination.ipynb`. Result: **total fold-level
  contamination** (0/5 clean folds in all schemes); HIC rank-consistent across sources, diff scale.
  See Key findings above.
- **Next:** choose the oracle-evaluation path given no clean fold (individual-level holdout vs custom
  germline regrouping vs reframed comparison), then build the HIC oracle (L0 vs L2 vs SSH2.0-direct).
