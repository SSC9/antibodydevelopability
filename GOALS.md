# Antibody Developability via Discrete Flow Matching — Project Ground Truth

> **Purpose of this file:** The single source of truth for *what* we are building and *why*.
> It is stable across Claude/work sessions. Update it when a strategic decision changes.
> Day-to-day progress, status, and next actions live in [`MEMORY.md`](./MEMORY.md), not here.

**Last reviewed:** 2026-06-15

---

## 1. One-line objective

Build a **discrete flow matching (DFM)** generative framework that samples therapeutic
antibody sequences from the natural antibody distribution while **biasing them toward
favorable developability**, using **posterior-tempered classifier guidance** and a scorer
**bootstrapped from OAS-scale weak supervision** to overcome the scarcity of labeled
developability data.

## 2. Motivation

Therapeutic antibodies must satisfy two largely independent criteria: high-affinity binding
**and developability** (aggregation propensity, colloidal/thermal stability, hydrophobicity,
self-interaction, immunogenicity risk). Historically developability is a post-hoc filter, not
a design objective → high late-stage attrition. We want to make developability a *generative*
objective while preserving naturalness (sequences must stay on the antibody manifold to be
therapeutically relevant).

**Key challenges:**
1. Sequence space is discrete (|A| = 20) — continuous diffusion is a poor fit.
2. Labeled developability data is scarce — GDPa1 has only **246 paired antibodies**.
3. Guidance must preserve naturalness — optimizing a score off-manifold is useless.

## 3. Positioning vs. prior work

- **Zhao et al. 2025** (arXiv:2507.02670, "Guided Generation for Developable Antibodies") —
  masked discrete diffusion on OAS + **SVDD** soft-value decoding. **This is our primary baseline.**
- **Our claim:** replacing SVDD with **DFM + posterior-tempered guidance** is more principled
  and data-efficient (one scorer call/step vs. M look-ahead rollouts; no value-function training
  on scarce labels), and the guidance scorer can be bootstrapped from **OAS-scale weak labels**.

## 4. Locked strategic decisions (2026-06-15)

| Decision | Choice | Rationale |
|---|---|---|
| **Generative core** | **Full pivot to discrete flow matching on tokens** | The existing repo does *continuous* CFM in a PLS latent w/ AbLang2 decoding; the novelty and clean story require genuine discrete generation on the amino-acid alphabet. |
| **Existing pipeline** | Demote to **baseline + reuse eval harness / GDPa1 tooling** | Don't throw away the evaluation code, data loaders, oracle, and reported numbers — they anchor our comparisons. |
| **Repo lineage** | **Build directly in the `SSC9/antibodydevelopability` fork** (already ours), cloned into this Drive folder | Continue existing lineage/attribution; add new code alongside, move legacy notebooks aside. |
| **Co-dev infra** | GitHub for collaboration; **`.venv`, `data/`, model weights gitignored** (never Drive-synced); `uv.lock` committed | Avoid Drive sync-churn on the venv; reproducible env via `uv sync`. |
| **Headline contributions** | (a) **DFM + posterior tempering** framework; (b) **SSH2.0 weak supervision** scorer (the proposal's "core novel contribution") | Two coupled claims forming one story: the framework needs a good scorer; the scorer is what makes guidance work under 246 labels. |
| **Backbone** | **Fine-tune EvoDiff** on OAS (not retrain from scratch) | Single-GPU/Colab Pro budget. Novelty is in scorer + guidance, not a new generative architecture. |
| **Source distribution** | **Germline-absorbing** (default), compare vs. masked-absorbing | Keeps intermediate states biologically valid (varying somatic hypermutation); strong inductive bias at no extra compute. |
| **OAS subset** | **~100k paired human sequences** for pseudo-labeling, deduped vs **GDPa1 ∪ Jain131 ∪ GDPa3** | Compute budget; dedup keeps SSH2.0 *generalizing* (not regurgitating training antibodies) and protects the held-out test. |
| **HIC-oracle eval / leakage control** | **External held-out on GDPa3** (80 abs; verified 0 sequence overlap with GDPa1 *and* with SSH2.0's Jain131 set). Train on all GDPa1 (+OAS weak-sup), test once on GDPa3. | A clean external test **dissolves the SSH2.0 ⊂ GDPa1 contamination** — no per-fold retraining needed. Test ∉ any training set ⇒ no leakage; GDPa3 is the community benchmark's own held-out ⇒ directly comparable. |
| **Compute** | Single GPU / Colab Pro | Constrains backbone + OAS subset size as above. |
| **Target venues** | NeurIPS 2026 workshops (see §9) | Submission-shaped scope (~8 pages, focused contribution). |

## 5. Technical framework (reference)

Full derivations are in `docs/antibody_dfm_working_doc.pdf`. Summary:

- **Discrete flow matching:** learn a rate matrix `R_t ∈ R^{|A|×|A|}` of a continuous-time
  Markov chain so that `p_0` = source and `p_1` = p_data (natural antibodies). A denoiser
  `f_θ(x_t, t)` predicts the clean sequence `x_1`; trained with cross-entropy (Eq. 7 in the doc);
  at inference `f_θ` plugs into the marginal rate (Eq. 6) and the CTMC is simulated `t: 0→1`.
- **Posterior tempering (guidance):** reweight the denoiser's predicted clean-token distribution
  `p̃(x_1 | x_t, t) ∝ f_θ(x_1|x_t,t) · exp(λ·s(x_1))`. Properties: (1) no differentiability
  required — `s` can be any black box; (2) no look-ahead sampling (cheaper than SVDD);
  (3) tunable naturalness–developability tradeoff via `s = s_dev + α·log p_AbLang`.
- **Position independence / epistasis:** standard factorized CTMC treats positions
  conditionally independent given `(x_t,t)`; epistatic CDR coupling is an acknowledged
  approximation and a possible discussion/limitation point.

## 6. Scorer design ladder (proposal §5)

| Level | Scorer | Labeled data | Inline (<100ms)? | Role for us |
|---|---|---|---|---|
| **L0** | ESM-2 (`facebook/esm2_t33_650M_UR50D`) + shallow regressor on GDPa1 | 246 pts | Yes | **Baseline** — the comparison point with Zhao et al. |
| **L1** | Physics features: TAP (via ImmuneBuilder/ABodyBuilder2) and PROPERMAB (iFeatureOmega, incl. SAP) | None | Partial (TAP ~3s = terminal only; PROPERMAB inline) | Ablation — physics-informed ML |
| **L2** | **SSH2.0 → GDPa1**: retrain SSH2.0 SVM locally (CKSAAGP), pseudo-label ~100k OAS, pretrain ESM-2 regressor (BCE), fine-tune (MSE) on GDPa1 | OAS subset | Yes | **CORE NOVELTY** — weak supervision at scale |
| **L3** | Deep ensemble (N=5) → LCB guidance `s = μ − β·σ` (Eq. 10) | same | Yes | Uncertainty-aware guidance (stretch) |
| **L4** | AbLang-2 pseudo-log-likelihood as naturalness regularizer | none (pretrained) | Yes | Naturalness–dev tradeoff sweep (secondary result) |

## 7. What we inherit vs. replace (legacy code now under `legacy/`)

**Reuse / port:**
- GDPa1 data + splits: `data/GDPa1_v1.2_20250814.csv` (5-fold hierarchical-cluster, isotype-stratified).
- Developability eval harness: oracle, CamSol/SAP scoring, error analysis
  (`legacy/developability_model/`, `legacy/developability_model/binding_preservation.ipynb`).
- AbLang2 utilities (embeddings, PLL, LoRA fine-tuning) and ESM/p-IgGen embedding code.
- Reported baseline numbers (see §8) as comparison anchors.

**Replace:**
- Continuous CFM in PLS latent (`legacy/generative_model/`) → discrete flow matching on tokens.
- Oracle-reranking guidance → posterior tempering in token space.
- Scratch-trained flow net → fine-tuned EvoDiff backbone with germline-absorbing source.

**Note:** the repo is messy (`antigen_v2/v3/v4`, `*_revised (1)(1).ipynb`, duplicate `final/`
dirs, old/new baselines). We build **directly in the fork** but quarantine the legacy notebooks
under `legacy/` and put all new work in a clean `src/abdev/` package (see §10), so the messy
history is preserved for reference without polluting the new codebase.

## 8. Baselines & evaluation

**Baselines to beat / compare:**
1. **Zhao et al.** (masked diffusion + SVDD) — primary conceptual baseline.
2. **Existing repo** continuous-CFM pipeline (n=50 held-out), reported:
   | Metric | Parent | Baseline (AbLang2) | Guided (Flow+AbLang2) |
   |---|---|---|---|
   | Oracle HIC (↓) | 2.738 | 2.466 | 2.493 |
   | Oracle AC-SINS (↓) | 3.741 | −0.545 | −0.441 |
   | CamSol (↑) | −0.004 | 0.564 | 0.649 |
   | SAP (↓) | 0.449 | 0.216 | 0.242 |
   *(Their finding: guided ≈ unguided → AbLang2 prior dominates at this data scale. Beating
   this gap is a concrete target — show DFM+L2 guidance moves the needle beyond the LM prior.)*

**Evaluation axes:** oracle HIC / AC-SINS; CamSol solubility; SAP; AbLang2 naturalness (PLL);
distributional metrics (sequence recovery, germline identity); binding preservation.

**Held-out test set — GDPa3** (`data/GDPa3_20260106_full.xlsx`, 80 abs, same PROPHET-Ab assays
incl. `hic_rt` on the GDPa1 scale). It is the Ginkgo benchmark's own held-out set and is verified
**disjoint** from GDPa1 and from SSH2.0's Jain131 set — our clean external test (see `MEMORY.md`
for the full HIC-oracle model table + leakage argument).

**Stats discipline:** N=246 is tiny — use the provided isotype-stratified CV folds for
model/hyperparameter selection only; make the headline generalization claim on the **external
GDPa3 held-out** (train on all GDPa1, test once), with bootstrap CIs. Metric: Spearman ρ (+ top-10%
recall). This replaces the earlier per-fold CV-leakage workaround.

## 9. Target venues (NeurIPS 2026 workshops)

Candidate workshops below. 2026 dates are unconfirmed; the 2025 deadlines are listed only as
rough external reference, **not** as an internal schedule.

| Workshop | 2025 deadline (ref. only) | Notes |
|---|---|---|
| **SPIGM** — Structured Probabilistic Inference & Generative Modeling | Aug 29, 2025 | Best fit. Chatterjee Lab submitted Entangled Schrödinger Bridge Matching here in 2025. |
| **AI for Science: Reach and Limit of AI for Scientific Discovery** | Aug 25, 2025 | Marinka Zitnik possibly co-chair. |
| **FPI** — Frontiers in Probabilistic Inference: Learning meets Sampling | Sep 1, 2025 | Strong fit for the DFM/CTMC sampling angle. |

## 10. Target repository structure (to be built)

Working copy = the **`SSC9/antibodydevelopability` fork cloned into this Drive folder**. New code,
the uv env, and these docs all live inside it (committed + shared via GitHub).

```
antibodydevelopability/           # the fork — THE working repo (cloned into Drive)
├── README.md                     # original; to be updated for the DFM rewrite
├── GOALS.md                      # this file (ground truth) — moved into the repo
├── MEMORY.md                     # living progress log — moved into the repo
├── docs/antibody_dfm_working_doc.pdf   # proposal / theory doc
├── pyproject.toml                # uv-managed env
├── uv.lock                       # committed for reproducibility
├── .venv/                        # GITIGNORED — local only, never synced/committed
├── data/                         # GITIGNORED (keep GDPa1 csv; OAS + weights downloaded on demand)
├── src/abdev/                    # new clean package
│   ├── data/                     #   OAS + GDPa1 loaders, splits
│   ├── backbone/                 #   EvoDiff adapter, germline-absorbing source, CTMC sampler
│   ├── guidance/                 #   posterior tempering, composite scorer
│   ├── scorers/                  #   L0 ESM-2, L1 physics, L2 SSH2.0, L3 ensemble, L4 AbLang2
│   └── eval/                     #   metrics ported from legacy code
├── notebooks/                    # exploration only (not the source of truth)
├── scripts/                      # CLI entry points for training/sampling/eval
└── legacy/                       # original BMI702 notebooks (baseline_model/, generative_model/, …)
```

*(Current physical state: this Drive folder holds `GOALS.md`, `MEMORY.md`, the PDF, and a
non-git tarball at `./repo/`. Once `git` is unblocked we re-clone the fork properly, move the
docs + PDF into it, reorganize legacy notebooks, and delete the tarball.)*

## 11. Workstreams (not a schedule)

These are the **components of work, deliberately unordered** — the team is tackling different
parts in parallel and decides what makes sense to pursue when. "Core" vs "stretch" denotes
importance to the headline claims, **not** sequence. Live status is tracked in `MEMORY.md`.

- **Developability oracle / scorers (core).** L0 ESM-2 baseline; **L2 SSH2.0 weak-supervision
  pipeline** ablated vs L0; L4 AbLang2 naturalness + α sweep. Supports headline claim (b) and the
  naturalness–developability result. *(Can be developed and validated entirely independently of
  the generative model — see §8.)*
- **DFM backbone (core).** Adapt + fine-tune EvoDiff on OAS; germline-absorbing source; CTMC
  sampler with a posterior-tempering hook; replicate Zhao-style guided generation under DFM
  sampling. Supports headline claim (a).
- **Infra & baseline reproduction.** uv env, `src/abdev` package, GDPa1 loaders + eval harness
  ported from `legacy/`, reproduce the existing baseline numbers as a sanity check.
- **Uncertainty & physics (stretch).** L1 PROPERMAB features; L3 deep ensemble + LCB guidance.
- **Writeup.** Ablations, figures, paper draft.

**Coupling (dependencies, not timing):** the oracle and the DFM backbone are independent and only
meet at guided-sampling/eval time. The oracle is a self-contained supervised-regression problem
that can be validated before any generative work. Baseline reproduction is useful early but is
not a hard gate on anything.
- **(Deferred / 2nd paper)** Pareto-front optimization (proposal §7).

## 12. Open questions / risks

- **EvoDiff ↔ OAS fine-tuning feasibility on single GPU** — needs an early spike to confirm.
- **SSH2.0 reproduction** — local SVM retrain on the published 131-seq set; web server is fragile.
- **OAS access** — `oas_utils` for a ~100k paired subset (~2–5 GB); confirm download path.
- **GDPa1 version** — repo has `v1.2_20250814`; confirm it's the version we standardize on.
- **Does posterior tempering beat the AbLang2 prior?** — the existing repo's null result (guided ≈
  unguided) is the risk to disprove; our scorer must add signal beyond the LM.

## 13. Key references

Zhao et al. 2025 (arXiv:2507.02670); GDPa1/PROPHET-Ab (Arsiwala et al., mAbs 2025);
OAS (Olsen et al., Protein Science 2022); SVDD (Li et al., arXiv:2408.08252);
Campbell et al. ICML 2024 (discrete flows); EvoDiff (Alamdari et al. 2023);
ESM-2 (Lin et al., Science 2023); AbLang-2 (Olsen et al. 2023);
SSH2.0 (Frontiers in Genetics 2022); PROPERMAB (bioRxiv 2024); TAP (Raybould et al., PNAS 2019);
germline-absorbing diffusion (Sanders et al., arXiv:2605.06720).
