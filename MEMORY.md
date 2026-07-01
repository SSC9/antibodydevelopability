# Project Memory / Progress Log

> **Purpose:** Living log of status, decisions, and next actions. Update this at the end of
> each work session. Stable goals/strategy live in [`GOALS.md`](./GOALS.md) — keep those there.
> Newest session entry at the top of the Changelog.

**Project:** Antibody Developability via Discrete Flow Matching
**Target:** NeurIPS 2026 workshops (SPIGM / AI for Science / FPI). See `GOALS.md`.

> The checklist below groups work by component for tracking. It is **not a fixed sequence** —
> the team works different parts in parallel and picks what makes sense to pursue.

---

## Current status (as of 2026-06-30)

**HIC weak-supervision oracle experiment: COMPLETE.** Verdict below; results committed (`3b906af`).
Infra (uv env, package, GDPa1/GDPa3/OAS data + embeddings + pseudo-labels) all in place.

### ▶ HANDOFF — read this first to continue in a new chat
- **Env:** `export PATH=/opt/homebrew/bin:$PATH; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/abdev"`;
  then `uv run ...`. Repo = the `SSC9/antibodydevelopability` fork in this Drive folder. Ground
  truth: `GOALS.md` + this file. Commit often, `git fetch` before push (Drive corrupts `.git`; if
  `git status` shows all-deleted, run `rm -f .git/index.lock && git reset HEAD`).
- **VERDICT (HIC oracle):** On the leakage-free external **GDPa3** test — OAS SSH2.0 weak-supervision
  pretraining **helps only in the low-data regime** (mean lift over backbones **+0.115 @ n=25**,
  +0.05 @ n=50, ~0 by n≥100, ~0 at full data). **MSE pretraining > BCE** (`mse_keepall`/`mse_reinit`
  carry the lift; `bce_reinit` ≈ 0). **Best absolute oracle = GoldOnly-Ridge, esm2-650m, CLS, no-std,
  α=1.0 ≈ 0.51** (reproduces benchmark `esm2_ridge`=0.403 exactly on esm2-8m/mean). SSH2-direct
  floor=0.175. Pretraining is validated (learns P_positive ρ=0.80 on OAS; pretrain-only→HIC ρ=0.21–0.30).
- **Final-oracle recommendation:** ESM2-650m · **CLS** · **Ridge** (no-std, α=1.0) on all GDPa1.
  WeakSup (MSE) only for scarce-label settings (<~100 labels).
- **KEY CAVEAT / next experiment:** we only pretrained a **shallow MLP on FROZEN embeddings** — the
  representation never changed. The traditional weak-sup win comes from **fine-tuning the encoder**.
  → **NEXT: LoRA fine-tune ESM-2 on the 100k OAS SSH2.0 labels, then on GDPa1 HIC, eval GDPa3**
  (matched control = GoldOnly-LoRA = LoRA on GDPa1 only). Needs GPU → Colab/notebook; `peft` dep;
  start with esm2-8m. This is the version most likely to move the *full-data* number.
- **How to reproduce:** `uv run python scripts/run_hic_experiment.py` (resumable → `results/hic_results.csv`);
  figures via `from abdev import viz; viz.lift_figure(); viz.data_efficiency_figure(); viz.summary_bar()`;
  narrative `notebooks/02_hic_oracle_experiment.ipynb`. Headline fig = `results/figures/lift.png`.
- **Key files:** `src/abdev/{weaksup,oracle,viz,eval/metrics,data/embeddings,data/gdpa1,data/gdpa3,scorers/ssh2}.py`;
  `scripts/{run_hic_experiment,pseudolabel_oas,convert_gdpa3_to_csv}.py`; `ssh2cli/` = SSH2.0 (P_positive).
- **Data:** GDPa1 (train, `data/GDPa1_v1.2_20250814.csv`, + `fold` col), GDPa3 (external test,
  `data/GDPa3_20260106.csv`), OAS embeddings `data/embeddings/{esm2-8m,esm2-650m,ablang2}/{gdpa1,gdpa3,oas}/`
  (long schema id/chain/pool/emb; ESM2 chains H,L pools mean,cls; AbLang2 chains H,L,P pool mean — use
  **P** as primary), OAS pseudo-labels `data/pseudolabels/ssh2_oas/`, OAS seqs `data/oas/oas_paired_human_100k.parquet`.
- **AbLang2 note:** underperforms ESM-2 for HIC (0.21–0.30) — this is **benchmark-confirmed + expected**
  (naturalness ≠ biophysics), not a bug; paired-P is correct. Its Ridge (0.21) < benchmark (0.356) →
  seqcoding extraction possibly improvable (ablation backbone; doesn't affect the verdict).

## Key findings

- **2026-06-15 — SSH2.0↔GDPa1 contamination is total at the fold level.** SSH2.0's training set =
  Jain et al. 2017 (137 mAbs); **135/137 are in GDPa1** (→ 135/246 = 55% contaminated, 111 clean
  individuals). **No clean fold in any of the 3 CV schemes** (hierarchical_cluster, isotype-strat,
  random — all 0/5). GDPa1 HIC vs Jain HIC are rank-consistent (Spearman 0.97) but different scale
  (Pearson 0.70). HIC distribution of the SSH2.0 subset ≈ full GDPa1 (no target-range shift).
  See `notebooks/01_ssh2_gdpa1_contamination.ipynb`. → standard cluster-CV can't give an
  SSH2.0-naive test fold; must choose an evaluation path (individual-level holdout / custom germline
  regrouping / reframed comparison) before building the HIC oracle.
  **[RESOLVED 2026-06-15 by GDPa3 — see next bullet.]**
- **2026-06-15 — GDPa3 is a clean external test → contamination dissolved.** GDPa3
  (`data/GDPa3_20260106_full.xlsx`, 80 abs, PROPHET-Ab assays incl. `hic_rt` on the GDPa1 scale) has
  **0 exact VH overlap with GDPa1 and 0 with SSH2.0/Jain137** (nearest neighbor ≤ 0.92 identity).
  It's the Ginkgo benchmark's own held-out set → train on all GDPa1 (+OAS weak-sup), test on GDPa3,
  dropping the per-fold/MRMD2.0 workaround entirely. See "Current experiment" below.

## Current experiment — HIC weak-supervision oracle (GDPa3 held-out)

**Question:** does SSH2.0 → OAS weak-supervision *pretraining* improve a sequence-based HIC oracle
vs gold-only, on a clean external test?

**Design (leakage-free):** train every model on **all of GDPa1 (246)**; evaluate once on
**GDPa3 (80)** — the Ginkgo benchmark's held-out set, verified **0 sequence overlap** with both
GDPa1 and SSH2.0's Jain131 training set. Since the test ∉ every training set there is no leakage;
the GDPa1↔Jain overlap is training-side only and **symmetric** across models (GoldOnly sees the Jain
antibodies' gold HIC; WeakSup sees that *plus* SSH2.0's noisy labels — so the only delta is OAS).
**Per-fold SSH2.0 / MRMD2.0 retraining is NOT needed** → use `ssh2cli` as-is to pseudo-label OAS.
GDPa1 isotype-stratified CV is used **only** for hyperparameter selection (never touch GDPa3 until
the final test). Metric: **Spearman ρ on `hic_rt`** (rank-based ⇒ scale-free) + top-10% recall,
bootstrap CIs (n=80). GDPa3 per-antibody HIC = average the tidy-format replicates (79 usable).

**Models** (all: train GDPa1 → test GDPa3; backbones = ESM-2 primary, AbLang2 ablation):

| # | Model | Backbone | Head | OAS pretrain | SSH2.0 role | Purpose |
|---|---|---|---|---|---|---|
| 1 | SSH2-direct | — (CKSAAGP) | 3×SVM (`ssh2cli`) | no | **is** predictor | floor |
| 2 | GoldOnly-Ridge·ESM2 | ESM-2 | Ridge | no | none | honest bar (benchmark-comparable) |
| 3 | GoldOnly-Ridge·AbLang2 | AbLang2 | Ridge | no | none | bar, antibody backbone |
| 4 | GoldOnly-MLP·ESM2 | ESM-2 | MLP | no | none | arch-matched control |
| 5 | GoldOnly-MLP·AbLang2 | AbLang2 | MLP | no | none | arch-matched control |
| 6 | SSH2-feature·ESM2 | ESM-2 + SSH2 prob | Ridge | no | input **feature** | isolates raw SSH2.0 signal (no OAS) |
| 7 | WeakSup-MLP·ESM2 | ESM-2 | MLP | ✅100k | OAS **labeler** | ★ the test (primary) |
| 8 | WeakSup-MLP·AbLang2 | AbLang2 | MLP | ✅100k | OAS **labeler** | the test (backbone ablation) |

**Key comparisons:** 7−4 (pretraining effect, same arch) · 7−2 (vs best practice) · 7−6 (OAS
amplification vs raw SSH2.0 signal) · 7 vs 8 / 2 vs 3 (backbone; AbLang2 already encodes OAS) ·
1 (floor) · data-efficiency = train GDPa1 subsets → test GDPa3.
Minimum viable headline = {1,2,4,7} on ESM-2; add {3,5,8} + 6 for the full story.

**SSH2-direct output & eval:** `ssh2cli` returns a continuous **risk probability** P(high HI risk).
We don't compare it on the HIC scale — we rank it against true `hic_rt` via Spearman (higher risk ⇒
higher HIC expected). Scale mismatch is irrelevant for a rank metric; probabilities are coarse
(voting+Platt) so expect a modest ceiling — fine for a floor.

**OAS:** ~100k paired human, deduped vs **GDPa1 ∪ Jain131 ∪ GDPa3** (test integrity + keeps SSH2.0
generalizing, not regurgitating).

**Embeddings (locked):** backbones = **ESM2-650M** (`t33_650M`, 1280-d — primary/downstream) +
**ESM2-8M** (`t6_8M`, 320-d — exact benchmark anchor) + **AbLang2-paired** (antibody ablation).
Benchmark-anchor recipe (matches `models/esm2_ridge`): VH & VL embedded **separately**, mask-aware
mean-pool of last hidden state, concat `[vh, vl]` (8M→640-d), Ridge α=1.0, no standardization.
Scheme for all ESM-2 = **separate-then-concat** (principled for single-chain LMs); AbLang2 uses its
**native paired** representation. **Store pooled only — no per-residue raw** (storage): per chain
save **mean-pool + CLS** (same dim; CLS is *not* lower-dim, just one token's vector; ESM-2 mean
usually ≥ CLS since ESM-2 has no CLS training objective — store both to compare). Embed GDPa1,
GDPa3, OAS with the identical fn per backbone. **ESM-C = future-work/ablation** (use ESM-2 for
benchmark parity; preempt the reviewer question in the writeup).

**`ssh2cli` validated (2026-06-15):** on Jain-131 — sens 1.00, spec 0.755, acc 0.824 (≈ paper LOOCV
1.00/0.777/0.840). Added monotonic **`P_positive`** = mean P(positive) over the 3 SVMs (130 distinct
values; Spearman 0.47 vs HIC RT). Use `P_positive` for SSH2-direct ranking + as the OAS soft
pseudo-label (NOT the `Probability`/confidence-in-call field). No retraining/MRMD2.0 — `ssh2cli`
(original weights) used as-is.

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
- **`ssh2cli/`** = clean pure-Python SSH2.0 reimplementation (CKSAAGP + 3 RBF SVMs, majority vote);
  validated bit-for-bit vs reference LIBSVM. This is the tool for generating SSH2.0 values/probs at
  scale. Run: `python3 ssh2cli/ssh2.py --hc h.fasta --lc l.fasta -o out.tsv`. The 68 MB original
  Windows release (`data/SSH2.0/SSH2.0_from_tar_file/`, `*.exe`) is gitignored.
- **Data hygiene caution:** CSVs live in Drive and can silently diverge from git (a stray
  `foralumab` once appeared in GDPa1 `AC-SINS_pH6.0` for blosozumab via a Drive edit; restored from
  git on 2026-06-15). Treat the git version as canonical; fix data at the source.

## Working preferences

- **Checkpointed, resumable long-running jobs.** For long generation/embedding steps, build scripts
  that checkpoint incrementally (per-shard / per-batch) and resume where they left off if
  interrupted — never restart from scratch. Surface progress clearly. Such steps should be
  standalone scripts a collaborator can run concurrently.

## Results so far (GDPa3 held-out, Spearman ρ on hic_rt; no-std Ridge α=1.0 unless noted)

| Model | ρ (95% CI) | note |
|---|---|---|
| SSH2-direct (floor) | 0.175 (−0.06..0.38) | raw weak labeler |
| GoldOnly esm2-8m mean | **0.403** (0.21..0.59) | **= benchmark `esm2_ridge` (0.403) → pipeline validated** |
| GoldOnly esm2-650m mean | 0.425 (0.23..0.59) | |
| GoldOnly **esm2-650m CLS** | **0.508** (0.32..0.66) | **best bar; beats all benchmark embeddings, rivals physics MOE 0.495** |
| GoldOnly ablang2-paired mean | 0.214 (−0.01..0.44) | "already-encodes-OAS" backbone is weaker |
| SSH2-feature esm2-650m (std) | 0.250 vs GoldOnly-std 0.264 | raw SSH2 feature adds no lift |

Findings: 650M > 8M; **CLS > mean (650M)**; AbLang2 < ESM-2; standardization hurts (8m 0.403→0.106).

**WeakSup result (data-efficiency sweep, 5 backbone×pooling combos × 5 sizes × 5 seeds → 511 runs,
`results/hic_results.csv`, fig `results/figures/lift.png`):**
- **OAS weak-supervision pretraining IS worth it in the low-data regime.** Mean lift
  (WeakSup − GoldOnly-MLP) over backbones: **+0.115 at n=25**, +0.05 at n=50, ~0 by n=100, ~0/slightly
  negative at full data (242). The realistic scarce-label regime is exactly where it helps.
- **MSE pretraining > BCE** (empirically): `mse_keepall`/`mse_reinit` carry the lift; `bce_reinit`
  (my proposed "principled primary") hugs zero. → treat `P_positive` as a regression target + warm-start
  the head. (Updates the earlier BCE recommendation.)
- **Best absolute oracle stays GoldOnly-Ridge esm2-650m CLS ρ≈0.51**; WeakSup doesn't beat the
  full-data bar — its value is *data efficiency*, not peak accuracy.
- MLP training: fixed-epoch regularized (epochs=60, wd=3e-3, dropout=0.3, LayerNorm input, no
  early-stop) — early-stopping on the tiny val split caused seed collapses (std 0.157→0.012).
- Presentation: `notebooks/02_hic_oracle_experiment.ipynb` (thin; renders table + figures).

## Track ownership (avoid Drive/file collisions)

- **Track A (this line of work):** `src/abdev/**`, `scripts/pseudolabel_oas.py`,
  `notebooks/02_hic_oracle_experiment.ipynb`, `ssh2cli/**`, tests.
- **Track B (collaborator):** `scripts/build_oas_subset.py`, `scripts/embed_sequences.py`,
  `data/oas/**`, `data/embeddings/**`.
- Both push to `main`; stage files explicitly (never `git add -A`), `git fetch` before pushing.

## Untested levers (for next sessions, priority order)

1. **LoRA fine-tune ESM-2 encoder on OAS SSH2.0 labels → GDPa1 HIC** (matched control GoldOnly-LoRA).
   The representation-learning version of weak-sup; most likely to move the full-data number. GPU/Colab,
   `peft`, target attn query/value, r≈8–16. Start esm2-8m.
2. **Multi-source weak labels** (SSH2.0 + SAP/charge/TAP/CamSol) multi-task pretraining — richer than
   one hydrophobicity labeler.
3. **Scale/tune pretraining** (corpus >100k, more epochs); **α/CV hyperparameter tuning** for Ridge/MLP.
4. **AbLang2 rescoding (per-residue) + pool** instead of seqcoding; **ESM-2 + AbLang2 ensemble**.
5. Then: fold the oracle into the DFM guidance side (posterior tempering), or write up the oracle
   data-efficiency result standalone.

## Changelog

### 2026-06-30 — Session 6 (WeakSup experiment run + verdict)
- Ran the full data-efficiency sweep (511 runs) → `results/hic_results.csv` + figures + executed
  `notebooks/02`. Verdict: OAS weak-sup helps only low-data (+0.115 @ n=25); MSE>BCE; best oracle =
  Ridge esm2-650m CLS (0.51). See Current status / Results.
- Fixed MLP instability (early-stop collapse → fixed-epoch+reg, std 0.157→0.012). Diagnostics confirm
  pretraining works (learns P_positive ρ=0.80; pretrain-only→HIC 0.21–0.30), so the null-at-full-data
  is genuine (for frozen-embedding shallow pretraining) — hence the LoRA next step.
- Built `weaksup.py`, `oracle.py`, `viz.py` (lift/data-efficiency/summary figures), `run_hic_experiment.py`.

### 2026-06-28 — Session 5 (Track B: OAS subset + embeddings delivered)
**Deliverable 1 — OAS paired-human subset: DONE → unblocks Track A's `scripts/pseudolabel_oas.py`.**
- `data/oas/oas_paired_human_100k.parquet` (gitignored, Drive-shared) + `*.manifest.json`.
  Columns: `id, heavy, light, study, v_gene_h, j_gene_h, v_gene_l, j_gene_l`. Exactly 100k, seed=42.
  **Track A: pseudo-label THIS exact file (reproducible, seed locked).**
- Built by `scripts/build_oas_subset.py` (CPU, resumable). Pipeline: download all 610 OPIG paired
  CSVs → keep metadata `Species=="human"` (580/610 files) → per-row productive/in-frame/no-stop,
  IGH + IGK/IGL, valid 20-AA len 90–200 → clonotype dedup `(v_gene_h,j_gene_h,cdr3_h,cdr3_l)` →
  held-out drop (exact + ≥95% heavy identity, rapidfuzz) vs GDPa1 ∪ Jain ∪ GDPa3 ∪
  GDPa1+Jain2024-calibrated → random sample 100k.
- Counts: 3,003,127 raw human paired → 2,392,734 unique clonotypes → drop 37,262 exact + 56,162
  near → 2,299,310 → sample 100k. Verified: 0 dup (heavy,light), 0 non-AA, **0 exact heavy
  overlap with held-out**. git SHA in manifest. (Gotcha: OPIG paired CSVs use single-letter loci
  H/K/L, not IGH/IGK/IGL.) Raw gz cache is OUTSIDE Drive at `~/oas_raw` (3.4 GB; `ABDEV_OAS_RAW` to move).

**Deliverable 2 — embeddings: GDPa1+GDPa3 DONE for all 3 backbones; OAS pending.**
- `scripts/embed_sequences.py` + `notebooks/colab_embed_sequences.ipynb` (Colab GPU, per-shard
  checkpoint/resume). Output contract (STABLE — Track A reads this):
  `data/embeddings/{backbone}/{dataset}/shard_*.parquet`, cols `[id, chain, pool, emb(float16)]`.
  - ESM-2 (esm2-8m 320-d, esm2-650m 1280-d): VH/VL separately, last hidden state → `mean`
    (mask-aware) + `cls`; `chain∈{H,L}`, `pool∈{mean,cls}`. Concat `[vh,vl]` at use-time.
  - AbLang2 (480-d): native paired `seqcoding` (`chain='P'`) + per-chain mean (`H`/`L`),
    `pool='mean'`. **Use 'P' as the primary AbLang2 antibody feature** (paired model; per-chain
    is ESM-parity ablation).
- On Drive + verified: GDPa1 (246) + GDPa3 (80) for all 3 backbones (dims correct, no NaN).
- TODO: notebook cell 7 → OAS embeddings (esm2-8m + ablang2 first, then esm2-650m long run).
- Added `rapidfuzz` to the `oas` extra; pyproject `pyarrow` pinned `>=24.0.0`.

### 2026-06-28 — Session 5 (oracle harness start + first result + .git recovery)
- Built `eval/metrics.py` (Spearman, top-k recall, bootstrap CI), `scorers/ssh2.py` (SSH2-direct +
  `ssh2_p_positive` helper), `scripts/pseudolabel_oas.py` (checkpointed OAS labeler, ready for the
  subset). **First result — SSH2-direct floor on GDPa3: Spearman 0.175 (95% CI −0.06..0.38),
  top-10% recall 0.25** (raw weak signal barely ranks HIC on the clean external set; the intended floor).
- **Drive corrupted `.git` (3rd incident):** deleted `.git/index`, left a stale `index.lock`.
  Recovered via `rm .git/index.lock && git reset HEAD` (HEAD/origin intact). **Action needed:**
  collaborator should work from their OWN local clone (non-Drive) + push to GitHub — sharing one
  Drive-synced working tree + `.git` between two writers is corrupting git itself.
- GDPa4 (bispecifics): out of scope for this experiment (multi-chain format our VH/VL embedding +
  OAS-trained generator don't handle; `hihplc_normretentiontime_median` is a different normalization).
  Future extension, not now.

### 2026-06-15 — Session 4 (GDPa3 held-out → simplified, leakage-free design)

### 2026-06-15 — Session 4 (GDPa3 held-out → simplified, leakage-free design)
- Found GDPa3 (Ginkgo benchmark held-out, 80 abs, `hic_rt` on GDPa1 scale). Verified **0 overlap**
  with GDPa1 and with SSH2.0/Jain137 → clean external test.
- **Switched the HIC experiment to GDPa3-held-out: train on all GDPa1 (+OAS weak-sup), test on
  GDPa3.** Dropped per-fold SSH2.0 / MRMD2.0 (no longer needed); use `ssh2cli` as-is to label OAS.
- Locked the 8-model set (see "Current experiment"): SSH2-direct, GoldOnly-{Ridge,MLP}×{ESM2,AbLang2},
  SSH2-feature, WeakSup-MLP×{ESM2,AbLang2}. Added SSH2-feature to isolate OAS amplification from raw
  SSH2.0 signal. OAS dedup = GDPa1 ∪ Jain131 ∪ GDPa3.
- Added `openpyxl` (read GDPa3 xlsx). ESM-2 default `t33_650M` pending a quick check vs the
  benchmark's featurizer for exact comparability.
- **Next:** `ssh2cli` validation gate → embeddings (checkpointed, incl. GDPa3) → 4-model headline
  {1,2,4,7} on ESM-2 → full set.

### 2026-06-15 — Session 3 (oracle experiment design + SSH2.0 recipe recovery)
- Locked the HIC weak-supervision experiment as **Option C**: standard 5-fold CV on **all 246**
  (isotype-stratified folds), with SSH2.0 **retrained blind to each test fold** (per-fold MRMD2.0
  reselection). Models: **SSH2-direct, GoldOnly-Ridge, GoldOnly-MLP, WeakSup-MLP** (ESM-2 frozen
  embeddings; Ridge = honest bar, MLP = matched control). Headline = WeakSup − GoldOnly.
- **Recovered SSH2.0's full training recipe** from the paper and verified against our data:
  label = (SMAC>12.8 OR SGAC-SINS<370 OR HIC>11.7) → 37 pos / 94 neg on 131 (drop the 6
  "Disclaimers" rows); feature subsets 29/31/35; RBF, γ in `.model` files, C via grid-search;
  train with `sklearn.svm.SVC` (= libsvm). **Validation gate:** reproduce LOOCV acc ~83.97% and
  match `ssh2cli` before any per-fold use.
- Ginkgo abdev-benchmark: same GDPa1 + folds; Spearman + top-10% recall; **Ridge** is the validated
  head (esm2_ridge HIC ≈ 0.40 heldout); physics (MOE/Aggrescan3D) lead absolute (~0.5–0.66).
- **Next:** within-cluster HIC sanity check → SSH2.0 reproduction + validation gate → embeddings
  (checkpointed script) → 4-model CV.

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
- Fixed a Drive-introduced corruption in GDPa1 (`foralumab` in blosozumab's `AC-SINS_pH6.0`) by
  restoring from git; verified column is numeric again. Audit results unaffected (HIC-only).
- Committed `ssh2cli/` (clean SSH2.0 reimplementation, smoke-tested) + `jain_et_al_ssh2_training.csv`
  (commit 239d250); gitignored the 68 MB Windows release + build artifacts.
- **Next:** choose the oracle-evaluation path given no clean fold (individual-level holdout vs custom
  germline regrouping vs reframed comparison), then build the HIC oracle (L0 vs L2 vs SSH2.0-direct,
  using `ssh2cli` to pseudo-label OAS).
