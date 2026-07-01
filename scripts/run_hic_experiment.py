"""Run the full HIC-oracle experiment → results/hic_results.csv (resumable).

For each (backbone, representation): GoldOnly-Ridge + SSH2-feature (full data), and a
data-efficiency sweep over GDPa1 subset sizes × seeds for GoldOnly-MLP and the three WeakSup
variants (bce_reinit primary, mse_reinit, mse_keepall). SSH2-direct is added once (backbone-free).

One row per run: backbone, representation, model, n_train, seed, rho, ci_lo, ci_hi, top10.
Rows are appended as they complete; rerunning skips finished (backbone,rep,model,n_train,seed) keys.
Runs on CPU (small MLP on precomputed embeddings). No standardization (per design).

Run:  uv run python scripts/run_hic_experiment.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from abdev import weaksup as ws
from abdev.data.embeddings import align_xy, load_features
from abdev.data.gdpa1 import load_gdpa1
from abdev.data.gdpa3 import load_gdpa3
from abdev.eval.metrics import bootstrap_ci, spearman, top_k_recall
from abdev.oracle import ridge_predict, with_ssh2_feature
from abdev.scorers.ssh2 import ssh2_p_positive

OUT = Path("results/hic_results.csv")
COMBOS = [("esm2-8m", "mean"), ("esm2-8m", "cls"),
          ("esm2-650m", "mean"), ("esm2-650m", "cls"),
          ("ablang2", "mean")]           # ablang2 mean = native paired 'P' (default chains)
SIZES = [25, 50, 100, 150, 242]
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["bce_reinit", "mse_reinit", "mse_keepall"]
COLS = ["backbone", "representation", "model", "n_train", "seed", "rho", "ci_lo", "ci_hi", "top10"]


def _done_keys() -> set:
    if not OUT.exists():
        return set()
    d = pd.read_csv(OUT)
    return {(r.backbone, r.representation, r.model, int(r.n_train), int(r.seed))
            for r in d.itertuples(index=False)}


def _append(row: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = not OUT.exists()
    pd.DataFrame([row])[COLS].to_csv(OUT, mode="a", header=header, index=False)


def _record(done, backbone, rep, model, n_train, seed, pred, yte):
    key = (backbone, rep, model, n_train, seed)
    if key in done:
        return
    rho, lo, hi = bootstrap_ci(pred, yte, seed=0)
    _append({"backbone": backbone, "representation": rep, "model": model, "n_train": n_train,
             "seed": seed, "rho": round(rho, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
             "top10": round(top_k_recall(pred, yte), 4)})
    done.add(key)
    print(f"  {backbone:9s} {rep:4s} {model:16s} n={n_train:3d} seed={seed} rho={rho:.3f}", flush=True)


def _oas_pseudolabels() -> dict:
    pl = pd.concat([pd.read_parquet(s)
                    for s in glob.glob("data/pseudolabels/ssh2_oas/shard_*.parquet")])
    return dict(zip(pl.id.astype(str), pl.ssh2_p_positive))


def main() -> None:
    done = _done_keys()
    g1, g3 = load_gdpa1(), load_gdpa3()
    seqs1 = {r.id: (r.vh, r.vl) for r in g1.itertuples(index=False)}
    seqs3 = {r.id: (r.vh, r.vl) for r in g3.itertuples(index=False)}
    plmap = _oas_pseudolabels()

    # SSH2-direct floor (backbone-free) — once
    if ("-", "-", "SSH2-direct", 0, 0) not in done:
        pred = ssh2_p_positive([seqs3[i][0] for i in g3.id], [seqs3[i][1] for i in g3.id], list(g3.id))
        _record(done, "-", "-", "SSH2-direct", 0, 0, pred, g3.hic.values)

    for backbone, rep in COMBOS:
        print(f"\n=== {backbone} / {rep} ===", flush=True)
        it, Xt = load_features(backbone, "gdpa1", rep)
        Xtr_full, ytr_full = align_xy(it, Xt, g1)
        tr_ids = [str(r.id) for r in g1.itertuples(index=False) if str(r.id) in set(it)]
        ie, Xe = load_features(backbone, "gdpa3", rep)
        Xte, yte = align_xy(ie, Xe, g3)
        te_ids = [str(r.id) for r in g3.itertuples(index=False) if str(r.id) in set(ie)]
        d = Xtr_full.shape[1]

        # GoldOnly-Ridge + SSH2-feature (full data, benchmark recipe: no-std)
        _record(done, backbone, rep, "GoldOnly-Ridge", len(ytr_full), 0,
                ridge_predict(Xtr_full, ytr_full, Xte, alpha=1.0, standardize=False), yte)
        Xtr_s = with_ssh2_feature(Xtr_full, tr_ids, seqs1)
        Xte_s = with_ssh2_feature(Xte, te_ids, seqs3)
        _record(done, backbone, rep, "SSH2-feature", len(ytr_full), 0,
                ridge_predict(Xtr_s, ytr_full, Xte_s, alpha=1.0, standardize=True), yte)

        # OAS features aligned to pseudo-labels + cached pretrained bodies
        oid, Xo = load_features(backbone, "oas", rep)
        m = np.array([i in plmap for i in oid])
        Xoas = Xo[m]
        yoas = np.array([plmap[i] for i, k in zip(oid, m) if k], float)
        pre = {"bce": ws.pretrain(d, Xoas, yoas, loss="bce", seed=0),
               "mse": ws.pretrain(d, Xoas, yoas, loss="mse", seed=0)}
        print(f"  pretrained (OAS n={len(yoas)}, dim={d})", flush=True)

        for n_train in SIZES:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                idx = (np.arange(len(ytr_full)) if n_train >= len(ytr_full)
                       else rng.choice(len(ytr_full), n_train, replace=False))
                Xs, ys = Xtr_full[idx], ytr_full[idx]
                _record(done, backbone, rep, "GoldOnly-MLP", n_train, seed,
                        ws.predict(ws.goldonly_mlp(d, Xs, ys, seed=seed), Xte), yte)
                for v in VARIANTS:
                    p = pre["bce" if v.startswith("bce") else "mse"]
                    model = ws.weaksup_finetune(p, Xs, ys, reinit_head=v.endswith("reinit"), seed=seed)
                    _record(done, backbone, rep, f"WeakSup-{v}", n_train, seed,
                            ws.predict(model, Xte), yte)
    print("\nDONE →", OUT, flush=True)


if __name__ == "__main__":
    main()
