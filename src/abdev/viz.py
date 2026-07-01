"""Publication figures for the HIC-oracle experiment (reads results/hic_results.csv).

  * data_efficiency_figure — Spearman ρ vs GDPa1 training size for GoldOnly-MLP vs WeakSup, with
    ±1σ (over seeds) bands and reference lines (GoldOnly-Ridge bar, SSH2-direct floor). This is the
    "is pretraining worth it, and at what data scale" figure.
  * summary_bar — full-data ρ across models, grouped by backbone/representation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- clean publication style -----------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "font.family": "sans-serif",
})
C = {"GoldOnly-MLP": "#4C72B0", "WeakSup-bce_reinit": "#C44E52",
     "WeakSup-mse_reinit": "#DD8452", "WeakSup-mse_keepall": "#8172B3"}


def _agg(df):
    """Mean ρ and σ over seeds per (backbone, representation, model, n_train)."""
    return (df.groupby(["backbone", "representation", "model", "n_train"])
              .agg(rho=("rho", "mean"), sd=("rho", "std")).reset_index().fillna({"sd": 0.0}))


def data_efficiency_figure(csv="results/hic_results.csv", backbone="esm2-650m", rep="cls",
                           out="results/figures/data_efficiency.png",
                           curves=("GoldOnly-MLP", "WeakSup-bce_reinit",
                                   "WeakSup-mse_reinit", "WeakSup-mse_keepall")):
    df = pd.read_csv(csv)
    a = _agg(df)
    sub = a[(a.backbone == backbone) & (a.representation == rep)]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for model in curves:
        m = sub[sub.model == model].sort_values("n_train")
        if m.empty:
            continue
        ax.plot(m.n_train, m.rho, "-o", ms=5, lw=2, color=C.get(model, "gray"),
                label=model.replace("WeakSup-", "WeakSup "))
        ax.fill_between(m.n_train, m.rho - m.sd, m.rho + m.sd, color=C.get(model, "gray"), alpha=0.15)
    # reference lines: GoldOnly-Ridge (full) and SSH2-direct floor
    ridge = df[(df.backbone == backbone) & (df.representation == rep) & (df.model == "GoldOnly-Ridge")]
    if len(ridge):
        ax.axhline(ridge.rho.iloc[0], ls="--", lw=1.3, color="#55A868",
                   label=f"GoldOnly-Ridge (full) = {ridge.rho.iloc[0]:.2f}")
    floor = df[df.model == "SSH2-direct"]
    if len(floor):
        ax.axhline(floor.rho.iloc[0], ls=":", lw=1.3, color="gray",
                   label=f"SSH2-direct floor = {floor.rho.iloc[0]:.2f}")
    ax.set(xlabel="GDPa1 training antibodies (n)", ylabel="Spearman ρ on GDPa3 (held-out)",
           title=f"Data efficiency — {backbone} / {rep}")
    ax.legend(loc="lower right", fontsize=9)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(str(out).replace(".png", ".pdf"))
    return fig


def lift_figure(csv="results/hic_results.csv",
                variants=("mse_keepall", "mse_reinit", "bce_reinit"),
                out="results/figures/lift.png"):
    """The headline: weak-supervision lift (WeakSup − GoldOnly-MLP) vs training size,
    mean ± σ across the backbone×representation combos. Positive at low n = worth it when scarce."""
    df = pd.read_csv(csv)
    recs = []
    for (bb, rep), g in df.groupby(["backbone", "representation"]):
        p = g.groupby(["model", "n_train"]).rho.mean().unstack("model")
        if "GoldOnly-MLP" not in p.columns:
            continue
        for n in p.index:
            for v in variants:
                mv = f"WeakSup-{v}"
                if mv in p.columns:
                    recs.append({"n": int(n), "variant": v, "lift": p.loc[n, mv] - p.loc[n, "GoldOnly-MLP"]})
    L = pd.DataFrame(recs)
    agg = {v: L[L.variant == v].groupby("n").lift.agg(["mean", "std"]).reset_index() for v in variants}
    ymax = max(float((a["mean"] + a["std"]).max()) for a in agg.values()) + 0.02
    ymin = min(float((a["mean"] - a["std"]).min()) for a in agg.values()) - 0.02
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhspan(0, ymax, color="#55A868", alpha=0.06)
    for v in variants:
        s = agg[v]
        ax.plot(s.n, s["mean"], "-o", ms=6, lw=2.2, color=C.get(f"WeakSup-{v}", "gray"),
                label=f"WeakSup {v}")
        ax.fill_between(s.n, s["mean"] - s["std"], s["mean"] + s["std"],
                        color=C.get(f"WeakSup-{v}", "gray"), alpha=0.13)
    ax.axhline(0, color="black", lw=1.1)
    ax.set_ylim(ymin, ymax)
    ax.set(xlabel="GDPa1 training antibodies (n)",
           ylabel="Δ Spearman ρ  (WeakSup − GoldOnly-MLP)",
           title="Weak-supervision lift vs training-set size\n(mean ± σ across 5 backbone×pooling combos)")
    ax.annotate("pretraining helps\nwhen labels are scarce", xy=(25, agg["mse_keepall"]["mean"].iloc[0]),
                xytext=(85, ymax * 0.7), fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#777"))
    ax.legend(loc="upper right", fontsize=9)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(str(out).replace(".png", ".pdf"))
    return fig


def summary_bar(csv="results/hic_results.csv", out="results/figures/summary_bar.png"):
    """Full-data ρ per model, grouped by backbone/representation."""
    df = pd.read_csv(csv)
    full = df[df.n_train == df.n_train.max()]
    piv = (full.groupby(["backbone", "representation", "model"]).rho.mean()
               .reset_index())
    piv["bb"] = piv.backbone + "/" + piv.representation
    order = ["GoldOnly-Ridge", "GoldOnly-MLP", "WeakSup-bce_reinit",
             "WeakSup-mse_reinit", "WeakSup-mse_keepall", "SSH2-feature"]
    bbs = [b for b in piv.bb.unique() if b != "-/-"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.8 / len(order)
    x = np.arange(len(bbs))
    for j, model in enumerate(order):
        vals = [piv[(piv.bb == b) & (piv.model == model)].rho.mean() for b in bbs]
        ax.bar(x + j * w, vals, w, label=model, color=C.get(model, None))
    floor = df[df.model == "SSH2-direct"].rho
    if len(floor):
        ax.axhline(floor.iloc[0], ls=":", color="gray", lw=1.2, label="SSH2-direct floor")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(bbs, rotation=15, ha="right")
    ax.set(ylabel="Spearman ρ on GDPa3 (full data)", title="HIC oracle — full-data performance")
    ax.legend(fontsize=8, ncol=2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(str(out).replace(".png", ".pdf"))
    return fig
