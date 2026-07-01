"""WeakSup HIC oracle + arch-matched GoldOnly-MLP (train on GDPa1 → predict GDPa3).

Frozen embeddings in; a small MLP (body + head) is the only trainable part. The value of
pretraining lives in the body (learned sequence→hydrophobicity representation); the head is
task-specific.

WeakSup variants (MEMORY.md "Current experiment"):
  * bce_reinit (primary): pretrain body+head on OAS SSH2.0 `P_positive` with BCE (it's a
    probability → sigmoid head), then REINIT a fresh linear head and fine-tune on HIC (MSE).
  * mse_reinit  (loss ablation): pretrain with MSE (linear head), reinit head, fine-tune.
  * mse_keepall (warm-start ablation): pretrain with MSE, KEEP the head, fine-tune.
GoldOnly-MLP = the same architecture trained on GDPa1 only (no pretrain) — the control.

Small MLP on precomputed embeddings → CPU/MPS is sufficient (no GPU needed).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 64), dropout: float = 0.3):
        super().__init__()
        # Input LayerNorm: batch-independent per-sample normalization → stable training on raw
        # (unstandardized) embeddings with tiny GDPa1 (no external scaler / OAS-stats question).
        layers, d = [nn.LayerNorm(in_dim)], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x, sigmoid: bool = False):
        o = self.head(self.body(x)).squeeze(-1)
        return torch.sigmoid(o) if sigmoid else o


def _fit(model, X, y, *, loss: str, epochs: int, lr: float, wd: float, batch: int,
         val_frac: float = 0.0, patience: int = 20, seed: int = 0, device: str = "cpu"):
    """Train `model` in place. loss='bce' (sigmoid output, soft targets) or 'mse'."""
    set_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    n = len(yt)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(val_frac * n) if val_frac > 0 else 0
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr = Xt[tr_idx].to(device), yt[tr_idx].to(device)
    Xva, yva = (Xt[val_idx].to(device), yt[val_idx].to(device)) if n_val else (None, None)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCELoss() if loss == "bce" else nn.MSELoss()
    sig = loss == "bce"
    best_state, best_val, bad = copy.deepcopy(model.state_dict()), float("inf"), 0

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ytr))
        for s in range(0, len(ytr), batch):
            idx = order[s:s + batch]
            opt.zero_grad()
            out = model(Xtr[idx], sigmoid=sig)
            lossf(out, ytr[idx]).backward()
            opt.step()
        if n_val:
            model.eval()
            with torch.no_grad():
                v = lossf(model(Xva, sigmoid=sig), yva).item()
            if v < best_val - 1e-6:
                best_val, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if n_val:
        model.load_state_dict(best_state)
    return model


def predict(model, X, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32).to(device)).cpu().numpy()


# --- high-level fit helpers -----------------------------------------------------------------
def pretrain(in_dim, X_oas, y_oas, *, loss: str, hidden=(256, 64), dropout=0.3,
             epochs=6, lr=1e-3, wd=0.0, batch=512, seed=0, device="cpu") -> MLP:
    set_seed(seed)
    model = MLP(in_dim, hidden, dropout)
    _fit(model, X_oas, y_oas, loss=loss, epochs=epochs, lr=lr, wd=wd, batch=batch,
         seed=seed, device=device)
    return model


def goldonly_mlp(in_dim, X_tr, y_tr, *, hidden=(256, 64), dropout=0.3, epochs=60, lr=1e-3,
                 wd=3e-3, batch=32, val_frac=0.0, seed=0, device="cpu") -> MLP:
    set_seed(seed)
    model = MLP(in_dim, hidden, dropout)
    _fit(model, X_tr, y_tr, loss="mse", epochs=epochs, lr=lr, wd=wd, batch=batch,
         val_frac=val_frac, seed=seed, device=device)
    return model


def weaksup_finetune(pretrained: MLP, X_tr, y_tr, *, reinit_head: bool, epochs=60, lr=1e-3,
                     wd=3e-3, batch=32, val_frac=0.0, seed=0, device="cpu") -> MLP:
    """Fine-tune a (deep-copied) pretrained MLP on HIC. reinit_head swaps in a fresh linear head."""
    model = copy.deepcopy(pretrained)
    if reinit_head:
        model.head = nn.Linear(model.head.in_features, 1)
    _fit(model, X_tr, y_tr, loss="mse", epochs=epochs, lr=lr, wd=wd, batch=batch,
         val_frac=val_frac, seed=seed, device=device)
    return model
