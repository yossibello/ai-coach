"""
Temperature scaling for calibrated workout-type confidences.

Implements Guo, Pleiss, Sun & Weinberger (2017), *On Calibration of Modern
Neural Networks* (ICML). After training, we fit a single scalar parameter T
on the held-out validation set so that the softmax temperature minimizes
negative log likelihood. T > 1 softens overconfident predictions; T < 1
sharpens underconfident ones. The fitted T is saved into the checkpoint
under config["temperature"] and applied at inference.

Usage:
    python -m ml.training.calibrate \\
        --data ./ml/data/synthetic_v2.parquet \\
        --checkpoint ./backend/models/cycling_coach_v2.pt
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.model import CyclingTransformer
from ml.training.dataset import CyclingDataset, athlete_split, collate_fn


def _load_model_and_cfg(path: str) -> tuple[CyclingTransformer, dict, dict]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        cfg = dict(state.get("config", {}) or {})
        sd = state["state_dict"]
    else:
        cfg, sd = {}, state
    proj_w = sd.get("input_proj.0.weight")
    if proj_w is not None:
        cfg["input_dim"] = int(proj_w.shape[1])
    cfg.pop("horizon_aware", None)
    model = CyclingTransformer(**cfg)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg, state


def _collect_logits_and_labels(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            di = batch["day_idx"].to(device)
            pm = batch["padding_mask"].to(device)
            hq = batch.get("horizon_query")
            if hq is not None:
                hq = hq.to(device)
            out = model(x, di, pm, horizon_query=hq)
            all_logits.append(out["workout_logits"].cpu())
            all_labels.append(batch["target_wt"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def _expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """ECE: average |confidence - accuracy| across confidence bins."""
    confs, preds = probs.max(dim=1)
    correct = preds.eq(labels).float()
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() > 0:
            acc = correct[mask].mean().item()
            conf = confs[mask].mean().item()
            ece += (mask.sum().item() / n) * abs(acc - conf)
    return ece


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single scalar T by minimizing NLL with LBFGS (Guo et al. 2017)."""
    T = nn.Parameter(torch.ones(1) * 1.0)
    nll = nn.CrossEntropyLoss()
    optim = torch.optim.LBFGS([T], lr=0.05, max_iter=100)

    def closure():
        optim.zero_grad()
        loss = nll(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss

    optim.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def main():
    parser = argparse.ArgumentParser(description="Temperature-scale a trained CyclingTransformer")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calibrating on {device}")

    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    _, val_ids = athlete_split(df, val_frac=args.val_frac, seed=args.seed)
    val_ds = CyclingDataset(df, seq_len=args.seq_len, athlete_ids=val_ids)
    print(f"Val sequences: {len(val_ds):,}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    model, cfg, state = _load_model_and_cfg(args.checkpoint)
    model.to(device)

    logits, labels = _collect_logits_and_labels(model, val_loader, device)

    pre_probs = torch.softmax(logits, dim=1)
    pre_ece = _expected_calibration_error(pre_probs, labels)
    pre_acc = (logits.argmax(dim=1) == labels).float().mean().item()
    print(f"Before calibration: acc={pre_acc*100:.2f}%  ECE={pre_ece:.4f}")

    T = fit_temperature(logits, labels)
    post_probs = torch.softmax(logits / T, dim=1)
    post_ece = _expected_calibration_error(post_probs, labels)
    print(f"Fitted temperature T = {T:.4f}")
    print(f"After  calibration: acc={pre_acc*100:.2f}%  ECE={post_ece:.4f}")

    # Persist T inside the checkpoint config so inference can pick it up.
    if isinstance(state, dict) and "config" in state:
        state["config"]["temperature"] = T
    else:
        state = {"state_dict": state, "config": {"temperature": T}}
    torch.save(state, args.checkpoint)
    print(f"Saved calibrated checkpoint with T={T:.4f} → {args.checkpoint}")


if __name__ == "__main__":
    main()
