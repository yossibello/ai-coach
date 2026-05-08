"""
Training script for the CyclingTransformer.

Pipeline:
  1. Load raw parquet (synthetic or real DB export, same schema).
  2. Split BY athlete_id (no leakage between train/val).
  3. Build lazy CyclingDataset for each split.
  4. Train with multi-task loss:
       • workout-type classification (CE, label smoothing)
       • next-ride intensity factor (MSE in IF units)
       • next-ride duration in hours (MSE)
       • 4-week FTP delta in watts (Huber)

Usage:
  # Full pre-training run on synthetic data:
  python -m ml.training.train \\
      --data ./ml/data/synthetic.parquet \\
      --output ./backend/models/cycling_coach.pt \\
      --epochs 100

  # Smoke test (~15 min on Ryzen 5600):
  python -m ml.training.train \\
      --data ./ml/data/synthetic_small.parquet \\
      --output ./backend/models/cycling_coach_fast.pt \\
      --fast --epochs 20

  # Fine-tune on real athlete data:
  python -m ml.training.train \\
      --data ./ml/data/activities.parquet \\
      --output ./backend/models/cycling_coach.pt \\
      --checkpoint ./backend/models/cycling_coach.pt \\
      --epochs 30 --lr 5e-5
"""
from __future__ import annotations

import argparse
import math
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app.ml.model import CyclingTransformer
from ml.training.dataset import CyclingDataset, athlete_split, collate_fn


def train(args):
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    print(f"PyTorch threads: {n_threads}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading data from {args.data}…")
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    print(f"Total rows: {len(df):,}  athletes: {df['athlete_id'].nunique():,}")

    # ── Split by athlete (no leakage) ─────────────────────────────────────
    train_ids, val_ids = athlete_split(df, val_frac=args.val_frac, seed=args.seed)
    print(f"Train athletes: {len(train_ids):,}  Val athletes: {len(val_ids):,}")

    train_ds = CyclingDataset(df, seq_len=args.seq_len, athlete_ids=train_ids)
    val_ds   = CyclingDataset(df, seq_len=args.seq_len, athlete_ids=val_ids)
    print(f"Train sequences: {len(train_ds):,}  Val sequences: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(4, n_threads),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(2, n_threads),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if args.fast:
        d_model, nhead, num_layers, d_ff = 64, 4, 4, 256
        print("FAST MODE: d=64, 4 layers (testing only — not production)")
    else:
        d_model, nhead   = args.d_model, args.nhead
        num_layers, d_ff = args.num_layers, args.d_ff

    model = CyclingTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_ff,
        dropout=args.dropout,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed from checkpoint: {args.checkpoint}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer / Scheduler / Losses ────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ce_loss    = nn.CrossEntropyLoss(label_smoothing=0.1)
    mse_loss   = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=10.0)

    best_val_loss = math.inf
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            x   = batch["x"].to(device)
            di  = batch["day_idx"].to(device)
            pm  = batch["padding_mask"].to(device)
            tgt_wt  = batch["target_wt"].to(device)
            tgt_if  = batch["target_if"].to(device)
            tgt_dur = batch["target_dur"].to(device)
            tgt_ftp = batch["ftp_delta"].to(device)

            optimizer.zero_grad()
            out = model(x, di, pm)

            loss_wt  = ce_loss(out["workout_logits"], tgt_wt)
            loss_if  = mse_loss(out["intensity"].squeeze(-1), tgt_if)
            loss_dur = mse_loss(out["duration"].squeeze(-1), tgt_dur)
            loss_ftp = huber_loss(out["ftp_delta"].squeeze(-1), tgt_ftp)

            loss = loss_wt + 0.5 * loss_if + 0.3 * loss_dur + 0.05 * loss_ftp
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()
        avg_train = train_loss / max(1, n_train_batches)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        correct_wt = 0
        total_wt = 0
        if_mae_sum = 0.0
        ftp_mae_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x   = batch["x"].to(device)
                di  = batch["day_idx"].to(device)
                pm  = batch["padding_mask"].to(device)
                tgt_wt  = batch["target_wt"].to(device)
                tgt_if  = batch["target_if"].to(device)
                tgt_dur = batch["target_dur"].to(device)
                tgt_ftp = batch["ftp_delta"].to(device)

                out = model(x, di, pm)
                loss = (
                    ce_loss(out["workout_logits"], tgt_wt)
                    + 0.5 * mse_loss(out["intensity"].squeeze(-1), tgt_if)
                    + 0.3 * mse_loss(out["duration"].squeeze(-1), tgt_dur)
                    + 0.05 * huber_loss(out["ftp_delta"].squeeze(-1), tgt_ftp)
                )
                val_loss += loss.item()
                n_val_batches += 1

                preds = out["workout_logits"].argmax(dim=1)
                correct_wt += (preds == tgt_wt).sum().item()
                total_wt += len(tgt_wt)
                if_mae_sum += (out["intensity"].squeeze(-1) - tgt_if).abs().sum().item()
                ftp_mae_sum += (out["ftp_delta"].squeeze(-1) - tgt_ftp).abs().sum().item()

        avg_val = val_loss / max(1, n_val_batches)
        wt_acc  = correct_wt / max(1, total_wt) * 100
        if_mae  = if_mae_sum / max(1, total_wt)
        ftp_mae = ftp_mae_sum / max(1, total_wt)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"wt_acc={wt_acc:.1f}%  IF_MAE={if_mae:.3f}  "
            f"FTPΔ_MAE={ftp_mae:.1f}W  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best model → {args.output}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CyclingTransformer")
    parser.add_argument("--data",       required=True)
    parser.add_argument("--output",     default="./backend/models/cycling_coach.pt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seq-len",    type=int, default=90)
    parser.add_argument("--val-frac",   type=float, default=0.1)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--d-model",    type=int, default=128)
    parser.add_argument("--nhead",      type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--d-ff",       type=int, default=512)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--fast",       action="store_true",
                        help="Smaller model for quick CPU testing")
    args = parser.parse_args()
    train(args)
