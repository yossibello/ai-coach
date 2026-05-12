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
from torch.amp import GradScaler, autocast
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
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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

    # Allow override via env var for high-core-count machines (default: 8 on GPU)
    _default_workers = 8 if device.type == "cuda" else min(2, n_threads)
    n_workers_train = int(os.environ.get("DATALOADER_WORKERS", min(_default_workers, n_threads)))
    n_workers_val   = max(1, n_workers_train // 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=n_workers_train,
        pin_memory=device.type == "cuda",
        persistent_workers=n_workers_train > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=n_workers_val,
        persistent_workers=n_workers_val > 0,
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

    if args.compile and device.type == "cuda" and hasattr(torch, "compile"):
        print("Compiling model with torch.compile…")
        model = torch.compile(model)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed from checkpoint: {args.checkpoint}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer / Scheduler / Losses ────────────────────────────────────
    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler     = GradScaler(device=device.type, enabled=device.type == "cuda" and not args.no_amp)
    ce_loss    = nn.CrossEntropyLoss(label_smoothing=0.1)
    mse_loss   = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=10.0)
    # Risk losses: 3-class softmax CE for over/under/neither (mutually exclusive)
    # and BCEWithLogits for independent injury prediction. Class weights bias
    # toward the minority over/under classes since 'neither' dominates.
    ce_risk_ot = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 0.5], device=device))
    bce_risk_inj = nn.BCEWithLogitsLoss()

    use_amp = device.type == "cuda" and not args.no_amp
    if use_amp:
        print("Automatic Mixed Precision (AMP) enabled")

    best_val_loss  = math.inf
    patience_count = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    steps_per_epoch = args.steps_per_epoch  # None = all batches
    total_train_batches = len(train_loader)
    if steps_per_epoch:
        print(f"Steps per epoch capped at {steps_per_epoch:,} (full epoch = {total_train_batches:,} batches)")
    else:
        print(f"Steps per epoch: {total_train_batches:,} batches")
    log_every = max(1, (steps_per_epoch or total_train_batches) // 10)  # print ~10x per epoch

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            if steps_per_epoch and n_train_batches >= steps_per_epoch:
                break
            x   = batch["x"].to(device)
            di  = batch["day_idx"].to(device)
            pm  = batch["padding_mask"].to(device)
            hq  = batch.get("horizon_query")
            if hq is not None:
                hq = hq.to(device)
            tgt_wt  = batch["target_wt"].to(device)
            tgt_if  = batch["target_if"].to(device)
            tgt_dur = batch["target_dur"].to(device)
            tgt_ftp = batch["ftp_delta"].to(device)
            tgt_rot = batch["target_risk_ot"].to(device)
            tgt_rinj = batch["target_risk_inj"].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(x, di, pm, horizon_query=hq)

                loss_wt  = ce_loss(out["workout_logits"], tgt_wt)
                loss_if  = mse_loss(out["intensity"].squeeze(-1), tgt_if)
                loss_dur = mse_loss(out["duration"].squeeze(-1), tgt_dur)
                loss_ftp = huber_loss(out["ftp_delta"].squeeze(-1), tgt_ftp)
                loss_rot  = ce_risk_ot(out["risk_ot_logits"], tgt_rot)
                loss_rinj = bce_risk_inj(out["risk_inj_logit"].squeeze(-1), tgt_rinj)

                loss = (
                    loss_wt
                    + 0.5  * loss_if
                    + 0.3  * loss_dur
                    + 0.05 * loss_ftp
                    + 0.10 * loss_rot
                    + 0.05 * loss_rinj
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_train_batches += 1

            if n_train_batches % log_every == 0:
                print(
                    f"  Epoch {epoch} | step {n_train_batches:,}/{steps_per_epoch or total_train_batches:,}"
                    f" | loss {train_loss/n_train_batches:.4f}",
                    flush=True,
                )

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
                hq  = batch.get("horizon_query")
                if hq is not None:
                    hq = hq.to(device)
                tgt_wt  = batch["target_wt"].to(device)
                tgt_if  = batch["target_if"].to(device)
                tgt_dur = batch["target_dur"].to(device)
                tgt_ftp = batch["ftp_delta"].to(device)
                tgt_rot = batch["target_risk_ot"].to(device)
                tgt_rinj = batch["target_risk_inj"].to(device)

                out = model(x, di, pm, horizon_query=hq)
                loss = (
                    ce_loss(out["workout_logits"], tgt_wt)
                    + 0.5 * mse_loss(out["intensity"].squeeze(-1), tgt_if)
                    + 0.3 * mse_loss(out["duration"].squeeze(-1), tgt_dur)
                    + 0.05 * huber_loss(out["ftp_delta"].squeeze(-1), tgt_ftp)
                    + 0.10 * ce_risk_ot(out["risk_ot_logits"], tgt_rot)
                    + 0.05 * bce_risk_inj(out["risk_inj_logit"].squeeze(-1), tgt_rinj)
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
            best_val_loss  = avg_val
            patience_count = 0
            # Save state_dict + model config so inference can rebuild the
            # exact same architecture (otherwise dim mismatches silently
            # fall back to cold-start).
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt = {
                "state_dict": raw_model.state_dict(),
                "config": {
                    "input_dim": raw_model.input_proj[0].in_features,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "dim_feedforward": d_ff,
                    "dropout": args.dropout,
                    "horizon_dim": getattr(raw_model, "horizon_proj", None) and raw_model.horizon_proj[0].in_features,
                    "horizon_aware": True,
                },
                "metrics": {
                    "epoch": epoch,
                    "val_loss": avg_val,
                    "wt_acc": wt_acc,
                    "if_mae": if_mae,
                    "ftp_mae": ftp_mae,
                },
            }
            torch.save(ckpt, args.output)
            print(f"  ✓ Saved best model → {args.output}")
        else:
            patience_count += 1
            if args.patience > 0 and patience_count >= args.patience:
                print(f"  Early stopping: val loss hasn't improved for {args.patience} epochs.")
                break

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
    parser.add_argument("--patience",   type=int, default=15,
                        help="Early stopping: epochs without val improvement (0=disable)")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Cap training steps per epoch (useful for huge datasets). Default: all batches.")
    parser.add_argument("--compile",    action="store_true",
                        help="Use torch.compile for extra GPU speed (PyTorch 2.0+)")
    parser.add_argument("--no-amp",     action="store_true",
                        help="Disable Automatic Mixed Precision even on GPU")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Cap training steps per epoch (useful for huge datasets). Default: all batches.")
    args = parser.parse_args()
    train(args)
