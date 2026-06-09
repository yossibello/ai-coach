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
       • 4-week FTP / 5-min / 1-min fractional delta (goal-weighted Huber)

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

  # Fine-tune on real athlete data (RECOMMENDED pattern):
  #   • mix synthetic REPLAY in to anchor the coaching policy (no forgetting)
  #   • tag provenance so policy heads outcome-weight the real data automatically
  #   • mask the circular workout-type/zone INPUT features on real data
  python -m ml.training.train \\
      --data "./ml/data/synthetic.parquet=synthetic,./ml/data/goldencheetah.parquet=real" \\
      --output ./backend/models/cycling_coach_ft.pt \\
      --checkpoint ./backend/models/cycling_coach.pt \\
      --mask-workout-type \\
      --epochs 30 --lr 5e-5

Two head families, two truth sources (see README / dataset.py):
  • POLICY   heads (workout_type, intensity, duration) learn WHAT TO PRESCRIBE.
    On real data they are outcome-weighted — the model imitates only the riders
    who actually got faster, never the amateur average. Synthetic = weight 1.0.
  • FORECAST heads (ftp/pc deltas, risk) learn WHAT THE BODY DOES. They train on
    ALL data unweighted — declines and overtraining are the negative examples
    the physiology model needs to predict and warn about.
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
from app.ml.norm import ACTIVITY_WORKOUT_TYPE_DIMS, ACTIVITY_ZONE_DIMS
from ml.training.dataset import CyclingDataset, athlete_split, collate_fn

# Per-goal loss weights for [ftp_delta, pc5min_delta, pc1min_delta].
# Indices match GOAL_TYPE_IDX in app.ml.norm:
#   0=general_fitness  1=ftp_improvement  2=weight_loss  3=event_specific
#   4=gran_fondo       5=criterium        6=climbing      7=triathlon
_GOAL_WEIGHTS = torch.tensor([
    [0.55, 0.25, 0.20],   # general_fitness  — FTP-heavy, touch all systems
    [0.70, 0.20, 0.10],   # ftp_improvement  — pure aerobic focus
    [0.60, 0.25, 0.15],   # weight_loss      — aerobic volume dominant
    [0.35, 0.30, 0.35],   # event_specific   — balanced, unknown event
    [0.55, 0.30, 0.15],   # gran_fondo       — aerobic + VO2max for climbs
    [0.25, 0.30, 0.45],   # criterium        — anaerobic surges dominant
    [0.50, 0.40, 0.10],   # climbing         — VO2max-limited on steep grades
    [0.65, 0.25, 0.10],   # triathlon        — aerobic efficiency, no sprint
], dtype=torch.float32)


def train(args):
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    print(f"PyTorch threads: {n_threads}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
    print(f"Training on: {device}  ({n_gpus} GPU{'s' if n_gpus > 1 else ''})")
    if device.type == "cuda":
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print("  Apple Silicon GPU (MPS)")

    # ── Load data ─────────────────────────────────────────────────────────
    # `--data` accepts a comma-separated list of sources, each optionally tagged
    # with its provenance via `path=source` (source ∈ {synthetic, real}). This
    # is how synthetic REPLAY is mixed into a real-data fine-tune in one run:
    #     --data "synthetic.parquet=synthetic,goldencheetah.parquet=real"
    # Provenance drives per-sample policy weighting in the dataset: synthetic
    # prescriptions are always imitated; real ones only when the rider improved.
    # When a path has no `=source` tag, the file's own `source` column is used
    # (defaulting to synthetic). Athlete IDs are namespaced across files so an
    # id collision between two sources never merges two different athletes.
    def _read_one(path: str) -> pd.DataFrame:
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    _parts = []
    _id_offset = 0
    for _item in args.data.split(","):
        _item = _item.strip()
        if not _item:
            continue
        if "=" in _item:
            _path, _src = _item.rsplit("=", 1)
            _path, _src = _path.strip(), _src.strip()
        else:
            _path, _src = _item, None
        print(f"Loading data from {_path}…" + (f"  (source={_src})" if _src else ""))
        _d = _read_one(_path)
        if _src is not None:
            _d["source"] = _src
        elif "source" not in _d.columns:
            _d["source"] = "synthetic"
        # Namespace athlete ids so disjoint sources never collide.
        _d["athlete_id"] = _d["athlete_id"].astype("int64") + _id_offset
        _id_offset = int(_d["athlete_id"].max()) + 1
        _parts.append(_d)

    df = _parts[0] if len(_parts) == 1 else pd.concat(_parts, ignore_index=True)
    del _parts
    _src_counts = df["source"].value_counts().to_dict()
    print(f"Total rows: {len(df):,}  athletes: {df['athlete_id'].nunique():,}  "
          f"by source: {_src_counts}")

    # ── Split by athlete (no leakage) ─────────────────────────────────────
    train_ids, val_ids = athlete_split(df, val_frac=args.val_frac, seed=args.seed)
    print(f"Train athletes: {len(train_ids):,}  Val athletes: {len(val_ids):,}")

    # Pre-split and sort before constructing datasets so the full 12M-row df
    # is freed before the expensive encode step runs (saves ~3.5 GB at peak).
    train_df = (df[df["athlete_id"].isin(set(train_ids))]
                .sort_values(["athlete_id", "date"]).reset_index(drop=True))
    val_df   = (df[df["athlete_id"].isin(set(val_ids))]
                .sort_values(["athlete_id", "date"]).reset_index(drop=True))
    del df

    # getattr defaults so hand-built arg objects (e.g. the Kaggle notebook's
    # argparse.Namespace) keep working even if they predate these flags.
    _ow  = not getattr(args, "no_outcome_weighting", False)
    _osc = getattr(args, "outcome_scale", 0.05)
    _odb = getattr(args, "outcome_deadband", 0.0)
    _sfw = getattr(args, "synthetic_forecast_weight", 1.0)
    train_ds = CyclingDataset(
        train_df, seq_len=args.seq_len, already_sorted=True,
        outcome_weighting=_ow, outcome_scale=_osc, outcome_deadband=_odb,
        synthetic_forecast_weight=_sfw,
    )

    # Compute inverse-frequency class weights for workout-type CE loss.
    # Without this, the model collapses to predicting "endurance" (~35% of
    # all rides), since the 11-class distribution is highly skewed.
    from collections import Counter as _Counter
    from app.ml.norm import WORKOUT_TYPES as _WT
    _wt_counts  = _Counter(int(v) for v in train_ds.wt_idx)
    _total      = sum(_wt_counts.values())
    _n          = len(_WT)
    _freq       = [max(_wt_counts.get(i, 1), 1) / _total for i in range(_n)]
    # Sqrt of inverse-frequency, capped at 3×, normalised so mean = 1.
    # Linear inverse-freq (cap 8×) makes gradients on rare classes too large
    # relative to the learning rate, causing oscillation. Sqrt is a softer
    # re-weighting that nudges the model toward rare classes without dominating.
    import math as _math
    _raw_w      = [min(_math.sqrt(1.0 / (f * _n)), 3.0) for f in _freq]
    _mean_w     = sum(_raw_w) / _n
    _wt_class_w = [w / _mean_w for w in _raw_w]
    print(f"Workout class distribution: {dict(zip(_WT, [round(_wt_counts.get(i,0)/_total*100,1) for i in range(_n)]))}")
    print(f"Workout class weights (mean-norm): {dict(zip(_WT, [round(w, 2) for w in _wt_class_w]))}")

    del train_df
    val_ds   = CyclingDataset(
        val_df, seq_len=args.seq_len, already_sorted=True,
        outcome_weighting=_ow, outcome_scale=_osc, outcome_deadband=_odb,
        synthetic_forecast_weight=_sfw,
    )
    del val_df
    print(f"Train sequences: {len(train_ds):,}  Val sequences: {len(val_ds):,}")

    # Allow override via env var for high-core-count machines (default: 8 on GPU)
    # 4 workers instead of 8: halves fork copy-on-write memory pressure on the
    # token matrix (~1.5 GB float16 × 4 = 6 GB vs × 8 = 12 GB) — avoids OOM
    # on Kaggle T4 x2 (29 GB RAM) with 50K athletes.
    _default_workers = 4 if device.type == "cuda" else min(2, n_threads)
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

    start_epoch = 0
    _resume_optimizer_state = None  # loaded below, applied after optimizer is created
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Handle both raw state_dict and full checkpoint dict (current format).
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            # Strip _orig_mod. prefix if checkpoint was saved from a non-compiled
            # model but we're now loading into a torch.compile-wrapped model,
            # or vice versa.
            sd = ckpt["state_dict"]
            model_keys = set(model.state_dict().keys())
            first_sd_key = next(iter(sd))
            first_model_key = next(iter(model_keys))
            if first_sd_key.startswith("_orig_mod.") and not first_model_key.startswith("_orig_mod."):
                sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
            elif not first_sd_key.startswith("_orig_mod.") and first_model_key.startswith("_orig_mod."):
                sd = {"_orig_mod." + k: v for k, v in sd.items()}
            model.load_state_dict(sd)
            saved_epoch = ckpt.get("metrics", {}).get("epoch", 0)
            # Resume only if mid-run (saved_epoch < total epochs).
            # If saved_epoch >= args.epochs it's a fine-tune: use weights only,
            # reset epoch counter so the loop and scheduler start fresh.
            if saved_epoch < args.epochs:
                start_epoch = saved_epoch
                _resume_optimizer_state = ckpt.get("optimizer_state")
                print(f"Resuming from checkpoint: {args.checkpoint} (epoch {start_epoch} → {args.epochs})")
            else:
                print(f"Fine-tune mode: loaded weights from epoch {saved_epoch}, starting fresh at epoch 1")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded checkpoint: {args.checkpoint}")

    if n_gpus > 1:
        # Force cuBLAS handle creation on every GPU before DataParallel spawns
        # its internal threads. torch.empty() only allocates; a real matmul is
        # needed to actually initialise the cuBLAS context on each device.
        for i in range(n_gpus):
            with torch.cuda.device(i):
                d = torch.randn(8, 8, device=f"cuda:{i}")
                _ = d @ d
                del d
        torch.cuda.set_device(0)
        model = nn.DataParallel(model)
        print(f"DataParallel across {n_gpus} GPUs — effective batch size: {args.batch_size} ({args.batch_size // n_gpus}/GPU)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer / Scheduler / Losses ────────────────────────────────────
    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if _resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(_resume_optimizer_state)
            print("  Optimizer state restored from checkpoint (AdamW momentum preserved).")
        except Exception as e:
            print(f"  Warning: could not restore optimizer state ({e}); starting optimizer fresh.")
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # Fast-forward scheduler to the resumed epoch. A dummy zero-grad step
    # registers initial_lr in param groups so the fast-forward doesn't trigger
    # PyTorch's "called before optimizer.step()" warning.
    if start_epoch > 0:
        # Dummy step so PyTorch registers optimizer.step() before scheduler.step()
        optimizer.zero_grad()
        optimizer.step()
        for _ in range(start_epoch):
            scheduler.step()
    _amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    scaler     = GradScaler(device=_amp_device, enabled=device.type == "cuda" and not args.no_amp)
    _wt_class_w_t = torch.tensor(_wt_class_w, dtype=torch.float32, device=device)
    ce_loss       = nn.CrossEntropyLoss(
        label_smoothing=0.1,
        weight=_wt_class_w_t,
    )
    # Per-sample (unreduced) variants for outcome-weighted POLICY losses.
    # We reduce manually so each sample can be scaled by its `policy_weight`
    # (1.0 for synthetic/expert, outcome-advantage for real). With all weights
    # 1.0 these reproduce the original reduced losses EXACTLY (see _pol_* below),
    # so pure-synthetic pre-training is unchanged.
    ce_loss_none  = nn.CrossEntropyLoss(
        label_smoothing=0.1, weight=_wt_class_w_t, reduction="none",
    )
    mse_none      = nn.MSELoss(reduction="none")
    mse_loss      = nn.MSELoss()
    # delta=0.05: transition from L2→L1 at 5% — appropriate for fractional targets
    huber_loss    = nn.HuberLoss(delta=0.05)
    huber_none    = nn.HuberLoss(delta=0.05, reduction="none")  # for per-sample weighting
    goal_w_table  = _GOAL_WEIGHTS.to(device)                    # (8, 3)
    # Risk losses: 3-class softmax CE for over/under/neither (mutually exclusive)
    # and BCEWithLogits for independent injury prediction. Class weights bias
    # toward the minority over/under classes since 'neither' dominates.
    ce_risk_ot = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 0.5], device=device))
    bce_risk_inj = nn.BCEWithLogitsLoss()

    def _pol_losses(out, tgt_wt, tgt_if, tgt_dur, pw):
        """Outcome-weighted policy losses (workout-type, intensity, duration).

        Each sample's contribution is scaled by `pw` ∈ [0,1] — its imitation
        weight. Synthetic samples carry pw=1.0 so the reductions below collapse
        to the original class-weighted-mean / mean losses; real samples are
        down-weighted toward 0 when the rider did not actually improve, so the
        model never learns to copy losing prescriptions.
        """
        cw = _wt_class_w_t[tgt_wt]                              # (B,) class weight per target
        wt_num = (ce_loss_none(out["workout_logits"], tgt_wt) * pw).sum()
        loss_wt = wt_num / (cw * pw).sum().clamp_min(1e-6)
        pw_den  = pw.sum().clamp_min(1e-6)
        loss_if  = (mse_none(out["intensity"].squeeze(-1), tgt_if)  * pw).sum() / pw_den
        loss_dur = (mse_none(out["duration"].squeeze(-1), tgt_dur) * pw).sum() / pw_den
        return loss_wt, loss_if, loss_dur

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
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        pw_sum = 0.0   # running mean of policy (imitation) weight — diagnostic

        for batch in train_loader:
            if steps_per_epoch and n_train_batches >= steps_per_epoch:
                break
            x   = batch["x"].to(device)
            di  = batch["day_idx"].to(device)
            pm  = batch["padding_mask"].to(device)
            hq  = batch.get("horizon_query")
            if hq is not None:
                hq = hq.to(device)
            if getattr(args, "mask_workout_type", False):
                x[:, :, ACTIVITY_ZONE_DIMS]         = 0.0
                x[:, :, ACTIVITY_WORKOUT_TYPE_DIMS] = 0.0
            tgt_wt   = batch["target_wt"].to(device)
            tgt_if   = batch["target_if"].to(device)
            tgt_dur  = batch["target_dur"].to(device)
            tgt_ftp  = batch["ftp_delta"].to(device)
            tgt_pc5  = batch["pc5min_delta"].to(device)
            tgt_pc1  = batch["pc1min_delta"].to(device)
            tgt_goal = batch["goal_idx"].to(device)
            tgt_rot  = batch["target_risk_ot"].to(device)
            tgt_rinj = batch["target_risk_inj"].to(device)
            pw       = batch["policy_weight"].to(device)
            fw       = batch["forecast_weight"].to(device)

            # Per-sample goal weights: (B, 3) → columns [w_ftp, w_pc5, w_pc1]
            gw = goal_w_table[tgt_goal]   # (B, 3)

            optimizer.zero_grad()
            with autocast(device_type=_amp_device, enabled=use_amp):
                out = model(x, di, pm, horizon_query=hq)

                # POLICY heads (what to prescribe): outcome-weighted imitation.
                loss_wt, loss_if, loss_dur = _pol_losses(out, tgt_wt, tgt_if, tgt_dur, pw)
                # FORECAST heads (what the body does): train on ALL data, never
                # outcome-weighted — declines are needed negative examples.
                loss_rot  = ce_risk_ot(out["risk_ot_logits"], tgt_rot)
                loss_rinj = bce_risk_inj(out["risk_inj_logit"].squeeze(-1), tgt_rinj)

                # Goal-weighted fitness losses — mask NaN for old data without capacity cols
                pred_ftp = out["ftp_delta"].squeeze(-1)
                pred_pc5 = out["pc5min_delta"].squeeze(-1)
                pred_pc1 = out["pc1min_delta"].squeeze(-1)

                valid_ftp = ~tgt_ftp.isnan()
                valid_pc5 = ~tgt_pc5.isnan()
                valid_pc1 = ~tgt_pc1.isnan()

                def _wloss(pred, tgt, valid, col):
                    # Forecast loss weighted per-sample by `fw` (forecast_weight):
                    # real rows = 1.0, synthetic rows = synthetic_forecast_weight
                    # (0.0 in fine-tune → forecast learns from real rows ONLY).
                    # Normalizing by fw.sum() means: when fw≡1 this is the plain
                    # goal-weighted mean (pretrain unchanged); when synthetic
                    # rows are 0 it becomes the mean over REAL rows only — same
                    # magnitude as training on real data alone.
                    if not valid.any():
                        return torch.tensor(0.0, device=device)
                    fwv = fw[valid]
                    num = (huber_none(pred[valid], tgt[valid]) * gw[valid, col] * fwv).sum()
                    return num / fwv.sum().clamp_min(1e-6)

                loss_ftp = _wloss(pred_ftp, tgt_ftp, valid_ftp, 0)
                loss_pc5 = _wloss(pred_pc5, tgt_pc5, valid_pc5, 1)
                loss_pc1 = _wloss(pred_pc1, tgt_pc1, valid_pc1, 2)

                loss = (
                    loss_wt
                    + 0.5  * loss_if
                    + 0.3  * loss_dur
                    + 0.10 * (loss_ftp + loss_pc5 + loss_pc1)
                    + 0.10 * loss_rot
                    + 0.05 * loss_rinj
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pw_sum += pw.mean().item()
            n_train_batches += 1

            if n_train_batches % log_every == 0:
                print(
                    f"  Epoch {epoch} | step {n_train_batches:,}/{steps_per_epoch or total_train_batches:,}"
                    f" | loss {train_loss/n_train_batches:.4f}",
                    flush=True,
                )

        scheduler.step()
        avg_train = train_loss / max(1, n_train_batches)
        avg_pw    = pw_sum / max(1, n_train_batches)  # ~1.0 pure synthetic; <1.0 once real data is weighted in

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        correct_wt = 0
        total_wt = 0
        if_mae_sum = 0.0
        ftp_mae_sum = 0.0

        # Cap validation steps to match training steps (keeps epoch time predictable).
        # Full val sweep would be 1000s of batches on large datasets.
        _max_val_steps = args.steps_per_epoch if args.steps_per_epoch else len(val_loader)
        with torch.no_grad():
            for _val_step, batch in enumerate(val_loader):
                if _val_step >= _max_val_steps:
                    break
                x   = batch["x"].to(device)
                di  = batch["day_idx"].to(device)
                pm  = batch["padding_mask"].to(device)
                hq  = batch.get("horizon_query")
                if hq is not None:
                    hq = hq.to(device)
                if getattr(args, "mask_workout_type", False):
                    x[:, :, ACTIVITY_ZONE_DIMS]         = 0.0
                    x[:, :, ACTIVITY_WORKOUT_TYPE_DIMS] = 0.0
                tgt_wt   = batch["target_wt"].to(device)
                tgt_if   = batch["target_if"].to(device)
                tgt_dur  = batch["target_dur"].to(device)
                tgt_ftp  = batch["ftp_delta"].to(device)
                tgt_pc5  = batch["pc5min_delta"].to(device)
                tgt_pc1  = batch["pc1min_delta"].to(device)
                tgt_goal = batch["goal_idx"].to(device)
                tgt_rot  = batch["target_risk_ot"].to(device)
                tgt_rinj = batch["target_risk_inj"].to(device)
                pw       = batch["policy_weight"].to(device)
                fw       = batch["forecast_weight"].to(device)

                gw = goal_w_table[tgt_goal]
                out = model(x, di, pm, horizon_query=hq)
                loss_wt_v, loss_if_v, loss_dur_v = _pol_losses(out, tgt_wt, tgt_if, tgt_dur, pw)

                pred_ftp = out["ftp_delta"].squeeze(-1)
                pred_pc5 = out["pc5min_delta"].squeeze(-1)
                pred_pc1 = out["pc1min_delta"].squeeze(-1)
                valid_ftp = ~tgt_ftp.isnan()
                valid_pc5 = ~tgt_pc5.isnan()
                valid_pc1 = ~tgt_pc1.isnan()

                loss_ftp_v = _wloss(pred_ftp, tgt_ftp, valid_ftp, 0)
                loss_pc5_v = _wloss(pred_pc5, tgt_pc5, valid_pc5, 1)
                loss_pc1_v = _wloss(pred_pc1, tgt_pc1, valid_pc1, 2)

                loss = (
                    loss_wt_v
                    + 0.5  * loss_if_v
                    + 0.3  * loss_dur_v
                    + 0.10 * (loss_ftp_v + loss_pc5_v + loss_pc1_v)
                    + 0.10 * ce_risk_ot(out["risk_ot_logits"], tgt_rot)
                    + 0.05 * bce_risk_inj(out["risk_inj_logit"].squeeze(-1), tgt_rinj)
                )
                val_loss += loss.item()
                n_val_batches += 1

                preds = out["workout_logits"].argmax(dim=1)
                correct_wt += (preds == tgt_wt).sum().item()
                total_wt += len(tgt_wt)
                if_mae_sum += (out["intensity"].squeeze(-1) - tgt_if).abs().sum().item()
                if valid_ftp.any():
                    ftp_mae_sum += (pred_ftp[valid_ftp] - tgt_ftp[valid_ftp]).abs().sum().item()

        avg_val  = val_loss / max(1, n_val_batches)
        wt_acc   = correct_wt / max(1, total_wt) * 100
        if_mae   = if_mae_sum / max(1, total_wt)
        ftp_mae  = ftp_mae_sum / max(1, total_wt)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"wt_acc={wt_acc:.1f}%  IF_MAE={if_mae:.3f}  "
            f"FTPΔ_MAE={ftp_mae*100:.2f}%  pol_w={avg_pw:.2f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            patience_count = 0
            # Save state_dict + model config so inference can rebuild the
            # exact same architecture (otherwise dim mismatches silently
            # fall back to cold-start).
            raw_model = model
            if hasattr(raw_model, "_orig_mod"):       # torch.compile wrapper
                raw_model = raw_model._orig_mod
            if isinstance(raw_model, nn.DataParallel): # DataParallel wrapper
                raw_model = raw_model.module
            ckpt = {
                "state_dict": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
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
            # Auto-backup to /kaggle/working/ (safe no-op outside Kaggle)
            import shutil as _shutil
            _kaggle_out = "/kaggle/working/cycling_coach_best.pt"
            try:
                _shutil.copy(args.output, _kaggle_out)
                print(f"  ✓ Backed up → {_kaggle_out}")
            except Exception:
                pass
        else:
            patience_count += 1
            if args.patience > 0 and patience_count >= args.patience:
                print(f"  Early stopping: val loss hasn't improved for {args.patience} epochs.")
                break

        # Always save a "last" checkpoint so resume picks up from the actual
        # last epoch, not the best epoch.
        raw_model = model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if isinstance(raw_model, nn.DataParallel):
            raw_model = raw_model.module
        last_ckpt = {
            "state_dict": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
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
        last_path = args.output.replace(".pt", "_last.pt")
        torch.save(last_ckpt, last_path)
        print(f"  ✓ Last checkpoint → {last_path}")
        import shutil as _shutil
        try:
            _shutil.copy(last_path, "/kaggle/working/cycling_coach_last.pt")
            print(f"  ✓ Backed up → /kaggle/working/cycling_coach_last.pt")
        except Exception:
            pass

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
    parser.add_argument("--mask-workout-type", action="store_true",
                        help="Zero out workout_type one-hot (dims 32-42) and zone fractions "
                             "(dims 17-23) in the activity input. Use when fine-tuning on real "
                             "data where workout_type is inferred from IF/duration (circular).")
    parser.add_argument("--no-outcome-weighting", action="store_true",
                        help="Disable outcome-weighting of the policy heads. By default, REAL "
                             "samples (source=real) only train the workout/intensity/duration "
                             "heads in proportion to the rider's realized FTP gain — so the model "
                             "imitates winners, not the amateur average. Synthetic data is "
                             "unaffected (weight 1.0). Pass this to revert to plain imitation.")
    parser.add_argument("--outcome-scale", type=float, default=0.05,
                        help="Fractional 4-week FTP gain that earns full imitation weight (1.0) "
                             "for real data. Default 0.05 = +5%%.")
    parser.add_argument("--outcome-deadband", type=float, default=0.0,
                        help="Minimum fractional FTP gain before a real sample gets any imitation "
                             "weight. Default 0.0 (any improvement counts; flat/declining → 0).")
    parser.add_argument("--synthetic-forecast-weight", type=float, default=1.0,
                        help="Weight of SYNTHETIC rows in the forecast (FTP/power-curve delta) "
                             "losses. 1.0 for pure-synthetic pretrain (default). Set 0.0 when "
                             "fine-tuning with real data so the forecast heads learn physiology "
                             "from MEASURED outcomes only — synthetic deltas are simulated and "
                             "would otherwise wash out the real dose-response. Policy heads still "
                             "use the synthetic replay regardless (that's what protects coaching).")
    args = parser.parse_args()
    train(args)
