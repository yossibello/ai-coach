"""
Compare main model vs GoldenCheetah fine-tuned model on real activity sequences.
Usage: PYTHONPATH=backend python3 ml/compare_models.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import torch

from app.ml.model import CyclingTransformer, encode_horizon
from app.ml.norm import encode_activity_dataframe, encode_profile_dataframe, WORKOUT_TYPES

# ── Load both models ──────────────────────────────────────────────────────────
def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg  = ckpt.get('config', {})
    m = CyclingTransformer(
        d_model         = cfg.get('d_model', 256),
        nhead           = cfg.get('nhead', 8),
        num_layers      = cfg.get('num_layers', 8),
        dim_feedforward = cfg.get('dim_feedforward', 1024),
        dropout         = 0.0,
    )
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    mt = ckpt.get('metrics', {})
    return m, mt

# Optional CLI overrides: compare_models.py [MAIN.pt] [FINE_TUNED.pt] [DATA.parquet]
_MAIN_PATH = sys.argv[1] if len(sys.argv) > 1 else 'backend/models/cycling_coach.pt'
_FT_PATH   = sys.argv[2] if len(sys.argv) > 2 else 'backend/models/cycling_coach_ft_gc.pt'
_DATA_PATH = sys.argv[3] if len(sys.argv) > 3 else 'ml/data/goldencheetah.parquet'

print("Loading models…")
main_model, main_meta = load_model(_MAIN_PATH)
ft_model,   ft_meta   = load_model(_FT_PATH)

print(f"  Main model : epoch {main_meta.get('epoch','?')}  "
      f"val_loss={main_meta.get('val_loss',0):.4f}  "
      f"wt_acc={main_meta.get('wt_acc',0):.1f}%")
print(f"  Fine-tuned : epoch {ft_meta.get('epoch','?')}  "
      f"val_loss={ft_meta.get('val_loss',0):.4f}  "
      f"wt_acc={ft_meta.get('wt_acc',0):.1f}%")
print()

# ── Load real GoldenCheetah sequences ─────────────────────────────────────────
SEQ_LEN = 90
df = pd.read_parquet(_DATA_PATH)

# Pick 5 athletes with enough rides and diverse workout types
counts = df.groupby('athlete_id').size()
good   = counts[counts >= SEQ_LEN + 5].index.tolist()
rng    = np.random.default_rng(42)
sample_ids = rng.choice(good, size=min(5, len(good)), replace=False)

WORKOUT_LABELS = WORKOUT_TYPES  # list of strings

def run_model(model, x_tensor, day_idx):
    with torch.no_grad():
        out = model(x_tensor, day_idx, padding_mask=None, horizon_query=None)
    wt_probs = torch.softmax(out['workout_logits'], dim=-1)[0].numpy()
    wt_pred  = WORKOUT_LABELS[int(wt_probs.argmax())]
    intensity = float(out['intensity'].squeeze())
    ftp_delta = float(out['ftp_delta'].squeeze()) * 100   # → %
    risk_ot   = torch.softmax(out['risk_ot_logits'], dim=-1)[0].numpy()
    risk_inj  = float(torch.sigmoid(out['risk_inj_logit'].squeeze()))
    return {
        'wt_pred':   wt_pred,
        'wt_conf':   float(wt_probs.max()) * 100,
        'intensity': intensity,
        'ftp_delta': ftp_delta,
        'risk_ot':   risk_ot,
        'risk_inj':  risk_inj * 100,
    }

print("=" * 70)
for aid in sample_ids:
    athlete_df = df[df.athlete_id == aid].sort_values('date').copy()
    if len(athlete_df) < SEQ_LEN + 1:
        continue

    # Last SEQ_LEN rides as context, next ride as "what would model predict"
    context = athlete_df.iloc[-(SEQ_LEN+1):-1].copy()
    actual  = athlete_df.iloc[-1]

    # Encode using norm.py — activity (52) + profile (11) = 63 dims per step
    act_enc  = encode_activity_dataframe(context)       # (SEQ_LEN, 52)
    prof_enc = encode_profile_dataframe(context)        # (SEQ_LEN, 11)
    encoded  = np.concatenate([act_enc, prof_enc], axis=1)  # (SEQ_LEN, 63)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)  # (1, SEQ_LEN, 63)
    day_idx = torch.zeros(1, SEQ_LEN, dtype=torch.long)

    m_out  = run_model(main_model, x, day_idx)
    ft_out = run_model(ft_model,   x, day_idx)

    actual_wt  = actual.get('workout_type', '?')
    actual_if  = actual.get('intensity_factor', float('nan'))
    actual_ftp = actual.get('ftp', float('nan'))

    print(f"Athlete {aid}  |  last {SEQ_LEN} rides  |  actual next: {actual_wt}  IF={actual_if:.2f}")
    print(f"  {'':20s}  {'MAIN MODEL':>22}   {'FINE-TUNED (GC)':>22}")
    print(f"  {'Workout type':20s}  {m_out['wt_pred']:>18} {m_out['wt_conf']:.0f}%   {ft_out['wt_pred']:>18} {ft_out['wt_conf']:.0f}%")
    print(f"  {'Intensity factor':20s}  {m_out['intensity']:>22.3f}   {ft_out['intensity']:>22.3f}   (actual {actual_if:.3f})")
    print(f"  {'FTP Δ (4-week)':20s}  {m_out['ftp_delta']:>21.1f}%   {ft_out['ftp_delta']:>21.1f}%")
    print(f"  {'Risk overtrain':20s}  over={m_out['risk_ot'][0]:.2f} under={m_out['risk_ot'][1]:.2f}   over={ft_out['risk_ot'][0]:.2f} under={ft_out['risk_ot'][1]:.2f}")
    print(f"  {'Risk injury':20s}  {m_out['risk_inj']:>21.1f}%   {ft_out['risk_inj']:>21.1f}%")
    print()

print("=" * 70)


# ── FORECAST CALIBRATION (the promotion gate) ──────────────────────────────────
# The policy heads can't be scored on real data (no ground-truth "optimal" next
# workout exists). But the FORECAST heads CAN: we know what the rider's FTP
# actually did. This measures how well each model predicts the realized 4-week
# FTP change on real athletes — the metric that should gate whether a fine-tuned
# checkpoint is allowed to replace the production model.
#
#   FTPΔ MAE   — mean abs error between predicted and realized fractional FTP Δ
#   sign acc   — % of time the model gets the DIRECTION right (up vs down)
#   corr       — Pearson correlation of predicted vs realized Δ
# Lower MAE + higher sign-acc/corr = better-calibrated physiology model.
FORECAST_DAYS = 28
H_MEDIUM = torch.tensor(encode_horizon("medium", FORECAST_DAYS),
                        dtype=torch.float32).unsqueeze(0)  # (1, 6)


def collect_calibration(model, df, sample_ids, max_cuts_per_athlete=6):
    pred_d, real_d = [], []
    for aid in sample_ids:
        a = df[df.athlete_id == aid].sort_values('date').copy()
        if len(a) < SEQ_LEN + 2:
            continue
        dates = a['date'].to_numpy(dtype='datetime64[ns]')
        ftp   = a['ftp'].to_numpy(dtype=np.float64)
        act_enc  = encode_activity_dataframe(a)
        prof_enc = encode_profile_dataframe(a)
        enc      = np.concatenate([act_enc, prof_enc], axis=1)

        # Candidate cut points: a history end with a ride ≥28d ahead to measure.
        cut_lo, cut_hi = SEQ_LEN, len(a) - 1
        if cut_hi <= cut_lo:
            continue
        cuts = np.linspace(cut_lo, cut_hi,
                           num=min(max_cuts_per_athlete, cut_hi - cut_lo), dtype=int)
        for cut in sorted(set(int(c) for c in cuts)):
            now_date = dates[cut - 1]
            future_idx = np.searchsorted(
                dates, now_date + np.timedelta64(FORECAST_DAYS, 'D'), side='left')
            if future_idx >= len(a):
                continue  # no measurement 4 weeks out
            ftp_now, ftp_fut = ftp[cut - 1], ftp[future_idx]
            if ftp_now <= 1:
                continue
            realized = (ftp_fut - ftp_now) / ftp_now

            window = enc[cut - SEQ_LEN:cut]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            day_idx = torch.tensor(
                ((dates[cut - SEQ_LEN:cut] - dates[cut - SEQ_LEN])
                 .astype('timedelta64[D]').astype(np.int64)),
                dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = model(x, day_idx, padding_mask=None, horizon_query=H_MEDIUM)
            pred_d.append(float(out['ftp_delta'].squeeze()))
            real_d.append(float(realized))
    return np.array(pred_d), np.array(real_d)


def calib_stats(pred, real):
    if len(pred) == 0:
        return None
    mae  = float(np.abs(pred - real).mean())
    sign = float(((pred >= 0) == (real >= 0)).mean()) * 100
    if pred.std() > 1e-9 and real.std() > 1e-9:
        corr = float(np.corrcoef(pred, real)[0, 1])
    else:
        corr = float('nan')
    return dict(n=len(pred), mae=mae, sign=sign, corr=corr,
                pred_mean=float(pred.mean()), real_mean=float(real.mean()))


# Larger athlete pool for stable calibration numbers.
calib_ids = good if len(good) <= 60 else list(rng.choice(good, size=60, replace=False))
print("\nFORECAST CALIBRATION — predicted vs realized 4-week FTP Δ on real athletes")
print(f"(pool: {len(calib_ids)} athletes, up to 6 cut points each)\n")
for name, model in (("MAIN", main_model), ("FINE-TUNED (GC)", ft_model)):
    s = calib_stats(*collect_calibration(model, df, calib_ids))
    if s is None:
        print(f"  {name:18s}  no eligible samples")
        continue
    print(f"  {name:18s}  n={s['n']:4d}  FTPΔ_MAE={s['mae']*100:5.2f}%  "
          f"sign_acc={s['sign']:5.1f}%  corr={s['corr']:+.3f}  "
          f"(pred μ={s['pred_mean']*100:+.2f}%  real μ={s['real_mean']*100:+.2f}%)")
print()
print("Gate rule of thumb: only promote the fine-tuned model if its FTPΔ_MAE is")
print("LOWER and sign_acc/corr are no worse than MAIN. If it wins here, real data")
print("is genuinely improving the physiology model — which is the whole point.")
print("=" * 70)
