"""
Direct Python verification of the horizon-aware multi-horizon recommendation
pipeline. Bypasses the HTTP layer; calls generate_multi_horizon_recommendation
directly against the dev SQLite DB for the real test user.

Usage:
    python -X utf8 verify_horizons.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure backend is on sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "backend"))

# Required env
os.environ.setdefault(
    "DATABASE_URL",
    "sqlite+aiosqlite:///c:/Users/yossi/ai coach/aicoach_dev.db",
)
os.environ.setdefault("SECRET_KEY", "dev-secret-stable")

# Allow caller to override which model is loaded
if len(sys.argv) > 1:
    os.environ["ML_MODEL_PATH"] = sys.argv[1]

from sqlalchemy import select as sa_select
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.ml.inference import generate_multi_horizon_recommendation


TEST_EMAIL = "yossi.bello@gmail.com"


def _summarize_horizon(label: str, payload: dict) -> None:
    nw = payload.get("next_workout", {}) or {}
    weekly = payload.get("weekly_plan", []) or []
    safety = payload.get("safety_notes", [])
    print(f"\n=== {label.upper()} — {payload.get('horizon_label', '')} ===")
    print(f"  model_version : {payload.get('model_version')}")
    print(f"  Day-0 type    : {nw.get('workout_type')}")
    print(f"  Day-0 dur(min): {nw.get('duration_minutes')}")
    print(f"  Day-0 TSS     : {nw.get('target_tss')}")
    print(f"  confidence    : {payload.get('confidence')}")
    alts = payload.get("top_alternatives") or payload.get("alternatives") or []
    if alts:
        print(f"  top alts      : {alts}")
    types = [w.get("workout_type") for w in weekly]
    tsss  = [w.get("target_tss")    for w in weekly]
    print(f"  week types    : {types}")
    print(f"  week TSS      : {tsss}  (sum={sum(t or 0 for t in tsss)})")
    if safety:
        for n in safety:
            print(f"  ⚠ safety     : {n}")


async def main() -> int:
    print(f"ML_MODEL_PATH = {os.environ.get('ML_MODEL_PATH', '<default>')}")
    async with AsyncSessionLocal() as db:
        res = await db.execute(sa_select(User).where(User.email == TEST_EMAIL))
        user = res.scalar_one_or_none()
        if user is None:
            print(f"ERROR: no user {TEST_EMAIL} in DB")
            return 1
        print(f"User: {user.email}  id={user.id}")

        result = await generate_multi_horizon_recommendation(user, db)

    print(f"\nis_cold_start  : {result.get('is_cold_start')}")
    print(f"active_horizon : {result.get('active_horizon')}")
    print(f"model_version  : {result.get('model_version')}")

    horizons = result.get("horizons", {})
    for label in ("short", "medium", "event"):
        if label in horizons:
            _summarize_horizon(label, horizons[label])

    # Compare differentiation
    print("\n--- DIFFERENTIATION CHECK ---")
    day0_types = {l: horizons[l]["next_workout"]["workout_type"] for l in horizons}
    week_types = {l: tuple(w["workout_type"] for w in horizons[l]["weekly_plan"]) for l in horizons}
    week_tss   = {l: tuple(w["target_tss"]    for w in horizons[l]["weekly_plan"]) for l in horizons}
    print(f"Day-0 workout types : {day0_types}")
    distinct_day0 = len(set(day0_types.values()))
    distinct_weeks_types = len(set(week_types.values()))
    distinct_weeks_tss   = len(set(week_tss.values()))
    print(f"# distinct Day-0 types     : {distinct_day0}/3")
    print(f"# distinct week patterns   : {distinct_weeks_types}/3")
    print(f"# distinct weekly TSS plans: {distinct_weeks_tss}/3")
    if distinct_weeks_tss == 3:
        print("✓ All three horizons produce different weekly TSS plans")
    else:
        print("⚠ Horizons collapse on weekly TSS — model may need more training")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
