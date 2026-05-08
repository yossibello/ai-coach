"""
Export activities from PostgreSQL into a flat parquet file for ML training.

Usage:
  python -m ml.data.export_to_parquet \
    --db postgresql://aicoach:secret@localhost/aicoach \
    --output ./data/activities.parquet
"""
from __future__ import annotations

import argparse
import pandas as pd
import sqlalchemy as sa

QUERY = """
SELECT
    a.user_id              AS athlete_id,
    a.date,
    a.duration_seconds,
    a.distance_meters,
    a.elevation_gain_meters,
    a.avg_power,
    a.max_power,
    a.normalized_power,
    a.intensity_factor,
    a.tss,
    a.variability_index,
    a.avg_hr,
    a.max_hr,
    a.hr_drift,
    a.aerobic_efficiency,
    a.avg_cadence,
    a.temperature_c,
    a.humidity_pct,
    a.wind_speed_kmh,
    a.workout_type,
    a.perceived_exertion,
    -- time-in-zones (JSON)
    (a.time_in_zones->>'z1_recovery')::float     AS z1,
    (a.time_in_zones->>'z2_endurance')::float    AS z2,
    (a.time_in_zones->>'z3_tempo')::float        AS z3,
    (a.time_in_zones->>'z4_threshold')::float    AS z4,
    (a.time_in_zones->>'z5_vo2max')::float       AS z5,
    (a.time_in_zones->>'z6_anaerobic')::float    AS z6,
    (a.time_in_zones->>'z7_neuromuscular')::float AS z7,
    -- profile
    p.age,
    p.weight_kg,
    p.height_cm,
    p.sex,
    p.ftp,
    p.max_hr                                     AS profile_max_hr,
    p.resting_hr,
    p.cycling_experience_years,
    p.primary_goal
FROM activities a
LEFT JOIN athlete_profiles p ON p.user_id = a.user_id
ORDER BY a.user_id, a.date
"""


def export(db_url: str, output: str):
    engine = sa.create_engine(db_url)
    print("Querying database…")
    df = pd.read_sql(QUERY, engine)
    print(f"Rows: {len(df):,}")

    # Fill NaN zones with 0
    for col in ["z1","z2","z3","z4","z5","z6","z7"]:
        df[col] = df[col].fillna(0)

    df.to_parquet(output, index=False)
    print(f"Saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",     required=True)
    parser.add_argument("--output", default="./data/activities.parquet")
    args = parser.parse_args()
    export(args.db, args.output)
