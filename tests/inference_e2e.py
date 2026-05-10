"""End-to-end inference test: inject synthetic activities, run recommendation."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_audit.db")
os.environ.setdefault("SECRET_KEY", "test-secret-do-not-use")

from datetime import date, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select

from app.core.database import Base
from app.models.user import User
from app.models.activity import Activity
from app.ml.inference import generate_recommendation, _load_model


async def main():
    engine = create_async_engine("sqlite+aiosqlite:///./test_audit.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        u = (await db.execute(select(User).limit(1))).scalar_one_or_none()
        if not u:
            print("No user found; run api_smoke first.")
            return
        print(f"User: {u.email}")
        existing = (await db.execute(select(Activity).where(Activity.user_id == u.id))).scalars().all()
        if len(existing) < 30:
            for i in range(90):
                db.add(Activity(
                    user_id=u.id,
                    source="manual",
                    external_id=f"syn-{i}",
                    date=date.today() - timedelta(days=90 - i),
                    duration_seconds=3600 + (i % 5) * 600,
                    distance_meters=30000.0,
                    avg_power=180.0 + (i % 10) * 5,
                    normalized_power=200.0,
                    avg_hr=140.0,
                    max_hr=170.0,
                    tss=70.0,
                    intensity_factor=0.75,
                    workout_type="endurance" if i % 3 else "sweetspot",
                    review_status="confirmed",
                ))
            await db.commit()
            print("Injected 90 synthetic activities")
        else:
            print(f"Already had {len(existing)} activities")

        m = _load_model()
        print(f"Model loaded: {m is not None}")
        rec = await generate_recommendation(u, db)
        nw = rec.payload["next_workout"]
        print(f"confidence={rec.confidence:.2f}  cold_start={rec.is_cold_start}  model={rec.model_version}")
        print(f"next_workout: {nw['workout_type']}  {nw['duration_minutes']} min  TSS={nw['target_tss']}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
