"""One-shot script: recompute TSS for existing activities + rebuild PMC."""
import asyncio
import sys
sys.path.insert(0, "backend")

from app.core.database import AsyncSessionLocal
from app.models.activity import Activity
from app.models.user import User, AthleteProfile
from app.models.recommendation import FitnessMetric
from app.services.metrics_service import compute_pmc_for_user
from sqlalchemy import select, update

async def main():
    async with AsyncSessionLocal() as db:
        # Fetch user+profile
        result = await db.execute(
            select(User).where(User.email == "yossi.bello@gmail.com")
        )
        user = result.scalar_one()
        profile_result = await db.execute(
            select(AthleteProfile).where(AthleteProfile.user_id == user.id)
        )
        profile = profile_result.scalar_one_or_none()
        ftp = (profile.ftp if profile and profile.ftp else 200)
        print(f"User: {user.email}, FTP: {ftp}")

        # Fetch all activities missing TSS but having avg_power
        acts_result = await db.execute(
            select(Activity).where(
                Activity.user_id == user.id,
                Activity.tss == None,
                Activity.avg_power != None,
            )
        )
        activities = acts_result.scalars().all()
        print(f"Activities missing TSS with avg_power: {len(activities)}")

        updated = 0
        for act in activities:
            if act.avg_power and act.duration_seconds:
                if_ = act.avg_power / ftp
                act.intensity_factor = round(if_, 3)
                act.tss = round((act.duration_seconds / 3600) * (if_ ** 2) * 100, 1)
                if not act.workout_type:
                    from app.services.metrics_service import _classify_workout
                    act.workout_type = _classify_workout(if_, act.duration_seconds)
                db.add(act)
                updated += 1

        await db.flush()
        print(f"Updated TSS for {updated} activities")

        # Rebuild PMC
        print("Rebuilding PMC (CTL/ATL/TSB)...")
        await compute_pmc_for_user(user.id, db)
        await db.commit()
        print("Done.")

        # Print current fitness stats
        from sqlalchemy import text
        r = await db.execute(text(
            "SELECT date, ctl, atl, tsb, ftp FROM fitness_metrics "
            "WHERE user_id=:uid ORDER BY date DESC LIMIT 3"
        ), {"uid": user.id})
        for row in r.fetchall():
            print(f"  {row[0][:10]}  CTL={row[1]:.1f}  ATL={row[2]:.1f}  TSB={row[3]:+.1f}  FTP={row[4]}")

asyncio.run(main())
