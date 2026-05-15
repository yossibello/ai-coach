"""
Strength training approach definitions.

Each approach is a named strategy with:
  - How many sessions per week per cycling phase
  - What exercises to do in each session
  - How sets/reps change across phases

Three approaches:

  friel
      Joe Friel's periodized gym programme from The Cyclist's Training Bible.
      Three phases: AA (Anatomical Adaptation) → MS (Max Strength) → SM (Maintenance).
      Phase is driven by the cycling periodization block, not a separate calendar.
      Designed for athletes with access to a gym or at least dumbbells/KB.

  minimum_dose
      Dan John / S&C minimalism: 5 movement patterns, 2-3 sessions/week, same
      session every time. No periodization — just consistent, enough.
      Works with a single kettlebell. Good for time-crunched riders.

  grease_the_groove
      Pavel Tsatsouline's GTG: sub-maximal sets of 1-3 exercises spread throughout
      the day, never to failure. Pure neural adaptation, no soreness, no fatigue cost.
      Bodyweight + one kettlebell. The lightest touch on recovery.

Adding a new approach:
  1. Create a class that inherits StrengthApproach
  2. Implement sessions_per_week() and get_sessions()
  3. Add it to APPROACHES dict at the bottom
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.strength.exercises import (
    GOBLET_SQUAT, SPLIT_SQUAT, STEP_UP,
    KB_DEADLIFT, ROMANIAN_DEADLIFT, KB_SWING, SINGLE_LEG_DEADLIFT,
    DEAD_BUG, PLANK, BIRD_DOG, COPENHAGEN_PLANK, STANDING_HIP_FLEXION,
    BAND_PULL_APART, FACE_PULL, DB_ROW,
    BOX_JUMP, SINGLE_LEG_HOP,
    GLUTE_BRIDGE, CALF_RAISE, PUSH_UP, FARMER_CARRY,
    Exercise,
)


@dataclass
class ExerciseSlot:
    exercise: Exercise
    sets: int
    reps: str
    rest_sec: int
    weight_guidance: str = ""

    def to_dict(self) -> dict:
        return {
            "name":            self.exercise.name,
            "sets":            self.sets,
            "reps":            self.reps,
            "rest_sec":        self.rest_sec,
            "weight_guidance": self.weight_guidance,
            "equipment":       self.exercise.equipment,
            "muscles_primary": self.exercise.muscles_primary,
            "cycling_benefit": self.exercise.cycling_benefit,
            "description":     self.exercise.description,
            "cue":             self.exercise.cue,
        }


@dataclass
class StrengthSession:
    name: str
    phase_label: str
    duration_minutes: int
    notes: str
    exercises: list[ExerciseSlot]

    def to_dict(self) -> dict:
        return {
            "name":             self.name,
            "phase_label":      self.phase_label,
            "duration_minutes": self.duration_minutes,
            "notes":            self.notes,
            "exercises":        [e.to_dict() for e in self.exercises],
        }


# ── Friel periodized approach ─────────────────────────────────────────────────

class FrielApproach:
    """
    Friel's 3-phase gym programme tied to the cycling block.

    base_build  → AA (Anatomical Adaptation): high reps, light load, 2-3x/week
    build       → MS (Max Strength): low reps, heavy load, 2x/week
    peak/race   → SM (Strength Maintenance): 1x/week, just enough to retain
    recovery    → 0 sessions (full rest)
    """

    label = "Friel Periodized"
    description = (
        "Joe Friel's classic programme from The Cyclist's Training Bible. "
        "Strength phases are matched to your cycling block: high-rep anatomical "
        "adaptation in base, heavy compound lifts in build, then a single "
        "maintenance session per week through race season. Requires dumbbells "
        "or kettlebells; a barbell is optional."
    )

    def sessions_per_week(self, phase: str) -> int:
        return {"base_build": 2, "build": 2, "peak": 1, "recovery_week": 0}.get(phase, 1)

    def get_sessions(self, phase: str) -> list[StrengthSession]:
        if phase == "recovery_week":
            return []
        if phase == "base_build":
            return [self._aa_session_a(), self._aa_session_b()]
        if phase == "build":
            return [self._ms_session_a(), self._ms_session_b()]
        # peak / default → maintenance
        return [self._sm_session()]

    # ── AA: Anatomical Adaptation ─────────────────────────────────────────────

    def _aa_session_a(self) -> StrengthSession:
        return StrengthSession(
            name="AA Session A",
            phase_label="Anatomical Adaptation",
            duration_minutes=40,
            notes=(
                "Light weight, high reps, full range of motion. "
                "This phase builds connective tissue and prepares joints for heavier work ahead. "
                "Never go to failure — stop 3 reps short."
            ),
            exercises=[
                ExerciseSlot(GOBLET_SQUAT,    sets=3, reps="20-25", rest_sec=60,
                             weight_guidance="Light — you could do 5 more reps at the end of each set."),
                ExerciseSlot(ROMANIAN_DEADLIFT, sets=3, reps="20", rest_sec=60,
                             weight_guidance="Light dumbbells or KB. Feel the hamstring stretch, not strain."),
                ExerciseSlot(GLUTE_BRIDGE,    sets=3, reps="20",   rest_sec=45,
                             weight_guidance="Bodyweight only. Squeeze hard at the top for 2 seconds."),
                ExerciseSlot(DEAD_BUG,        sets=3, reps="10 each side", rest_sec=30, weight_guidance=""),
                ExerciseSlot(BAND_PULL_APART, sets=3, reps="20",   rest_sec=30,
                             weight_guidance="Light band — focus on squeezing shoulder blades together."),
                ExerciseSlot(CALF_RAISE,      sets=3, reps="20 each", rest_sec=30, weight_guidance="Bodyweight."),
            ],
        )

    def _aa_session_b(self) -> StrengthSession:
        return StrengthSession(
            name="AA Session B",
            phase_label="Anatomical Adaptation",
            duration_minutes=40,
            notes=(
                "Focus on hip flexors, unilateral work, and upper back. "
                "These muscles are neglected on the bike and need the most attention."
            ),
            exercises=[
                ExerciseSlot(SPLIT_SQUAT,          sets=3, reps="15 each", rest_sec=60,
                             weight_guidance="Bodyweight or very light. Master the pattern first."),
                ExerciseSlot(KB_SWING,             sets=3, reps="15",      rest_sec=60,
                             weight_guidance="Light KB — learn the hip hinge. Power from hips, not arms."),
                ExerciseSlot(STANDING_HIP_FLEXION, sets=3, reps="15 each", rest_sec=30,
                             weight_guidance="Light band. Slow and controlled."),
                ExerciseSlot(BIRD_DOG,             sets=3, reps="10 each", rest_sec=30, weight_guidance=""),
                ExerciseSlot(FACE_PULL,            sets=3, reps="20",      rest_sec=30,
                             weight_guidance="Light band."),
                ExerciseSlot(COPENHAGEN_PLANK,     sets=3, reps="15-20 sec each", rest_sec=30, weight_guidance=""),
            ],
        )

    # ── MS: Maximum Strength ──────────────────────────────────────────────────

    def _ms_session_a(self) -> StrengthSession:
        return StrengthSession(
            name="MS Session A",
            phase_label="Maximum Strength",
            duration_minutes=50,
            notes=(
                "Heavy compound work. 3-5 reps at ~80-85% of your max. "
                "Full rest between sets — the goal is maximal force production, not cardio. "
                "Do this session on a rest day or after an easy ride, never before a hard ride."
            ),
            exercises=[
                ExerciseSlot(GOBLET_SQUAT,    sets=4, reps="4-5", rest_sec=180,
                             weight_guidance="Heavy — the last rep should be a real effort. "
                                             "Add weight each session if form is solid."),
                ExerciseSlot(ROMANIAN_DEADLIFT, sets=4, reps="5", rest_sec=180,
                             weight_guidance="Heavy. Hinge deep, feel the hamstrings load, then drive."),
                ExerciseSlot(STEP_UP,         sets=3, reps="6 each", rest_sec=120,
                             weight_guidance="Add dumbbells. Step height should challenge hip extension."),
                ExerciseSlot(DEAD_BUG,        sets=3, reps="8 each", rest_sec=45, weight_guidance=""),
                ExerciseSlot(DB_ROW,          sets=3, reps="8 each", rest_sec=90,
                             weight_guidance="Moderately heavy. Full range."),
            ],
        )

    def _ms_session_b(self) -> StrengthSession:
        return StrengthSession(
            name="MS Session B",
            phase_label="Maximum Strength",
            duration_minutes=50,
            notes=(
                "Posterior chain and explosive focus. KB Swing develops the fast-twitch "
                "glute power that cycling alone never builds. The plyometric at the end "
                "is low volume — quality over quantity."
            ),
            exercises=[
                ExerciseSlot(SPLIT_SQUAT,      sets=4, reps="5 each", rest_sec=180,
                             weight_guidance="Add dumbbells or KB. This should be challenging."),
                ExerciseSlot(KB_SWING,         sets=4, reps="8",      rest_sec=120,
                             weight_guidance="Heavy KB — you should be breathing hard after each set."),
                ExerciseSlot(SINGLE_LEG_DEADLIFT, sets=3, reps="6 each", rest_sec=120,
                             weight_guidance="KB or dumbbell. Go slow on the way down."),
                ExerciseSlot(COPENHAGEN_PLANK, sets=3, reps="25-30 sec each", rest_sec=45, weight_guidance=""),
                ExerciseSlot(BOX_JUMP,         sets=3, reps="4",      rest_sec=120,
                             weight_guidance="Bodyweight only. Land softly. Full recovery between sets."),
            ],
        )

    # ── SM: Strength Maintenance ──────────────────────────────────────────────

    def _sm_session(self) -> StrengthSession:
        return StrengthSession(
            name="Maintenance Session",
            phase_label="Strength Maintenance",
            duration_minutes=30,
            notes=(
                "One session per week keeps everything you built. "
                "Moderate weight, moderate reps — just enough stimulus to prevent regression. "
                "Keep it under 35 minutes; your cycling sessions are the priority now."
            ),
            exercises=[
                ExerciseSlot(GOBLET_SQUAT,    sets=2, reps="8",      rest_sec=120,
                             weight_guidance="Moderate-heavy. Two hard sets is all you need."),
                ExerciseSlot(KB_SWING,        sets=2, reps="10",     rest_sec=90,
                             weight_guidance="Heavy KB. Explosive and crisp."),
                ExerciseSlot(DEAD_BUG,        sets=2, reps="8 each", rest_sec=30, weight_guidance=""),
                ExerciseSlot(BAND_PULL_APART, sets=2, reps="15",     rest_sec=30,
                             weight_guidance="Light band. Shoulder health work."),
            ],
        )


# ── Minimum Dose approach ─────────────────────────────────────────────────────

class MinimumDoseApproach:
    """
    Dan John-inspired minimum effective dose: 5 fundamental movement patterns,
    same session every time, 2-3x/week. No periodization — just consistency.

    Pattern:  Hinge · Squat · Push · Pull · Core
    Duration: ~30 minutes
    Equipment: one kettlebell + resistance band
    """

    label = "Minimum Dose"
    description = (
        "5 movement patterns, same session every time, 2-3× per week. "
        "No periodization, no complexity. Based on Dan John's 'Never Let Go' philosophy: "
        "the minimum that actually works done consistently beats the perfect programme "
        "done occasionally. One kettlebell and a band are all you need."
    )

    def sessions_per_week(self, phase: str) -> int:
        return 0 if phase == "recovery_week" else 2

    def get_sessions(self, phase: str) -> list[StrengthSession]:
        if phase == "recovery_week":
            return []
        session = StrengthSession(
            name="The 5-Pattern Session",
            phase_label="Minimum Dose",
            duration_minutes=30,
            notes=(
                "Hit all 5 patterns every time. Weight selection: you should finish "
                "each set feeling like you could do 2-3 more reps. Never go to failure. "
                "If you only have 20 minutes, do 2 sets instead of 3."
            ),
            exercises=[
                ExerciseSlot(KB_SWING,        sets=3, reps="10",     rest_sec=60,
                             weight_guidance="Moderate KB. Explosive snap at the top."),
                ExerciseSlot(GOBLET_SQUAT,    sets=3, reps="8",      rest_sec=60,
                             weight_guidance="Moderate KB or DB. Chest tall, deep squat."),
                ExerciseSlot(PUSH_UP,         sets=3, reps="10",     rest_sec=45,
                             weight_guidance="Bodyweight. Full range. Elevate hands if needed."),
                ExerciseSlot(BAND_PULL_APART, sets=3, reps="15",     rest_sec=30,
                             weight_guidance="Light-medium band."),
                ExerciseSlot(DEAD_BUG,        sets=3, reps="8 each", rest_sec=30, weight_guidance=""),
            ],
        )
        return [session, session]  # same session both days

    def get_sessions(self, phase: str) -> list[StrengthSession]:  # type: ignore[no-redef]
        if phase == "recovery_week":
            return []
        session = self._the_session()
        return [session, session]

    def _the_session(self) -> StrengthSession:
        return StrengthSession(
            name="The 5-Pattern Session",
            phase_label="Minimum Dose",
            duration_minutes=30,
            notes=(
                "Hit all 5 patterns every time. Weight: finish each set feeling like "
                "you could do 2-3 more reps. If you only have 20 minutes, do 2 sets. "
                "Do this consistently 2× per week and it works."
            ),
            exercises=[
                ExerciseSlot(KB_SWING,        sets=3, reps="10",     rest_sec=60,
                             weight_guidance="Moderate KB. Explosive hip snap."),
                ExerciseSlot(GOBLET_SQUAT,    sets=3, reps="8",      rest_sec=60,
                             weight_guidance="Moderate KB or DB. Deep squat, chest tall."),
                ExerciseSlot(PUSH_UP,         sets=3, reps="10",     rest_sec=45,
                             weight_guidance="Bodyweight. Add a plate on your back to progress."),
                ExerciseSlot(BAND_PULL_APART, sets=3, reps="15",     rest_sec=30,
                             weight_guidance="Light-medium band."),
                ExerciseSlot(DEAD_BUG,        sets=3, reps="8 each", rest_sec=30,
                             weight_guidance=""),
            ],
        )


# ── Grease the Groove approach ────────────────────────────────────────────────

class GreaseTheGrooveApproach:
    """
    Pavel Tsatsouline's GTG: sub-maximal sets spread throughout the day.

    The rule: pick 1-2 exercises, do 3-5 reps every 1-2 hours, NEVER go above
    50-60% of max effort, never go to failure. Total daily volume is high but
    each mini-set is easy. Builds strength through pure neural adaptation —
    no soreness, no DOMS, no fatigue cost on the bike.

    Best for: athletes who don't want gym sessions, recovery-sensitive athletes,
    anyone who just wants stronger glutes and a healthier posterior chain.
    """

    label = "Grease the Groove"
    description = (
        "Pavel Tsatsouline's GTG method: 5-10 micro-sets of 3-5 reps spread "
        "throughout the day, every day. Never hard, never sore. "
        "Pure neural adaptation — you get stronger without ever 'working out'. "
        "A kettlebell swing and a goblet squat are all you need. "
        "Zero fatigue cost on your cycling."
    )

    def sessions_per_week(self, phase: str) -> int:
        # GTG runs every day — it's presented as daily 'active' sessions
        return 0 if phase == "recovery_week" else 5

    def get_sessions(self, phase: str) -> list[StrengthSession]:
        if phase == "recovery_week":
            return []
        session = StrengthSession(
            name="GTG Daily Practice",
            phase_label="Grease the Groove",
            duration_minutes=5,
            notes=(
                "NOT a workout session — a daily practice. "
                "Do 3-5 reps of each exercise every 1-2 hours throughout the day. "
                "Set a phone reminder. Total daily volume: ~40-60 swings, ~30-50 squats. "
                "Key rule: NEVER go above 60% effort. If you feel tired, do fewer reps. "
                "The goal is perfect reps, not hard reps."
            ),
            exercises=[
                ExerciseSlot(KB_SWING,     sets=1, reps="5",   rest_sec=0,
                             weight_guidance="Moderate KB — comfortable, fast, crisp. "
                                             "Stop before you slow down."),
                ExerciseSlot(GOBLET_SQUAT, sets=1, reps="3-5", rest_sec=0,
                             weight_guidance="Light-moderate KB. Deep and slow. "
                                             "Never to failure."),
            ],
        )
        return [session] * 5  # 5 'active' days


# ── Registry ──────────────────────────────────────────────────────────────────

APPROACHES: dict[str, FrielApproach | MinimumDoseApproach | GreaseTheGrooveApproach] = {
    "friel":              FrielApproach(),
    "minimum_dose":       MinimumDoseApproach(),
    "grease_the_groove":  GreaseTheGrooveApproach(),
}

APPROACH_LABELS: dict[str, str] = {k: v.label for k, v in APPROACHES.items()}
APPROACH_DESCRIPTIONS: dict[str, str] = {k: v.description for k, v in APPROACHES.items()}
