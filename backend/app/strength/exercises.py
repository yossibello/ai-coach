"""
Cyclist-specific exercise library.

Every exercise targets one or more of the five weaknesses cyclists accumulate
from spending thousands of hours in the saddle:
  1. Weak glutes (hip extension dominated by quads on the bike)
  2. Tight/weak hip flexors (permanently shortened in riding position)
  3. Weak posterior chain (hamstrings, lower back)
  4. Poor core stability (power leaks at the pelvis)
  5. Rounded upper back / weak rear delts (aero position)

Exercises are equipment-tagged so the scheduler can respect what the user has.
Equipment levels:
  "none"       — bodyweight only
  "band"       — resistance band (~$10)
  "kettlebell" — one kettlebell
  "dumbbell"   — pair of dumbbells
  "barbell"    — barbell + rack (gym)
  "box"        — plyo box or sturdy step
  "bench"      — bench or chair
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

Equipment = Literal["none", "band", "kettlebell", "dumbbell", "barbell", "box", "bench"]
MuscleGroup = Literal[
    "quads", "glutes", "hamstrings", "hip_flexors", "adductors",
    "core", "lower_back", "upper_back", "rear_delts", "calves",
]


@dataclass
class Exercise:
    name: str
    equipment: list[Equipment]
    muscles_primary: list[MuscleGroup]
    muscles_secondary: list[MuscleGroup]
    cycling_benefit: str
    description: str
    cue: str
    # Default set/rep scheme — overridden per approach phase
    default_sets: int = 3
    default_reps: str = "8-10"
    default_rest_sec: int = 90


# ── Lower body — squat pattern ────────────────────────────────────────────────

GOBLET_SQUAT = Exercise(
    name="Goblet Squat",
    equipment=["kettlebell", "dumbbell"],
    muscles_primary=["quads", "glutes"],
    muscles_secondary=["core", "adductors"],
    cycling_benefit="Builds quad and glute strength for climbing and sustained power output.",
    description="Hold a kettlebell or dumbbell at chest height, squat deep keeping "
                "chest tall and knees tracking over toes.",
    cue="Elbows inside your knees at the bottom — use them to push knees out.",
    default_sets=3,
    default_reps="8-10",
    default_rest_sec=90,
)

SPLIT_SQUAT = Exercise(
    name="Split Squat",
    equipment=["none", "dumbbell", "kettlebell"],
    muscles_primary=["quads", "glutes"],
    muscles_secondary=["hip_flexors", "adductors"],
    cycling_benefit="Corrects left/right imbalances — critical because cycling is bilaterally symmetric "
                    "but real power output rarely is.",
    description="Rear foot elevated on a bench or step (Bulgarian Split Squat). "
                "Lower back knee toward the floor, front shin stays vertical.",
    cue="Think 'down' not 'forward' — drop your hips straight down.",
    default_sets=3,
    default_reps="8 each leg",
    default_rest_sec=90,
)

STEP_UP = Exercise(
    name="Step-Up",
    equipment=["box", "bench"],
    muscles_primary=["quads", "glutes"],
    muscles_secondary=["hamstrings", "calves"],
    cycling_benefit="Direct transfer to pedalling mechanics — the movement pattern closely "
                    "mirrors the downstroke.",
    description="Step onto a box (knee height), drive through the heel to stand fully, "
                "lower back slowly. Add dumbbells to increase load.",
    cue="Push through the heel of the working leg — don't push off the back foot.",
    default_sets=3,
    default_reps="10 each leg",
    default_rest_sec=60,
)

# ── Lower body — hip hinge ────────────────────────────────────────────────────

KB_DEADLIFT = Exercise(
    name="Kettlebell Deadlift",
    equipment=["kettlebell", "dumbbell"],
    muscles_primary=["glutes", "hamstrings"],
    muscles_secondary=["lower_back", "core"],
    cycling_benefit="Posterior chain strength — the glutes and hamstrings are massively "
                    "underused in cycling; this wakes them up.",
    description="KB between feet, hinge at hips keeping back flat, stand tall by "
                "driving hips forward. Not a squat — the hips start higher than the knees.",
    cue="Screw your feet into the floor as you stand — creates hip external rotation and glute activation.",
    default_sets=3,
    default_reps="8-10",
    default_rest_sec=90,
)

ROMANIAN_DEADLIFT = Exercise(
    name="Romanian Deadlift",
    equipment=["dumbbell", "barbell", "kettlebell"],
    muscles_primary=["hamstrings", "glutes"],
    muscles_secondary=["lower_back"],
    cycling_benefit="Hamstring length and strength — cyclists chronically shorten their "
                    "hamstrings; this lengthens them under load.",
    description="Hinge forward pushing hips back, weights slide down your legs. "
                "Feel a stretch in the hamstrings, not rounding in the lower back.",
    cue="Push your hips to the wall behind you — you should feel the stretch before the weights reach mid-shin.",
    default_sets=3,
    default_reps="8-10",
    default_rest_sec=90,
)

KB_SWING = Exercise(
    name="Kettlebell Swing",
    equipment=["kettlebell"],
    muscles_primary=["glutes", "hamstrings"],
    muscles_secondary=["core", "upper_back"],
    cycling_benefit="Explosive posterior chain power — the hip-snap recruits fast-twitch "
                    "glute fibres that pure cycling never touches. Key for sprint finish and "
                    "punchy accelerations.",
    description="Hinge back loading the hips, then explosively snap the hips forward "
                "driving the KB to shoulder height. The power comes from the hips — "
                "arms are just a rope.",
    cue="Squeeze your glutes so hard at the top that you could crack a walnut.",
    default_sets=3,
    default_reps="10-15",
    default_rest_sec=60,
)

SINGLE_LEG_DEADLIFT = Exercise(
    name="Single-Leg Deadlift",
    equipment=["kettlebell", "dumbbell"],
    muscles_primary=["hamstrings", "glutes"],
    muscles_secondary=["core", "lower_back"],
    cycling_benefit="Unilateral hamstring and balance — exposes and fixes the imbalances "
                    "that cause knee tracking issues on the bike.",
    description="Hinge forward on one leg, opposite leg extends behind as a counterbalance. "
                "Keep hips square to the floor.",
    cue="Imagine your back leg and torso are a see-saw — they move together.",
    default_sets=3,
    default_reps="8 each leg",
    default_rest_sec=90,
)

# ── Core — anti-extension ─────────────────────────────────────────────────────

DEAD_BUG = Exercise(
    name="Dead Bug",
    equipment=["none"],
    muscles_primary=["core"],
    muscles_secondary=["hip_flexors"],
    cycling_benefit="Deep core stability — teaches the spine to stay neutral while limbs "
                    "move, which is exactly what you need to transfer power through the pelvis on the bike.",
    description="Lying on back, arms up, knees at 90°. Slowly lower opposite arm+leg "
                "toward the floor while keeping lower back pressed flat. Return and repeat.",
    cue="If your lower back lifts off the floor even slightly, you've gone too far — reduce range.",
    default_sets=3,
    default_reps="8 each side",
    default_rest_sec=45,
)

PLANK = Exercise(
    name="Plank",
    equipment=["none"],
    muscles_primary=["core"],
    muscles_secondary=["upper_back", "glutes"],
    cycling_benefit="Isometric core endurance — the position you hold on the bike for hours.",
    description="Forearms on floor, body in a straight line from head to heels. "
                "Do NOT let hips sag or pike up.",
    cue="Imagine pulling your elbows toward your feet — this activates the entire anterior core.",
    default_sets=3,
    default_reps="30-45 sec",
    default_rest_sec=45,
)

BIRD_DOG = Exercise(
    name="Bird Dog",
    equipment=["none"],
    muscles_primary=["core", "lower_back"],
    muscles_secondary=["glutes"],
    cycling_benefit="Lumbar stability and glute activation — trains the back extensors "
                    "to work with the glutes, which is the key power-transfer mechanism in the saddle.",
    description="On hands and knees, extend opposite arm and leg simultaneously while "
                "keeping hips level and spine neutral.",
    cue="Think 'long' not 'high' — reach out, don't lift up.",
    default_sets=3,
    default_reps="10 each side",
    default_rest_sec=45,
)

# ── Core — hip flexors / adductors ───────────────────────────────────────────

COPENHAGEN_PLANK = Exercise(
    name="Copenhagen Plank",
    equipment=["bench"],
    muscles_primary=["adductors", "hip_flexors"],
    muscles_secondary=["core"],
    cycling_benefit="The most neglected muscle group in cycling. Strong adductors keep "
                    "the knee tracking straight and prevent the wobble that bleeds power.",
    description="Side plank with top foot on a bench. Bottom leg hangs or rests on the "
                "bench for full vs. modified version. Hold, or add reps by lowering bottom hip.",
    cue="Drive your top knee INTO the bench — you'll feel the adductors fire immediately.",
    default_sets=3,
    default_reps="20-30 sec each side",
    default_rest_sec=45,
)

STANDING_HIP_FLEXION = Exercise(
    name="Standing Hip Flexion",
    equipment=["band"],
    muscles_primary=["hip_flexors"],
    muscles_secondary=["core"],
    cycling_benefit="Cyclists have the strongest hip flexors in any sport AND the tightest. "
                    "This builds active hip flexion strength through full range — the upstroke.",
    description="Anchor a band behind you at ankle height. Facing away, drive the knee "
                "up to hip height against the band resistance. Controlled return.",
    cue="Don't let your lower back arch as the knee comes up — brace the core first.",
    default_sets=3,
    default_reps="12 each leg",
    default_rest_sec=45,
)

# ── Upper body — posture / rear delts ────────────────────────────────────────

BAND_PULL_APART = Exercise(
    name="Band Pull-Apart",
    equipment=["band"],
    muscles_primary=["rear_delts", "upper_back"],
    muscles_secondary=[],
    cycling_benefit="Counteracts the internally rotated, rounded-shoulder posture from "
                    "the drops. Keeps the shoulder healthy and the upper back from fatiguing "
                    "on long rides.",
    description="Hold a band at arm's length in front of you. Pull it apart to your sides "
                "by squeezing the shoulder blades together. Return slowly.",
    cue="Thumbs rotate out (supinate) as you pull — this hits the rear delt harder.",
    default_sets=3,
    default_reps="15-20",
    default_rest_sec=30,
)

FACE_PULL = Exercise(
    name="Face Pull",
    equipment=["band"],
    muscles_primary=["rear_delts", "upper_back"],
    muscles_secondary=[],
    cycling_benefit="External shoulder rotation strength — the direct antidote to the "
                    "internal rotation cycling locks you into.",
    description="Anchor a band at face height. Pull the band toward your face, elbows "
                "high and wide, hands finishing beside your ears.",
    cue="Finish with your hands BEHIND your ears, not beside them — that's external rotation.",
    default_sets=3,
    default_reps="15",
    default_rest_sec=30,
)

DB_ROW = Exercise(
    name="Dumbbell Row",
    equipment=["dumbbell"],
    muscles_primary=["upper_back"],
    muscles_secondary=["rear_delts"],
    cycling_benefit="Upper back pulling strength prevents the collapse into the bars that "
                    "costs power and causes neck pain on long efforts.",
    description="One knee and hand on a bench for support. Row the dumbbell toward your "
                "hip, elbow close to the body, full range of motion.",
    cue="Lead with the elbow, not the hand — think 'put your elbow in your back pocket'.",
    default_sets=3,
    default_reps="10 each side",
    default_rest_sec=60,
)

# ── Power / plyometric ────────────────────────────────────────────────────────

BOX_JUMP = Exercise(
    name="Box Jump",
    equipment=["box"],
    muscles_primary=["quads", "glutes"],
    muscles_secondary=["calves", "core"],
    cycling_benefit="Fast-twitch recruitment — trains the explosive neuromuscular patterns "
                    "used in sprint finishes and out-of-saddle accelerations.",
    description="Stand in front of a box, dip and jump landing softly with knees bent. "
                "Step down (never jump down). Start with a low box.",
    cue="Land as quietly as possible — if you're loud, you're absorbing force poorly.",
    default_sets=3,
    default_reps="5",
    default_rest_sec=120,
)

SINGLE_LEG_HOP = Exercise(
    name="Single-Leg Hop",
    equipment=["none"],
    muscles_primary=["quads", "glutes", "calves"],
    muscles_secondary=["core"],
    cycling_benefit="Unilateral explosive power — develops the neuromuscular coordination "
                    "that makes each pedal stroke powerful, not just the average of both legs.",
    description="Hop forward on one leg, land softly absorbing with the whole leg. "
                "3-5 hops per set, each leg. Progress to lateral hops.",
    cue="Soft landing — catch the floor, don't slap it.",
    default_sets=3,
    default_reps="5 each leg",
    default_rest_sec=90,
)

# ── Accessory / corrective ────────────────────────────────────────────────────

GLUTE_BRIDGE = Exercise(
    name="Glute Bridge",
    equipment=["none", "band"],
    muscles_primary=["glutes"],
    muscles_secondary=["hamstrings", "core"],
    cycling_benefit="Wakes up the glutes that chronic sitting and riding switches off. "
                    "Essential prep before any lower body session or ride.",
    description="Lying on back, feet flat. Drive hips to the ceiling squeezing glutes "
                "hard at the top. Hold 2 seconds. Add a band around knees to increase "
                "glute medius activation.",
    cue="At the top your body forms a straight line from knee to shoulder — no hyperextension.",
    default_sets=3,
    default_reps="15",
    default_rest_sec=45,
)

CALF_RAISE = Exercise(
    name="Single-Leg Calf Raise",
    equipment=["none", "dumbbell"],
    muscles_primary=["calves"],
    muscles_secondary=[],
    cycling_benefit="Calf strength improves ankle stiffness on the pedal — "
                    "stiff ankles transfer more power, especially at high cadence.",
    description="Stand on the edge of a step on one foot. Lower heel below step level "
                "then rise as high as possible. Use a wall for balance only.",
    cue="Full range — all the way down and all the way up. Half reps train nothing.",
    default_sets=3,
    default_reps="15 each leg",
    default_rest_sec=45,
)

PUSH_UP = Exercise(
    name="Push-Up",
    equipment=["none"],
    muscles_primary=["upper_back"],
    muscles_secondary=["core", "rear_delts"],
    cycling_benefit="Pressing strength for out-of-saddle climbing and sprinting — "
                    "you push against the bars as hard as you push the pedals.",
    description="Standard push-up with hands slightly wider than shoulders. Body in a "
                "straight line from head to heels. Full range.",
    cue="At the top, push the floor away and let your upper back round slightly — serratus anterior activation.",
    default_sets=3,
    default_reps="10-15",
    default_rest_sec=45,
)

FARMER_CARRY = Exercise(
    name="Farmer Carry",
    equipment=["kettlebell", "dumbbell"],
    muscles_primary=["core", "upper_back"],
    muscles_secondary=["glutes", "calves"],
    cycling_benefit="Full-body stability and grip — builds the anti-lateral-flexion "
                    "core strength that keeps you efficient on the bike over long hours.",
    description="Pick up heavy weights in each hand, stand tall, and walk for 20-30 "
                "metres. Keep shoulders back and down.",
    cue="Imagine someone is trying to pull your shoulder toward the weight — resist them.",
    default_sets=3,
    default_reps="20-30 m",
    default_rest_sec=60,
)

# ── Full library ──────────────────────────────────────────────────────────────

ALL_EXERCISES: dict[str, Exercise] = {
    "goblet_squat":         GOBLET_SQUAT,
    "split_squat":          SPLIT_SQUAT,
    "step_up":              STEP_UP,
    "kb_deadlift":          KB_DEADLIFT,
    "romanian_deadlift":    ROMANIAN_DEADLIFT,
    "kb_swing":             KB_SWING,
    "single_leg_deadlift":  SINGLE_LEG_DEADLIFT,
    "dead_bug":             DEAD_BUG,
    "plank":                PLANK,
    "bird_dog":             BIRD_DOG,
    "copenhagen_plank":     COPENHAGEN_PLANK,
    "standing_hip_flexion": STANDING_HIP_FLEXION,
    "band_pull_apart":      BAND_PULL_APART,
    "face_pull":            FACE_PULL,
    "db_row":               DB_ROW,
    "box_jump":             BOX_JUMP,
    "single_leg_hop":       SINGLE_LEG_HOP,
    "glute_bridge":         GLUTE_BRIDGE,
    "calf_raise":           CALF_RAISE,
    "push_up":              PUSH_UP,
    "farmer_carry":         FARMER_CARRY,
}
