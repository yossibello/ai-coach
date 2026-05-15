"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, useMemo, useCallback } from "react";
import { coachAPI } from "@/lib/api";
import { Brain, RefreshCw, AlertTriangle, CheckCircle, Info, Zap, Calendar, Target, ThumbsUp, ThumbsDown, X, Dumbbell, ChevronDown, ChevronUp } from "lucide-react";
import { cn, formatDuration } from "@/lib/utils";
import type { CoachRecommendation, WorkoutPlan, CoachInsight, TrainingRisk, HorizonKey, HorizonPayload, Macrocycle, MacrocycleWeek } from "@/types";
import toast from "react-hot-toast";

function isStale(generatedAt: string | undefined): boolean {
  if (!generatedAt) return true;
  const recDate = new Date(generatedAt).toDateString();
  const today = new Date().toDateString();
  return recDate !== today;
}

export default function CoachPage() {
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["coach-recommendation"],
    queryFn: () => coachAPI.getRecommendation(),
  });

  // Multi-horizon: short / medium / event side-by-side. Always fresh.
  const { data: multi, isLoading: multiLoading } = useQuery({
    queryKey: ["coach-multi-horizon"],
    queryFn: () => coachAPI.getMultiHorizon(),
    staleTime: 5 * 60 * 1000, // 5 min
  });

  // Macrocycle: reverse-periodization plan. 404 silently if no goal_event_date.
  const { data: macro } = useQuery({
    queryKey: ["coach-macrocycle"],
    queryFn: () => coachAPI.getMacrocycle().catch(() => null),
    staleTime: 30 * 60 * 1000, // 30 min — changes only when plan/event changes
  });

  // User-selected horizon override — persisted in localStorage so a page
  // refresh doesn't reset it back to the backend's default.
  const [horizonOverride, setHorizonOverride] = useState<HorizonKey | null>(() => {
    try {
      const saved = localStorage.getItem("coach-horizon");
      if (saved === "short" || saved === "medium" || saved === "event") return saved;
    } catch {}
    return null;
  });
  const setHorizon = (h: HorizonKey) => {
    try { localStorage.setItem("coach-horizon", h); } catch {}
    setHorizonOverride(h);
  };
  const activeHorizon: HorizonKey | null = useMemo(() => {
    if (!multi) return null;
    if (horizonOverride && multi.horizons[horizonOverride]) return horizonOverride;
    return multi.active_horizon;
  }, [multi, horizonOverride]);
  const horizonPayload: HorizonPayload | undefined =
    multi && activeHorizon ? multi.horizons[activeHorizon] : undefined;

  const refresh = useMutation({
    mutationFn: () => coachAPI.refreshRecommendation(),
    onSuccess: (d) => {
      qc.setQueryData(["coach-recommendation"], d);
      qc.invalidateQueries({ queryKey: ["coach-multi-horizon"] });
      qc.invalidateQueries({ queryKey: ["coach-macrocycle"] });
      toast.success("Recommendation updated");
    },
  });

  // Auto-refresh if the stored recommendation is from a previous day
  useEffect(() => {
    if (!isLoading && isStale(data?.generated_at)) {
      refresh.mutate();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading]);

  if (isLoading) {
    return (
      <div className="p-6 flex items-center justify-center h-[80vh]">
        <div className="text-center space-y-3">
          <Brain className="w-10 h-10 text-brand-500 animate-pulse mx-auto" />
          <p className="text-slate-400 text-sm">AI Coach is thinking…</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="p-6 flex items-center justify-center h-[80vh]">
        <div className="text-center space-y-3 max-w-sm">
          <Brain className="w-10 h-10 text-slate-500 mx-auto" />
          <p className="text-white font-medium">No recommendation yet</p>
          <p className="text-slate-400 text-sm">
            Upload at least one ride to get your first AI coaching recommendation.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Brain className="w-6 h-6 text-brand-500" />
            AI Coach
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Confidence: {Math.round(data.confidence * 100)}%
            {data.is_cold_start && (
              <span className="ml-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
                Guided start — upload more rides to personalize
              </span>
            )}
          </p>
        </div>
        <button
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-surface-card border border-surface-border hover:border-brand-500/50 text-slate-300 hover:text-white text-sm rounded-xl transition-all"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", refresh.isPending && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Horizon picker (multi-horizon: short / medium / event) */}
      {multi && (
        <HorizonPicker
          multi={multi}
          activeHorizon={activeHorizon}
          onSelect={setHorizon}
          isUserOverride={horizonOverride !== null}
        />
      )}
      {multiLoading && !multi && (
        <div className="text-xs text-slate-500">Loading horizon options…</div>
      )}

      {/* Macrocycle: reverse-periodization plan to event date */}
      {macro && !macro.error && macro.weeks && macro.weeks.length > 0 && (
        <MacrocycleCard macro={macro} />
      )}

      {/* Risks */}
      {data.risks?.length > 0 && <RiskBanner risks={data.risks} />}

      {/* Insights */}
      {data.insights?.length > 0 && <InsightGrid insights={data.insights} />}

      {/* Weekly plan */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-brand-400" />
          {(() => {
            const plan = horizonPayload?.weekly_plan ?? data.weekly_plan;
            const n = plan?.length ?? 0;
            return `${n}-Day Training Plan`;
          })()}
          {horizonPayload && (
            <span className="text-xs text-slate-500 font-normal">
              · {horizonPayload.horizon_label}
            </span>
          )}
        </h2>
        <div
          className="overflow-x-auto -mx-4 px-4"
          style={{ scrollbarWidth: "thin", scrollbarColor: "#334155 transparent" }}
        >
          <div className="space-y-3 min-w-[560px]">
            {(horizonPayload?.weekly_plan ?? data.weekly_plan)?.map((w, i) => (
              // Day 0 always uses the authoritative standard rec (full safety
              // pipeline applied). Horizon tabs only differentiate days 1–6.
              <WorkoutCard
                key={i}
                plan={i === 0 && data.next_workout ? data.next_workout : w}
                highlight={i === 0}
                recId={i === 0 ? data.id : undefined}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Forecast */}
      {(horizonPayload?.forecast ?? data.forecast) && (
        <ForecastCard forecast={horizonPayload?.forecast ?? data.forecast} />
      )}
    </div>
  );
}

function RiskBanner({ risks }: { risks: TrainingRisk[] }) {
  return (
    <div className="space-y-2">
      {risks.map((r, i) => (
        <div
          key={i}
          className={cn(
            "flex items-start gap-3 p-4 rounded-xl border text-sm",
            r.severity === "high"   && "bg-red-500/10 border-red-500/25 text-red-300",
            r.severity === "medium" && "bg-amber-500/10 border-amber-500/25 text-amber-300",
            r.severity === "low"    && "bg-blue-500/10 border-blue-500/25 text-blue-300"
          )}
        >
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-medium capitalize">{r.type.replace("_", " ")}</span>
            {" — "}
            {r.message}
          </div>
        </div>
      ))}
    </div>
  );
}

function InsightGrid({ insights }: { insights: CoachInsight[] }) {
  const icons = {
    progress:    { icon: CheckCircle, color: "text-brand-400" },
    warning:     { icon: AlertTriangle, color: "text-amber-400" },
    tip:         { icon: Info, color: "text-blue-400" },
    achievement: { icon: Zap, color: "text-yellow-400" },
  };

  return (
    <div className="grid sm:grid-cols-2 gap-3">
      {insights.map((ins, i) => {
        const { icon: Icon, color } = icons[ins.type] ?? icons.tip;
        return (
          <div key={i} className="bg-surface-card border border-surface-border rounded-xl p-4">
            <div className={cn("flex items-center gap-2 mb-2 text-sm font-medium", color)}>
              <Icon className="w-4 h-4" />
              {ins.title}
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{ins.body}</p>
            {ins.value !== undefined && (
              <div className={cn("mt-2 text-lg font-bold", color)}>
                {ins.value}
                {ins.unit && <span className="text-xs text-slate-400 ml-1">{ins.unit}</span>}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function WorkoutCard({ plan, highlight, recId }: { plan: WorkoutPlan & { is_strength?: boolean; strength_session?: any; strength_addon?: any; gtg_practice?: any }; highlight: boolean; recId?: string }) {
  // Hooks must be called unconditionally before any early return
  const [feedbackSent, setFeedbackSent] = useState<"accepted" | "rejected" | null>(null);

  const feedback = useMutation({
    mutationFn: (action: "accepted" | "rejected") =>
      coachAPI.postFeedback(recId!, action),
    onSuccess: (_data, action) => {
      setFeedbackSent(action);
      toast.success(action === "accepted" ? "Logged — great work!" : "Noted, we'll adjust future recommendations.");
    },
    onError: () => toast.error("Couldn't save feedback"),
  });

  if (plan.is_strength) return <StrengthCard plan={plan} highlight={highlight} />;

  const labelDate = new Date();
  labelDate.setDate(labelDate.getDate() + plan.day_offset);
  const dayLabel =
    plan.day_offset === 0
      ? "Today"
      : plan.day_offset === 1
      ? "Tomorrow"
      : labelDate.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });

  const TYPE_COLOR: Record<string, string> = {
    easy: "text-blue-400 bg-blue-500/10 border-blue-500/20",
    endurance: "text-green-400 bg-green-500/10 border-green-500/20",
    tempo: "text-yellow-400 bg-yellow-500/10 border-yellow-500/20",
    sweetspot: "text-orange-400 bg-orange-500/10 border-orange-500/20",
    threshold: "text-orange-500 bg-orange-600/10 border-orange-600/20",
    vo2max: "text-red-400 bg-red-500/10 border-red-500/20",
    recovery: "text-slate-400 bg-slate-500/10 border-slate-500/20",
    rest: "text-slate-500 bg-slate-600/10 border-slate-600/20",
  };

  const color = TYPE_COLOR[plan.workout_type] ?? "text-slate-400 bg-slate-500/10 border-slate-500/20";

  return (
    <div
      className={cn(
        "bg-surface-card border rounded-xl p-4 transition-colors",
        highlight ? "border-brand-500/40 bg-brand-500/5" : "border-surface-border"
      )}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-slate-500 font-medium">{dayLabel}</span>
            <span className={cn("text-xs px-2 py-0.5 rounded-full border font-medium capitalize", color)}>
              {plan.workout_type.replace("_", " ")}
            </span>
          </div>
          <span className="font-semibold text-white">
            {plan.duration_minutes} min · TSS {plan.target_tss}
          </span>
        </div>
      </div>
      <p className="text-sm text-slate-400 mb-3 leading-relaxed">{plan.description}</p>

      {/* Intervals */}
      {plan.structure?.length > 0 && (
        <div className="space-y-1 mb-3">
          {plan.structure.map((seg, j) => (
            <div key={j} className="flex items-center gap-2 text-xs text-slate-500">
              <span className="capitalize text-slate-400 w-16">{seg.phase}</span>
              <span>{seg.duration_minutes} min</span>
              {seg.power_target_pct_ftp && (
                <span className="text-brand-400">{seg.power_target_pct_ftp}% FTP</span>
              )}
              {seg.hr_target_bpm && (
                <span className="text-pink-400">≤{seg.hr_target_bpm} bpm</span>
              )}
              <span className="flex-1 truncate">{seg.description}</span>
            </div>
          ))}
        </div>
      )}

      {plan.rationale && (
        <p className="text-xs text-slate-500 italic border-t border-surface-border pt-2 mt-2">
          💡 {plan.rationale}
        </p>
      )}

      {/* Strength add-on after easy ride */}
      {plan.strength_addon && (
        <div className="border-t border-surface-border pt-3 mt-3">
          <div className="flex items-center gap-2 mb-2">
            <Dumbbell className="w-3.5 h-3.5 text-violet-400" />
            <span className="text-xs font-medium text-violet-400">
              Add-on: {plan.strength_addon.name}
            </span>
            <span className="text-xs text-slate-500">
              {plan.strength_addon.duration_minutes} min after this ride
            </span>
          </div>
          <SessionExerciseList session={plan.strength_addon} />
        </div>
      )}

      {/* GTG daily practice badge */}
      {plan.gtg_practice && (
        <div className="border-t border-surface-border pt-3 mt-3">
          <div className="flex items-center gap-2 mb-1">
            <Dumbbell className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-xs font-medium text-emerald-400">GTG today</span>
            <span className="text-xs text-slate-500">
              5 reps every 1-2h throughout the day
            </span>
          </div>
          <div className="flex gap-2 flex-wrap mt-1">
            {plan.gtg_practice.exercises?.map((ex: any, i: number) => (
              <span
                key={i}
                className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 capitalize"
              >
                {ex.name.replace(/_/g, " ")} × {ex.reps}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Feedback widget — only on today's workout, only if not yet sent */}
      {recId && (
        <div className="border-t border-surface-border pt-3 mt-3">
          {feedbackSent ? (
            <p className="text-xs text-slate-500 flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
              {feedbackSent === "accepted" ? "Workout logged" : "Feedback saved"}
            </p>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 flex-1">Did you complete this workout?</span>
              <button
                onClick={() => feedback.mutate("accepted")}
                disabled={feedback.isPending}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 hover:bg-emerald-500/20 transition-colors disabled:opacity-50"
              >
                <ThumbsUp className="w-3 h-3" /> Yes
              </button>
              <button
                onClick={() => feedback.mutate("rejected")}
                disabled={feedback.isPending}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-slate-500/10 border border-slate-500/25 text-slate-400 hover:bg-slate-500/20 transition-colors disabled:opacity-50"
              >
                <ThumbsDown className="w-3 h-3" /> No
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Strength training components ───────────────────────────────────────────

function ExerciseRow({ ex }: { ex: any }) {
  const [open, setOpen] = useState(false);
  const restLabel =
    ex.rest_sec >= 60
      ? `${Math.round(ex.rest_sec / 60)} min rest`
      : ex.rest_sec > 0
      ? `${ex.rest_sec}s rest`
      : "";

  return (
    <div className="text-sm">
      <button onClick={() => setOpen(!open)} className="flex items-start gap-3 w-full text-left">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-white capitalize">
              {ex.name.replace(/_/g, " ")}
            </span>
            <span className="text-xs text-violet-400">
              {ex.sets}×{ex.reps}
            </span>
            {restLabel && <span className="text-xs text-slate-500">{restLabel}</span>}
          </div>
          <div className="text-xs text-slate-500 mt-0.5 truncate">
            {ex.equipment} · {ex.cycling_benefit}
          </div>
        </div>
        {open ? (
          <ChevronUp className="w-3.5 h-3.5 text-slate-500 flex-shrink-0 mt-0.5" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 text-slate-500 flex-shrink-0 mt-0.5" />
        )}
      </button>
      {open && (
        <div className="mt-2 space-y-1.5 text-xs text-slate-400 pl-0">
          <p className="leading-relaxed">{ex.description}</p>
          {ex.cue && (
            <p className="text-brand-400 font-medium">Cue: {ex.cue}</p>
          )}
          {ex.weight_guidance && (
            <p className="text-amber-400/80">Weight: {ex.weight_guidance}</p>
          )}
        </div>
      )}
    </div>
  );
}

function SessionExerciseList({ session }: { session: any }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs text-slate-400 hover:text-white transition-colors w-full mt-2"
      >
        <span>{session.exercises.length} exercises</span>
        {expanded ? (
          <ChevronUp className="w-3.5 h-3.5 ml-auto" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 ml-auto" />
        )}
      </button>
      {expanded && (
        <div className="mt-3 space-y-3 border-t border-surface-border pt-3">
          {session.exercises.map((ex: any, i: number) => (
            <ExerciseRow key={i} ex={ex} />
          ))}
        </div>
      )}
    </>
  );
}

function StrengthCard({
  plan,
  highlight,
}: {
  plan: WorkoutPlan & { strength_session?: any };
  highlight: boolean;
}) {
  const session = plan.strength_session;
  if (!session) return null;

  const labelDate = new Date();
  labelDate.setDate(labelDate.getDate() + plan.day_offset);
  const dayLabel =
    plan.day_offset === 0
      ? "Today"
      : plan.day_offset === 1
      ? "Tomorrow"
      : labelDate.toLocaleDateString(undefined, {
          weekday: "short",
          month: "short",
          day: "numeric",
        });

  return (
    <div
      className={cn(
        "bg-surface-card border rounded-xl p-4 transition-colors",
        highlight
          ? "border-violet-500/40 bg-violet-500/5"
          : "border-surface-border"
      )}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-xs text-slate-500 font-medium">{dayLabel}</span>
            <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-violet-500/10 border-violet-500/20 text-violet-400">
              Strength
            </span>
            <span className="text-xs text-slate-400">{session.phase_label}</span>
          </div>
          <span className="font-semibold text-white">{session.name}</span>
          <div className="text-xs text-slate-400 mt-0.5">
            {session.duration_minutes} min · TSS ~{plan.target_tss}
          </div>
        </div>
        <Dumbbell className="w-5 h-5 text-violet-400 flex-shrink-0 mt-1" />
      </div>

      <p className="text-sm text-slate-400 mb-2 leading-relaxed">{session.notes}</p>

      <SessionExerciseList session={session} />

      {plan.rationale && (
        <p className="text-xs text-slate-500 italic border-t border-surface-border pt-2 mt-3">
          💡 {plan.rationale}
        </p>
      )}
    </div>
  );
}

function ForecastCard({ forecast }: { forecast: CoachRecommendation["forecast"] }) {
  const sign = forecast.predicted_ftp_change_watts >= 0 ? "+" : "";
  return (
    <div className="bg-surface-card border border-surface-border rounded-2xl p-6">
      <h2 className="font-semibold text-white mb-4">
        {forecast.weeks}-Week Fitness Forecast
      </h2>
      <div className="grid grid-cols-3 gap-6">
        <div>
          <div className="text-2xl font-bold text-brand-400">
            {sign}{forecast.predicted_ftp_change_watts} W
          </div>
          <div className="text-xs text-slate-400 mt-1">Projected FTP change</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-blue-400">
            {Math.round(forecast.predicted_ctl_peak)}
          </div>
          <div className="text-xs text-slate-400 mt-1">Peak CTL target</div>
        </div>
        {forecast.event_readiness_pct !== undefined && (
          <div>
            <div className="text-2xl font-bold text-purple-400">
              {Math.round(forecast.event_readiness_pct)}%
            </div>
            <div className="text-xs text-slate-400 mt-1">Event readiness</div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Macrocycle: reverse-periodization plan to event date ──────────────────
// Week-by-week phase + target TSS scaffold computed from current CTL → peak CTL.
// Day-to-day model recs slot inside this skeleton.
const PHASE_STYLES: Record<string, string> = {
  base:       "bg-blue-500/15 border-blue-500/30 text-blue-300",
  build:      "bg-amber-500/15 border-amber-500/30 text-amber-300",
  peak:       "bg-orange-500/15 border-orange-500/30 text-orange-300",
  taper:      "bg-emerald-500/15 border-emerald-500/30 text-emerald-300",
  event_week: "bg-fuchsia-500/15 border-fuchsia-500/30 text-fuchsia-300",
};

const FEASIBILITY_BADGE: Record<Macrocycle["feasibility"], string> = {
  comfortable:  "bg-emerald-500/10 border-emerald-500/30 text-emerald-300",
  balanced:     "bg-blue-500/10 border-blue-500/30 text-blue-300",
  ambitious:    "bg-amber-500/10 border-amber-500/30 text-amber-300",
  unrealistic:  "bg-red-500/10 border-red-500/30 text-red-300",
};

function MacrocycleCard({ macro }: { macro: Macrocycle }) {
  const eventDate = new Date(macro.event_date);
  const eventLabel = eventDate.toLocaleDateString(undefined, {
    weekday: "short", month: "short", day: "numeric",
  });

  return (
    <div className="bg-surface-card border border-surface-border rounded-2xl">
      {/* Sticky header — always visible */}
      <div className="px-5 pt-5 pb-3 border-b border-surface-border">
        <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Target className="w-5 h-5 text-brand-400" />
            Macrocycle to {macro.event_name || "event"}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            {eventLabel} · {macro.days_to_event} days · {macro.weeks_to_event} weeks
            {macro.event_type && (
              <span className="ml-1 capitalize">· {macro.event_type.replace(/_/g, " ")}</span>
            )}
          </p>
        </div>
        <span className={cn(
          "text-xs px-2.5 py-1 rounded-full border capitalize",
          FEASIBILITY_BADGE[macro.feasibility]
        )}>
          {macro.feasibility}
        </span>
        </div>
      </div>

      {/* Body */}
      <div className="px-5 py-4 space-y-4">
        {/* CTL ramp summary */}
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-surface-bg border border-surface-border rounded-xl p-3">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Current CTL</div>
            <div className="text-xl font-bold text-white mt-1">{macro.current_ctl.toFixed(0)}</div>
          </div>
          <div className="bg-surface-bg border border-surface-border rounded-xl p-3">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Peak Target</div>
            <div className="text-xl font-bold text-brand-400 mt-1">{macro.peak_ctl_target.toFixed(0)}</div>
          </div>
          <div className="bg-surface-bg border border-surface-border rounded-xl p-3">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Race-day TSB</div>
            <div className="text-xl font-bold text-emerald-400 mt-1">+{macro.planned_tsb_event.toFixed(0)}</div>
          </div>
        </div>

        {/* Week-by-week strip */}
        <div>
          <div className="text-xs font-medium text-slate-400 mb-2">Weekly schedule</div>
          <div className="overflow-x-auto -mx-1 px-1">
            <div className="flex gap-2 min-w-max pb-1">
              {macro.weeks.map((w) => (
                <MacrocycleWeekCell key={w.week_index} week={w} />
              ))}
            </div>
          </div>
        </div>

        {/* Plan summary */}
        {macro.summary?.length > 0 && (
          <ul className="text-xs text-slate-400 space-y-1 pt-1 border-t border-surface-border">
            {macro.summary.map((s, i) => (
              <li key={i} className="flex gap-2"><span className="text-slate-600">·</span>{s}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function MacrocycleWeekCell({ week }: { week: MacrocycleWeek }) {
  const date = new Date(week.week_start);
  const label = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const phaseStyle = PHASE_STYLES[week.phase] ?? PHASE_STYLES.build;

  return (
    <div
      title={`${week.notes}\nFocus: ${week.workout_focus.join(", ")}`}
      className={cn(
        "min-w-[88px] rounded-xl border p-2.5 text-center transition-transform hover:-translate-y-0.5",
        phaseStyle,
        week.is_recovery_week && "ring-1 ring-slate-500/40 ring-offset-1 ring-offset-surface-card",
      )}
    >
      <div className="text-[10px] opacity-70">W{week.week_index + 1} · {label}</div>
      <div className="text-xs font-semibold capitalize mt-0.5">
        {week.phase.replace("_", " ")}
      </div>
      <div className="text-base font-bold text-white mt-1">{week.target_weekly_tss}</div>
      <div className="text-[10px] opacity-70">TSS · CTL {week.target_ctl_end.toFixed(0)}</div>
      {week.is_recovery_week && (
        <div className="text-[9px] mt-1 uppercase tracking-wide opacity-80">recovery</div>
      )}
    </div>
  );
}

// ─── Horizon Picker ─────────────────────────────────────────────────────────
// Three-tab selector for short / medium / event horizons.
// Backend recommends one (`active_horizon`); user can override to compare.
function HorizonPicker({
  multi,
  activeHorizon,
  onSelect,
  isUserOverride,
}: {
  multi: import("@/types").MultiHorizonRecommendation;
  activeHorizon: HorizonKey | null;
  onSelect: (h: HorizonKey | null) => void;
  isUserOverride: boolean;
}) {
  const order: HorizonKey[] = ["short", "medium", "event"];
  const present = order.filter((k) => multi.horizons[k]);
  if (present.length <= 1) return null;

  const ICONS: Record<HorizonKey, JSX.Element> = {
    short:  <Zap className="w-3.5 h-3.5" />,
    medium: <Calendar className="w-3.5 h-3.5" />,
    event:  <Target className="w-3.5 h-3.5" />,
  };
  const TITLES: Record<HorizonKey, string> = {
    short:  "Short term",
    medium: "Medium build",
    event:  "Event peak",
  };

  const activeH = activeHorizon && multi.horizons[activeHorizon] ? multi.horizons[activeHorizon]! : null;

  return (
    <div className="bg-surface-card border border-surface-border rounded-2xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">
          Planning horizon
          {!isUserOverride && (
            <span className="ml-2 text-slate-500 lowercase tracking-normal">
              · auto-selected
            </span>
          )}
        </div>
        {isUserOverride && (
          <button onClick={() => onSelect(null)} className="text-xs text-slate-400 hover:text-white">
            Reset
          </button>
        )}
      </div>

      {/* ── Mobile: compact tab strip ── */}
      <div className="flex sm:hidden gap-2 mb-3">
        {present.map((key) => {
          const active = activeHorizon === key;
          const isAuto = !isUserOverride && multi.active_horizon === key;
          return (
            <button
              key={key}
              onClick={() => onSelect(key)}
              className={cn(
                "flex-1 flex flex-col items-center gap-1 py-2 px-1 rounded-xl border text-xs font-medium transition-colors",
                active
                  ? "bg-brand-500/10 border-brand-500/40 text-brand-400"
                  : "border-surface-border text-slate-400"
              )}
            >
              {ICONS[key]}
              <span className="leading-tight text-center">{TITLES[key]}</span>
              {isAuto && <span className="text-[9px] text-brand-400">AI</span>}
            </button>
          );
        })}
      </div>

      {/* Mobile: selected horizon details */}
      {activeH && (
        <div className="sm:hidden text-xs text-slate-400 space-y-0.5 mb-1">
          <div className="font-medium text-slate-300">{activeH.horizon_label}</div>
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-slate-500">
            <span>{activeH.next_workout?.duration_minutes ?? 0} min</span>
            <span>TSS {activeH.next_workout?.target_tss ?? 0}</span>
            <span className="capitalize">{activeH.next_workout?.workout_type?.replace("_", " ")}</span>
          </div>
        </div>
      )}

      {/* ── Desktop: full cards ── */}
      <div className="hidden sm:grid grid-cols-3 gap-2">
        {present.map((key) => {
          const h = multi.horizons[key]!;
          const active = activeHorizon === key;
          const isAuto = !isUserOverride && multi.active_horizon === key;
          return (
            <button
              key={key}
              onClick={() => onSelect(key)}
              className={cn(
                "text-left p-3 rounded-xl border transition-colors",
                active
                  ? "bg-brand-500/10 border-brand-500/40 text-white"
                  : "bg-surface-card-hover border-surface-border text-slate-300 hover:border-brand-500/30"
              )}
            >
              <div className="flex items-center gap-2 text-sm font-semibold">
                {ICONS[key]}
                {TITLES[key]}
                {isAuto && (
                  <span className="ml-auto text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 rounded px-1.5 py-0.5">
                    AI pick
                  </span>
                )}
              </div>
              <div className="text-xs text-slate-400 mt-1 leading-snug">{h.horizon_label}</div>
              <div className="text-xs text-slate-500 mt-1.5 flex flex-wrap gap-x-2 gap-y-0.5">
                <span>{h.next_workout?.duration_minutes ?? 0} min</span>
                <span>·</span>
                <span>TSS {h.next_workout?.target_tss ?? 0}</span>
                <span>·</span>
                <span className="capitalize">{h.next_workout?.workout_type?.replace("_", " ")}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
