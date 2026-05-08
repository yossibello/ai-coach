"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { coachAPI } from "@/lib/api";
import { Brain, RefreshCw, AlertTriangle, CheckCircle, Info, Zap, Calendar } from "lucide-react";
import { cn, formatDuration } from "@/lib/utils";
import type { CoachRecommendation, WorkoutPlan, CoachInsight, TrainingRisk } from "@/types";
import toast from "react-hot-toast";

export default function CoachPage() {
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["coach-recommendation"],
    queryFn: () => coachAPI.getRecommendation(),
  });

  const refresh = useMutation({
    mutationFn: () => coachAPI.refreshRecommendation(),
    onSuccess: (d) => {
      qc.setQueryData(["coach-recommendation"], d);
      toast.success("Recommendation updated");
    },
  });

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
    <div className="p-6 space-y-6 animate-fade-in">
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

      {/* Risks */}
      {data.risks?.length > 0 && <RiskBanner risks={data.risks} />}

      {/* Insights */}
      {data.insights?.length > 0 && <InsightGrid insights={data.insights} />}

      {/* Weekly plan */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-brand-400" />
          7-Day Training Plan
        </h2>
        <div className="space-y-3">
          {data.weekly_plan?.map((w, i) => (
            <WorkoutCard key={i} plan={w} highlight={i === 0} />
          ))}
        </div>
      </div>

      {/* Forecast */}
      {data.forecast && <ForecastCard forecast={data.forecast} />}
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

function WorkoutCard({ plan, highlight }: { plan: WorkoutPlan; highlight: boolean }) {
  const dayLabel = plan.day_offset === 0 ? "Today" : plan.day_offset === 1 ? "Tomorrow" : `Day ${plan.day_offset + 1}`;

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
        "bg-surface-card border rounded-xl p-5 transition-colors",
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
