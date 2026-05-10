"use client";

import type { CoachRecommendation } from "@/types";
import Link from "next/link";
import { Brain, ArrowRight, AlertTriangle } from "lucide-react";

interface Props {
  recommendation: CoachRecommendation;
}

export function CoachTip({ recommendation }: Props) {
  const { next_workout, risks, is_cold_start } = recommendation;
  const hasHighRisk = risks?.some((r) => r.severity === "high");

  return (
    <div className="bg-gradient-to-br from-brand-500/10 to-emerald-600/5 border border-brand-500/25 rounded-2xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="w-5 h-5 text-brand-400" />
          <span className="text-sm font-semibold text-brand-300">
            AI Coach {is_cold_start && <span className="text-xs text-slate-500">(guided start)</span>}
          </span>
        </div>
        {hasHighRisk && (
          <div className="flex items-center gap-1 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-2 py-1">
            <AlertTriangle className="w-3 h-3" />
            Risk detected
          </div>
        )}
      </div>

      {next_workout ? (
        <>
          <div className="mb-2">
            <span className="text-lg font-bold text-white capitalize">
              {next_workout.day_offset === 0 ? "Today" : next_workout.day_offset === 1 ? "Tomorrow" : `Day ${next_workout.day_offset + 1}`}:{" "}
              {next_workout.workout_type.replace("_", " ")}
            </span>
            <span className="text-slate-400 text-sm ml-2">· {next_workout.duration_minutes} min</span>
          </div>
          <p className="text-sm text-slate-300 mb-3 leading-relaxed">{next_workout.rationale}</p>
          <div className="text-xs text-slate-400 mb-4">
            🎯 <strong className="text-white">{next_workout.key_metric}</strong>
          </div>
        </>
      ) : (
        <p className="text-sm text-slate-300 mb-4">No workout recommendation yet.</p>
      )}

      <Link
        href="/coach"
        className="inline-flex items-center gap-1.5 text-sm text-brand-400 hover:text-brand-300 font-medium"
      >
        Full plan & insights <ArrowRight className="w-3.5 h-3.5" />
      </Link>
    </div>
  );
}
