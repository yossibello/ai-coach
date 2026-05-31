"use client";

import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { MetricExplainer } from "./MetricExplainer";

interface Props {
  icon: LucideIcon;
  label: string;
  value: string;
  sub: string;
  color: string;
  detail?: string;
  confidence?: number; // 0–1
  explainer?: "FTP" | "CTL" | "ATL" | "TSB" | "TSS";
  numericValue?: number;
}

export function MetricCard({ icon: Icon, label, value, sub, color, detail, confidence, explainer, numericValue }: Props) {
  const confPct = confidence != null ? Math.round(confidence * 100) : null;
  const confColor =
    confPct == null ? "" :
    confPct >= 80 ? "bg-emerald-500" :
    confPct >= 55 ? "bg-amber-500" : "bg-rose-500";

  return (
    <div className="bg-surface-card border border-surface-border rounded-2xl p-5 hover:border-brand-500/30 transition-colors">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
          {explainer && <MetricExplainer metric={explainer} currentValue={numericValue} />}
        </div>
        <Icon className={cn("w-4 h-4", color)} />
      </div>
      <div className={cn("text-2xl font-bold mb-1", color)}>{value}</div>
      <div className="text-xs text-slate-500 truncate">{sub}</div>
      {detail && (
        <div className="text-[11px] text-slate-500 truncate mt-0.5">{detail}</div>
      )}
      {confPct != null && (
        <div className="mt-2 flex items-center gap-1.5">
          <div className="flex-1 h-1 rounded-full bg-surface-border overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all", confColor)}
              style={{ width: `${confPct}%` }}
            />
          </div>
          <span className="text-[10px] text-slate-500 tabular-nums">{confPct}%</span>
        </div>
      )}
    </div>
  );
}
