"use client";

import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  icon: LucideIcon;
  label: string;
  value: string;
  sub: string;
  color: string;
}

export function MetricCard({ icon: Icon, label, value, sub, color }: Props) {
  return (
    <div className="bg-surface-card border border-surface-border rounded-2xl p-5 hover:border-brand-500/30 transition-colors">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
        <Icon className={cn("w-4 h-4", color)} />
      </div>
      <div className={cn("text-2xl font-bold mb-1", color)}>{value}</div>
      <div className="text-xs text-slate-500 truncate">{sub}</div>
    </div>
  );
}
