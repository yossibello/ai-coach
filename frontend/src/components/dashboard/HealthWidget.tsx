"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Heart, Moon, Activity, Battery, Plus, TrendingDown, TrendingUp, Minus, Wind } from "lucide-react";
import { healthAPI, HealthDay, Readiness, DriftStatus, ManualHealthLog } from "@/lib/api";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";

const STATUS_BG: Record<Readiness["status"], string> = {
  green: "bg-emerald-500/15 border-emerald-500/40 text-emerald-300",
  amber: "bg-amber-500/15 border-amber-500/40 text-amber-300",
  red:   "bg-rose-500/15 border-rose-500/40 text-rose-300",
};

const STATUS_DOT: Record<Readiness["status"], string> = {
  green: "bg-emerald-400",
  amber: "bg-amber-400",
  red:   "bg-rose-400",
};

const DRIFT_COLORS: Record<DriftStatus["state"], string> = {
  stable:    "text-emerald-400 bg-emerald-500/10 border-emerald-500/25",
  decoupled: "text-amber-400 bg-amber-500/10 border-amber-500/25",
  stressed:  "text-rose-400 bg-rose-500/10 border-rose-500/25",
  unknown:   "text-slate-400 bg-slate-500/10 border-slate-500/25",
};

function MiniSpark({ values, tone }: { values: (number | null | undefined)[]; tone: string }) {
  const nums = values.map((v) => (v == null ? null : Number(v)));
  const valid = nums.filter((v): v is number => v != null);
  if (valid.length < 2) return <div className="h-6 text-[10px] text-slate-500">—</div>;
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const range = max - min || 1;
  const w = 80; const h = 24;
  const stepX = w / Math.max(1, nums.length - 1);
  const pts = nums
    .map((v, i) => v == null ? null : `${i * stepX},${(h - ((v - min) / range) * h).toFixed(1)}`)
    .filter(Boolean).join(" ");
  return (
    <svg width={w} height={h} className={cn("block", tone)}>
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function StatBlock({ icon: Icon, label, value, unit, trend, tone }: {
  icon: React.ComponentType<{ className?: string }>;
  label: string; value: string; unit?: string;
  trend?: (number | null | undefined)[]; tone: string;
}) {
  return (
    <div className="rounded-xl bg-surface-bg/50 border border-surface-border p-3 flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-slate-400">
        <Icon className={cn("w-3.5 h-3.5", tone)} />{label}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-semibold text-white">{value}</span>
        {unit && <span className="text-[11px] text-slate-500">{unit}</span>}
      </div>
      {trend && <MiniSpark values={trend} tone={tone} />}
    </div>
  );
}

function DriftCard({ drift }: { drift: DriftStatus }) {
  if (drift.state === "unknown") return null;
  const TrendIcon = drift.trend === "improving" ? TrendingDown : drift.trend === "worsening" ? TrendingUp : Minus;
  return (
    <div className={cn("rounded-xl border p-3 flex items-start gap-3", DRIFT_COLORS[drift.state])}>
      <Wind className="w-4 h-4 flex-shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold capitalize">HR Drift · {drift.state}</span>
          {drift.drift_pct != null && (
            <span className="text-xs opacity-80">{drift.drift_pct.toFixed(1)}%</span>
          )}
          <span className="flex items-center gap-0.5 text-xs opacity-70">
            <TrendIcon className="w-3 h-3" />{drift.trend}
          </span>
          {drift.overtraining_risk && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-500/20 border border-rose-500/30 text-rose-300">
              overtraining risk
            </span>
          )}
        </div>
        <p className="text-xs opacity-70 mt-0.5">{drift.action}</p>
      </div>
    </div>
  );
}

function ManualLogForm({ onClose, hasToday }: { onClose: () => void; hasToday: boolean }) {
  const qc = useQueryClient();
  const [form, setForm] = useState<ManualHealthLog>({});

  const save = useMutation({
    mutationFn: () => healthAPI.logManual(form),
    onSuccess: () => {
      toast.success("Health data logged");
      qc.invalidateQueries({ queryKey: ["health-recent"] });
      onClose();
    },
    onError: () => toast.error("Failed to save"),
  });

  const set = (k: keyof ManualHealthLog, v: string) =>
    setForm((f) => ({ ...f, [k]: v === "" ? undefined : Number(v) }));

  return (
    <div className="rounded-xl border border-surface-border bg-surface-bg/70 p-4 space-y-3">
      <p className="text-xs text-slate-400">
        {hasToday ? "Update today's manual entry." : "Log today's recovery signals manually."}
        {" "}Garmin data (if synced) takes priority except for fields you set explicitly.
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {[
          { key: "hrv_ms",      label: "HRV", unit: "ms",  placeholder: "e.g. 55",  min: 10,  max: 250 },
          { key: "resting_hr",  label: "Resting HR", unit: "bpm", placeholder: "e.g. 52", min: 25, max: 130 },
          { key: "sleep_hours", label: "Sleep", unit: "h", placeholder: "e.g. 7.5", min: 0, max: 24, step: 0.5 },
          { key: "sleep_score", label: "Sleep quality", unit: "/100", placeholder: "0–100", min: 0, max: 100 },
          { key: "energy_level",label: "Energy", unit: "/100", placeholder: "0–100", min: 0, max: 100 },
          { key: "stress_level",label: "Stress", unit: "/100", placeholder: "0–100", min: 0, max: 100 },
        ].map(({ key, label, unit, placeholder, min, max, step }) => (
          <div key={key}>
            <label className="block text-[11px] text-slate-400 mb-1">{label} <span className="text-slate-600">{unit}</span></label>
            <input
              type="number"
              min={min} max={max} step={step ?? 1}
              placeholder={placeholder}
              value={(form as any)[key] ?? ""}
              onChange={(e) => set(key as keyof ManualHealthLog, e.target.value)}
              className="w-full bg-surface-card border border-surface-border rounded-lg px-2 py-1.5 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-brand-500/60"
            />
          </div>
        ))}
      </div>
      <div className="flex gap-2 pt-1">
        <button
          onClick={() => save.mutate()}
          disabled={save.isPending || Object.keys(form).filter(k => (form as any)[k] != null).length === 0}
          className="flex-1 py-2 rounded-xl bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 disabled:opacity-40 transition-colors"
        >
          {save.isPending ? "Saving…" : "Save"}
        </button>
        <button onClick={onClose} className="px-4 py-2 rounded-xl border border-surface-border text-slate-400 text-sm hover:text-white">
          Cancel
        </button>
      </div>
    </div>
  );
}

export function HealthWidget() {
  const [showLog, setShowLog] = useState(false);
  const { data, isLoading } = useQuery({
    queryKey: ["health-recent"],
    queryFn: () => healthAPI.recent(30),
    staleTime: 0,   // always refetch on mount so drift stays current
  });

  if (isLoading) {
    return (
      <div className="rounded-2xl border border-surface-border bg-surface-card p-6 text-sm text-slate-400">
        Loading recovery signals…
      </div>
    );
  }

  const days = data?.days ?? [];
  const readiness = data?.readiness;
  const drift = data?.drift ?? null;
  console.log("[HealthWidget] data:", JSON.stringify({ drift, hasData: !!data }));
  const hasManualToday = data?.has_manual_today ?? false;
  const today: HealthDay | undefined = days[days.length - 1];
  const hasGarminData = today != null;

  const hrvSeries  = days.map((d) => d.hrv_overnight_avg_ms);
  const rhrSeries  = days.map((d) => d.resting_hr);
  const sleepSeries = days.map((d) => d.sleep_score);
  const bbSeries   = days.map((d) => d.body_battery_high);
  const sleepHours = today?.sleep_total_seconds != null
    ? (today.sleep_total_seconds / 3600).toFixed(1) : "—";

  return (
    <div className="rounded-2xl border border-surface-border bg-surface-card p-6 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="font-semibold text-white">Recovery & Readiness</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            {hasGarminData ? `${today.source === "manual" ? "Manual entry" : "Garmin"} · last 30 days` : "No device data yet"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {readiness && (
            <div className={cn("flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm font-medium", STATUS_BG[readiness.status])}>
              <span className={cn("w-2 h-2 rounded-full", STATUS_DOT[readiness.status])} />
              {Math.round(readiness.score)}/100
              <span className="opacity-70 capitalize">{readiness.status}</span>
            </div>
          )}
          <button
            onClick={() => setShowLog((v) => !v)}
            title="Log health manually"
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors",
              showLog
                ? "bg-brand-500/15 border-brand-500/40 text-brand-400"
                : "border-surface-border text-slate-400 hover:text-white hover:border-slate-500"
            )}
          >
            <Plus className="w-3.5 h-3.5" />
            {hasManualToday ? "Edit log" : "Log today"}
          </button>
        </div>
      </div>

      {/* Manual log form */}
      {showLog && (
        <ManualLogForm onClose={() => setShowLog(false)} hasToday={hasManualToday} />
      )}

      {/* HR Drift card — shown even without Garmin */}
      {drift && <DriftCard drift={drift} />}

      {/* Stats grid */}
      {hasGarminData ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatBlock icon={Activity} label="HRV (overnight)"
            value={today.hrv_overnight_avg_ms ? today.hrv_overnight_avg_ms.toFixed(0) : "—"}
            unit="ms" trend={hrvSeries} tone="text-violet-300" />
          <StatBlock icon={Heart} label="Resting HR"
            value={today.resting_hr != null ? String(today.resting_hr) : "—"}
            unit="bpm" trend={rhrSeries} tone="text-rose-300" />
          <StatBlock icon={Moon} label="Sleep"
            value={sleepHours}
            unit={`h · ${today.sleep_score ?? "—"}/100`}
            trend={sleepSeries} tone="text-blue-300" />
          <StatBlock icon={Battery} label="Energy"
            value={today.body_battery_high != null ? String(today.body_battery_high) : "—"}
            unit="/100" trend={bbSeries} tone="text-emerald-300" />
        </div>
      ) : (
        <p className="text-sm text-slate-400">
          No health data yet — connect Garmin in Profile, or use the <span className="text-brand-400">Log today</span> button above to enter HRV, sleep, and energy manually.
        </p>
      )}

      {/* Advice */}
      {readiness && (
        <div className="rounded-xl bg-surface-bg/50 border border-surface-border p-3 space-y-2">
          <p className="text-sm text-slate-200">{readiness.advice}</p>
          {readiness.drivers.length > 0 && (
            <ul className="text-xs text-slate-400 space-y-0.5 list-disc list-inside">
              {readiness.drivers.map((d, i) => <li key={i}>{d}</li>)}
            </ul>
          )}
          <div className="flex flex-wrap gap-3 pt-1 text-[11px] text-slate-500">
            {readiness.hrv_z != null && (
              <span>HRV z-score: <span className="text-slate-300">{readiness.hrv_z.toFixed(2)}</span></span>
            )}
            {readiness.rhr_delta != null && (
              <span>RHR Δ vs 30d: <span className="text-slate-300">{readiness.rhr_delta > 0 ? "+" : ""}{readiness.rhr_delta.toFixed(1)} bpm</span></span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
