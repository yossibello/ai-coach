"use client";

import { useQuery } from "@tanstack/react-query";
import { Heart, Moon, Activity, Battery } from "lucide-react";
import { healthAPI, HealthDay, Readiness } from "@/lib/api";

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

function MiniSpark({ values }: { values: (number | null | undefined)[] }) {
  const nums = values.map((v) => (v == null ? null : Number(v)));
  const valid = nums.filter((v): v is number => v != null);
  if (valid.length < 2) {
    return <div className="h-6 text-[10px] text-slate-500">no trend</div>;
  }
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const range = max - min || 1;
  const w = 80;
  const h = 24;
  const stepX = w / Math.max(1, nums.length - 1);
  const pts = nums
    .map((v, i) => {
      if (v == null) return null;
      const y = h - ((v - min) / range) * h;
      return `${i * stepX},${y.toFixed(1)}`;
    })
    .filter(Boolean)
    .join(" ");
  return (
    <svg width={w} height={h} className="block">
      <polyline
        points={pts}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function StatBlock({
  icon: Icon,
  label,
  value,
  unit,
  trend,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  unit?: string;
  trend?: (number | null | undefined)[];
  tone: string;
}) {
  return (
    <div className="rounded-xl bg-surface-bg/50 border border-surface-border p-3 flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-slate-400">
        <Icon className={`w-3.5 h-3.5 ${tone}`} />
        {label}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-semibold text-white">{value}</span>
        {unit && <span className="text-[11px] text-slate-500">{unit}</span>}
      </div>
      {trend && (
        <div className={`${tone}`}>
          <MiniSpark values={trend} />
        </div>
      )}
    </div>
  );
}

export function HealthWidget() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["health-recent"],
    queryFn: () => healthAPI.recent(30),
    staleTime: 60_000,
  });

  if (isLoading) {
    return (
      <div className="rounded-2xl border border-surface-border bg-surface-card p-6 text-sm text-slate-400">
        Loading recovery signals…
      </div>
    );
  }

  if (isError || !data) {
    return null;
  }

  const { days, readiness } = data;
  const today: HealthDay | undefined = days[days.length - 1];

  if (!today) {
    return (
      <div className="rounded-2xl border border-surface-border bg-surface-card p-6">
        <h2 className="font-semibold text-white mb-1">Recovery & Readiness</h2>
        <p className="text-sm text-slate-400">
          Connect Garmin in Profile → Integrations to see HRV, resting HR,
          sleep, and body battery here.
        </p>
      </div>
    );
  }

  const hrvSeries = days.map((d) => d.hrv_overnight_avg_ms);
  const rhrSeries = days.map((d) => d.resting_hr);
  const sleepSeries = days.map((d) => d.sleep_score);
  const bbSeries = days.map((d) => d.body_battery_high);
  const sleepHours =
    today.sleep_total_seconds != null
      ? (today.sleep_total_seconds / 3600).toFixed(1)
      : "—";

  return (
    <div className="rounded-2xl border border-surface-border bg-surface-card p-6 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-semibold text-white">Recovery & Readiness</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Garmin daily wellness · last 30 days
          </p>
        </div>
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm font-medium ${STATUS_BG[readiness.status]}`}
        >
          <span className={`w-2 h-2 rounded-full ${STATUS_DOT[readiness.status]}`} />
          {Math.round(readiness.score)}/100
          <span className="opacity-70 capitalize">{readiness.status}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatBlock
          icon={Activity}
          label="HRV (overnight)"
          value={today.hrv_overnight_avg_ms ? today.hrv_overnight_avg_ms.toFixed(0) : "—"}
          unit="ms"
          trend={hrvSeries}
          tone="text-violet-300"
        />
        <StatBlock
          icon={Heart}
          label="Resting HR"
          value={today.resting_hr != null ? String(today.resting_hr) : "—"}
          unit="bpm"
          trend={rhrSeries}
          tone="text-rose-300"
        />
        <StatBlock
          icon={Moon}
          label="Sleep"
          value={sleepHours}
          unit={`h · ${today.sleep_score ?? "—"}/100`}
          trend={sleepSeries}
          tone="text-blue-300"
        />
        <StatBlock
          icon={Battery}
          label="Body Battery"
          value={today.body_battery_high != null ? String(today.body_battery_high) : "—"}
          unit="/100"
          trend={bbSeries}
          tone="text-emerald-300"
        />
      </div>

      <div className="rounded-xl bg-surface-bg/50 border border-surface-border p-3 space-y-2">
        <p className="text-sm text-slate-200">{readiness.advice}</p>
        {readiness.drivers.length > 0 && (
          <ul className="text-xs text-slate-400 space-y-0.5 list-disc list-inside">
            {readiness.drivers.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
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
    </div>
  );
}
