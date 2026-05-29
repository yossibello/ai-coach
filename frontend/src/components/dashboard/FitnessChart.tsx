"use client";

import { useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { format } from "date-fns";
import { Heart } from "lucide-react";
import { cn } from "@/lib/utils";
import type { FitnessSnapshot } from "@/types";
import type { HealthDay } from "@/lib/api";

interface Props {
  data: FitnessSnapshot[];
  healthDays?: HealthDay[];
}

// Compute how much healthier/worse the athlete is vs. what TSB alone shows.
// Returns a TSB adjustment in the same units (-20 to +15).
function healthModifier(h: HealthDay): number {
  let mod = 0;
  // HRV: overnight vs 7-day baseline — most reliable signal
  if (h.hrv_overnight_avg_ms && h.hrv_7d_avg_ms && h.hrv_7d_avg_ms > 0) {
    const delta = (h.hrv_overnight_avg_ms - h.hrv_7d_avg_ms) / h.hrv_7d_avg_ms;
    mod += delta * 15; // 10% HRV drop → -1.5 TSB
  }
  // Sleep quality
  if (h.sleep_score != null) {
    mod += (h.sleep_score - 70) / 10; // 70=neutral, 100→+3, 40→-3
  }
  // Body battery (Garmin readiness proxy)
  if (h.body_battery_high != null) {
    mod += (h.body_battery_high - 60) / 20; // 60=neutral
  }
  return Math.max(-20, Math.min(15, mod));
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { color: string; name: string; value: number }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-surface-card border border-surface-border rounded-xl p-3 text-xs space-y-1 min-w-[160px]">
      <div className="text-slate-400 mb-2 font-medium">{label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex justify-between gap-4">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="text-white font-semibold">{Math.round(p.value)}</span>
        </div>
      ))}
    </div>
  );
};

// Decay rate per day: 0.75 → modifier halves in ~2.4 days, gone by ~10 days.
// Mirrors typical physiological recovery from a bad night or hard block.
const HEALTH_DECAY = 0.75;
const HEALTH_MIN   = 0.3; // below this magnitude, snap to 0

export function FitnessChart({ data, healthDays }: Props) {
  const [showHealth, setShowHealth] = useState(true);

  // Build health lookup by YYYY-MM-DD
  const healthByDate = new Map(
    (healthDays ?? []).map((h) => [h.date.slice(0, 10), h])
  );

  // First pass: compute raw modifier for each day (null = no data)
  const withModifier = data
    .slice()
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((d) => {
      const h = healthByDate.get(d.date.slice(0, 10));
      return { ...d, rawMod: h != null ? healthModifier(h) : null };
    });

  // Second pass: carry modifier forward with exponential decay on gap days
  let prevMod = 0;
  const formatted = withModifier.map((d) => {
    let mod: number;
    if (d.rawMod != null) {
      mod = d.rawMod;          // anchor: real health measurement
    } else {
      mod = prevMod * HEALTH_DECAY;
      if (Math.abs(mod) < HEALTH_MIN) mod = 0;
    }
    prevMod = mod;
    return {
      ...d,
      date:        format(new Date(d.date), "MMM d"),
      has_health:  d.rawMod != null,
      tsb_display: +(d.tsb + mod).toFixed(1),
    };
  });

  const adjCount = formatted.filter((d) => d.has_health).length;
  const hasAdjusted = adjCount > 0;
  const showAdj = hasAdjusted && showHealth;

  return (
    <div>
      {hasAdjusted && (
        <div className="flex justify-end mb-2">
          <button
            onClick={() => setShowHealth((v) => !v)}
            className={cn(
              "flex items-center gap-1.5 text-xs px-3 py-1 rounded-full border transition-all",
              showHealth
                ? "border-purple-500/50 bg-purple-500/10 text-purple-300"
                : "border-surface-border text-slate-500 hover:text-slate-300"
            )}
          >
            <Heart className="w-3 h-3" />
            Health-adjusted form
            <span className="ml-1 opacity-60">({adjCount}d)</span>
          </button>
        </div>
      )}
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={formatted} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <defs>
            <linearGradient id="adjGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#c084fc" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#c084fc" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="ctlGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#60a5fa" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="atlGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#f87171" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="tsbGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#4ade80" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#4ade80" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2737" />
          <ReferenceLine y={0} stroke="#334155" strokeWidth={1} />
          <XAxis
            dataKey="date"
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickLine={false} axisLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#94a3b8", paddingTop: 8 }}
            formatter={(val) => <span style={{ color: "#94a3b8" }}>{val}</span>}
          />
          <Area type="monotone" dataKey="ctl" name="CTL (Fitness)" stroke="#60a5fa" fill="url(#ctlGrad)" strokeWidth={2} dot={false} />
          <Area type="monotone" dataKey="atl" name="ATL (Fatigue)" stroke="#f87171" fill="url(#atlGrad)" strokeWidth={2} dot={false} />
          {/* Form line — swaps to health-adjusted values when toggle is on */}
          <Area
            type="monotone"
            dataKey={showAdj ? "tsb_display" : "tsb"}
            name={showAdj ? "Form (health-adj)" : "TSB (Form)"}
            stroke={showAdj ? "#c084fc" : "#4ade80"}
            fill={showAdj ? "url(#adjGrad)" : "url(#tsbGrad)"}
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
          />
        </ComposedChart>
      </ResponsiveContainer>
      {showAdj && (
        <p className="text-xs text-slate-500 mt-1">
          Form line shifted by HRV, sleep &amp; body battery on {adjCount} days with health data.
        </p>
      )}
    </div>
  );
}
