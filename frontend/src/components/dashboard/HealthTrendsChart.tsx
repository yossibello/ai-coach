"use client";

import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Heart, Moon, Battery, Wind } from "lucide-react";
import { HealthDay } from "@/lib/api";

function fmt(d: string) {
  return new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function ChartCard({
  title, icon: Icon, iconColor, children, empty,
}: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  iconColor: string;
  children: React.ReactNode;
  empty?: boolean;
}) {
  return (
    <div className="bg-surface-card border border-surface-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Icon className={`w-4 h-4 ${iconColor}`} />
        <span className="text-sm font-medium text-slate-300">{title}</span>
      </div>
      {empty ? (
        <div className="h-[120px] flex items-center justify-center text-xs text-slate-500">
          No data for this period
        </div>
      ) : (
        <div className="h-[120px]">{children}</div>
      )}
    </div>
  );
}

const TOOLTIP_STYLE = {
  backgroundColor: "#0f172a",
  border: "1px solid #1e293b",
  borderRadius: 8,
  fontSize: 12,
  color: "#cbd5e1",
};

const DAY_MS = 86_400_000;

function buildTimeSeries(days: HealthDay[]) {
  if (!days.length) return [];
  const byDate = new Map(days.map((d) => [d.date.slice(0, 10), d]));
  const sorted = [...days].sort((a, b) => a.date.slice(0, 10).localeCompare(b.date.slice(0, 10)));
  const start = new Date(sorted[0].date.slice(0, 10)).getTime();
  const end   = new Date(sorted[sorted.length - 1].date.slice(0, 10)).getTime();
  const rows = [];
  for (let t = start; t <= end; t += DAY_MS) {
    const key = new Date(t).toISOString().slice(0, 10);
    const d = byDate.get(key);
    rows.push({
      ts:      t,
      hrv:     d?.hrv_overnight_avg_ms  ?? null,
      hrv7:    d?.hrv_7d_avg_ms         ?? null,
      rhr:     d?.resting_hr            ?? null,
      sleep:   d?.sleep_total_seconds != null ? +(d.sleep_total_seconds / 3600).toFixed(1) : null,
      battery: d?.body_battery_high     ?? null,
    });
  }
  return rows;
}

export default function HealthTrendsChart({ days }: { days: HealthDay[] }) {
  if (!days || days.length === 0) return null;

  const data = buildTimeSeries(days);
  const domain: [number, number] = [data[0].ts, data[data.length - 1].ts];
  const tickFmt = (ts: number) => new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });

  const hasHrv     = data.some((d) => d.hrv != null);
  const hasRhr     = data.some((d) => d.rhr != null);
  const hasSleep   = data.some((d) => d.sleep != null);
  const hasBattery = data.some((d) => d.battery != null);

  return (
    <div>
      <h2 className="text-base font-semibold text-white mb-3">Health Trends</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">

        {/* HRV */}
        <ChartCard title="Heart Rate Variability (ms)" icon={Wind} iconColor="text-violet-400" empty={!hasHrv}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <defs>
                <linearGradient id="gHrv" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="ts" type="number" scale="time" domain={domain} tickFormatter={tickFmt} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} tickCount={5} />
              <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(ts) => tickFmt(ts as number)} formatter={(v: number) => [`${v} ms`, "HRV"]} />
              <Area dataKey="hrv" stroke="#a78bfa" strokeWidth={2} fill="url(#gHrv)" dot={false} connectNulls />
              <Line dataKey="hrv7" stroke="#7c3aed" strokeWidth={1.5} strokeDasharray="4 2" dot={false} connectNulls name="7d avg" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Resting HR */}
        <ChartCard title="Resting Heart Rate (bpm)" icon={Heart} iconColor="text-rose-400" empty={!hasRhr}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <defs>
                <linearGradient id="gRhr" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#fb7185" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#fb7185" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="ts" type="number" scale="time" domain={domain} tickFormatter={tickFmt} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} tickCount={5} />
              <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} domain={["auto", "auto"]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(ts) => tickFmt(ts as number)} formatter={(v: number) => [`${v} bpm`, "RHR"]} />
              <Area dataKey="rhr" stroke="#fb7185" strokeWidth={2} fill="url(#gRhr)" dot={false} connectNulls />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Sleep */}
        <ChartCard title="Sleep" icon={Moon} iconColor="text-blue-400" empty={!hasSleep}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="ts" type="number" scale="time" domain={domain} tickFormatter={tickFmt} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} tickCount={5} />
              <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} unit="h" />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(ts) => tickFmt(ts as number)} formatter={(v: number) => [`${v}h`, "Sleep"]} />
              <Bar dataKey="sleep" fill="#60a5fa" opacity={0.8} radius={[3, 3, 0, 0]} maxBarSize={12} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Body Battery */}
        <ChartCard title="Body Battery" icon={Battery} iconColor="text-emerald-400" empty={!hasBattery}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <defs>
                <linearGradient id="gBat" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="ts" type="number" scale="time" domain={domain} tickFormatter={tickFmt} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} tickCount={5} />
              <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} domain={[0, 100]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(ts) => tickFmt(ts as number)} formatter={(v: number) => [`${v}`, "Body Battery"]} />
              <Area dataKey="battery" stroke="#34d399" strokeWidth={2} fill="url(#gBat)" dot={false} connectNulls />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

      </div>
    </div>
  );
}
