"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import { format } from "date-fns";
import type { FitnessSnapshot } from "@/types";

interface Props {
  data: FitnessSnapshot[];
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { color: string; name: string; value: number }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-surface-card border border-surface-border rounded-xl p-3 text-xs space-y-1 min-w-[140px]">
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

export function FitnessChart({ data }: Props) {
  const formatted = data.map((d) => ({
    ...d,
    date: format(new Date(d.date), "MMM d"),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={formatted} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <defs>
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
        <Area type="monotone" dataKey="ctl" name="CTL (Fitness)"  stroke="#60a5fa" fill="url(#ctlGrad)" strokeWidth={2} dot={false} />
        <Area type="monotone" dataKey="atl" name="ATL (Fatigue)"  stroke="#f87171" fill="url(#atlGrad)" strokeWidth={2} dot={false} />
        <Area type="monotone" dataKey="tsb" name="TSB (Form)"     stroke="#4ade80" fill="url(#tsbGrad)" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
