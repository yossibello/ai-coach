"use client";

import { useQuery } from "@tanstack/react-query";
import { fitnessAPI } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Trophy, Mountain, Bike, Zap, Timer, TrendingUp, Lock } from "lucide-react";

const CATEGORY_META: Record<string, { label: string; icon: React.ReactNode; color: string }> = {
  group_ride:  { label: "Group Rides",   icon: <Bike className="w-4 h-4" />,     color: "text-blue-400 border-blue-500/20 bg-blue-500/5"  },
  gran_fondo:  { label: "Gran Fondos",   icon: <Mountain className="w-4 h-4" />, color: "text-purple-400 border-purple-500/20 bg-purple-500/5" },
  gravel:      { label: "Gravel Races",  icon: <Zap className="w-4 h-4" />,      color: "text-amber-400 border-amber-500/20 bg-amber-500/5" },
  ultra:       { label: "Ultra Distance",icon: <Timer className="w-4 h-4" />,    color: "text-rose-400 border-rose-500/20 bg-rose-500/5"  },
};

const TIER_STYLE: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  gold:   { bg: "bg-yellow-500/10", border: "border-yellow-500/40", text: "text-yellow-300", glow: "shadow-yellow-500/20 shadow-lg" },
  silver: { bg: "bg-slate-400/10",  border: "border-slate-400/40",  text: "text-slate-300",  glow: "" },
  bronze: { bg: "bg-amber-700/10",  border: "border-amber-700/40",  text: "text-amber-600",  glow: "" },
};

interface Event {
  id: string;
  name: string;
  category: string;
  icon: string;
  description: string;
  distance_km: number | null;
  elevation_m: number | null;
  tier: string | null;
  tier_emoji: string;
  tier_name: string;
  next_tier: string | null;
  next_tier_name: string | null;
  event_speed_kmh: number | null;   // what gold demands
  athlete_speed_kmh: number | null; // what the athlete can do
  est_time_h: number | null;
  athlete_wkg: number;
  athlete_flat_kmh: number;
  athlete_max_dur_h: number;
}

interface CapabilityData {
  athlete: { ftp_w: number; weight_kg: number; wkg: number; ctl: number };
  events: Event[];
}

function formatTime(h: number): string {
  const hrs = Math.floor(h);
  const mins = Math.round((h - hrs) * 60);
  return mins > 0 ? `${hrs}h ${mins}m` : `${hrs}h`;
}

function EventCard({ ev }: { ev: Event }) {
  const tierStyle = ev.tier ? TIER_STYLE[ev.tier] : null;
  const catMeta   = CATEGORY_META[ev.category];

  return (
    <div className={cn(
      "rounded-2xl border p-5 flex flex-col gap-3 transition-all",
      tierStyle
        ? `${tierStyle.bg} ${tierStyle.border} ${tierStyle.glow}`
        : "bg-surface-card border-surface-border opacity-70"
    )}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{ev.icon}</span>
          <div>
            <h3 className="font-semibold text-white text-sm leading-tight">{ev.name}</h3>
            <span className={cn("text-[10px] px-2 py-0.5 rounded-full border font-medium", catMeta?.color)}>
              {catMeta?.label}
            </span>
          </div>
        </div>
        {/* Medal */}
        <div className="text-right shrink-0">
          {ev.tier ? (
            <>
              <div className={cn("text-2xl leading-none", tierStyle?.text)}>{ev.tier_emoji}</div>
              <div className={cn("text-[10px] font-bold mt-0.5", tierStyle?.text)}>
                {ev.tier.toUpperCase()}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-0.5">
              <Lock className="w-5 h-5 text-slate-600" />
              <span className="text-[10px] text-slate-600">Locked</span>
            </div>
          )}
        </div>
      </div>

      {/* Event stats */}
      <div className="flex flex-wrap gap-3 text-xs text-slate-500">
        {ev.distance_km && <span>📍 {ev.distance_km}km</span>}
        {ev.elevation_m && <span>⬆️ {ev.elevation_m.toLocaleString()}m</span>}
        {ev.est_time_h  && <span>⏱ est. {formatTime(ev.est_time_h)}</span>}
        {ev.event_speed_kmh && (
          <span>🏁 needs {ev.event_speed_kmh} km/h</span>
        )}
        {ev.athlete_speed_kmh && ev.category === "group_ride" && (
          <span className={cn(
            "font-medium",
            ev.athlete_speed_kmh >= (ev.event_speed_kmh ?? 0) ? "text-emerald-400" : "text-amber-400"
          )}>
            you: {ev.athlete_speed_kmh} km/h
          </span>
        )}
      </div>

      {/* Achievement */}
      <div className={cn(
        "rounded-xl px-3 py-2 text-sm font-medium",
        ev.tier ? `${tierStyle?.bg} ${tierStyle?.text}` : "bg-slate-800 text-slate-500"
      )}>
        {ev.tier_emoji} {ev.tier_name}
      </div>

      {/* Description */}
      <p className="text-xs text-slate-500 leading-relaxed">{ev.description}</p>

      {/* Next tier unlock */}
      {ev.next_tier && (
        <div className="border-t border-surface-border pt-2 mt-1">
          <p className="text-[11px] text-slate-500">
            <TrendingUp className="w-3 h-3 inline mr-1 text-brand-400" />
            Next: <span className="text-brand-400 font-medium">{ev.next_tier_name}</span>
          </p>
        </div>
      )}
    </div>
  );
}

function StatPill({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-surface-card border border-surface-border rounded-xl px-4 py-3 text-center">
      <div className="text-xs text-slate-500 mb-0.5">{label}</div>
      <div className="text-lg font-bold text-white">{value}</div>
      {sub && <div className="text-[10px] text-slate-600">{sub}</div>}
    </div>
  );
}

export default function CapabilitiesPage() {
  const { data, isLoading } = useQuery<CapabilityData>({
    queryKey: ["capabilities"],
    queryFn:  () => fitnessAPI.getCapabilities(),
    staleTime: 10 * 60 * 1000,
  });

  const categories = ["group_ride", "gran_fondo", "gravel", "ultra"];

  if (isLoading) {
    return (
      <div className="p-6 flex items-center justify-center h-[60vh]">
        <div className="text-center space-y-2">
          <Trophy className="w-10 h-10 text-yellow-500 animate-pulse mx-auto" />
          <p className="text-slate-400 text-sm">Calculating your capabilities…</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { athlete, events } = data;
  const gold   = events.filter((e) => e.tier === "gold").length;
  const silver = events.filter((e) => e.tier === "silver").length;
  const bronze = events.filter((e) => e.tier === "bronze").length;

  return (
    <div className="p-4 space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Trophy className="w-6 h-6 text-yellow-400" />
          Capability Predictor
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          What you can do right now, based on your FTP, W/kg and fitness.
        </p>
      </div>

      {/* Athlete stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatPill label="FTP"    value={`${athlete.ftp_w}W`}       sub="threshold power" />
        <StatPill label="W/kg"   value={`${athlete.wkg.toFixed(2)}`} sub="power to weight" />
        <StatPill label="CTL"    value={`${athlete.ctl}`}           sub="fitness" />
        <StatPill label="Medals" value={`🥇${gold} 🥈${silver} 🥉${bronze}`} sub="unlocked" />
      </div>

      {/* Events by category */}
      {categories.map((cat) => {
        const catEvents = events.filter((e) => e.category === cat);
        if (!catEvents.length) return null;
        const meta = CATEGORY_META[cat];
        return (
          <section key={cat}>
            <div className={cn("flex items-center gap-2 mb-3 text-sm font-semibold px-1", meta?.color.split(" ")[0])}>
              {meta?.icon}
              {meta?.label}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {catEvents.map((ev) => <EventCard key={ev.id} ev={ev} />)}
            </div>
          </section>
        );
      })}

      <p className="text-xs text-slate-600 text-center pb-4">
        Estimates based on cycling physics models and published race data.
        Actual performance depends on terrain, weather, nutrition and race experience.
      </p>
    </div>
  );
}
