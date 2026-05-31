"use client";

import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface Zone {
  label: string;
  range: string;
  color: string;
  emoji: string;
  description: string;
}

interface MetricInfo {
  name: string;
  emoji: string;
  tagline: string;
  what: string;
  analogy: string;
  zones: Zone[];
  formula?: string;
}

const METRICS: Record<string, MetricInfo> = {
  FTP: {
    name: "FTP — Functional Threshold Power",
    emoji: "⚡",
    tagline: "Your ceiling",
    what: "The maximum power you can sustain for roughly one hour. Everything else is measured relative to it.",
    analogy: "It's your redline. Go above it and you'll blow up. Stay below and you're in control.",
    formula: "Typically: best 20-min power × 0.95",
    zones: [
      { label: "Getting started",   range: "< 150W",     color: "bg-slate-500",   emoji: "🐢", description: "Just beginning. Every watt counts." },
      { label: "Recreational",      range: "150–220W",   color: "bg-blue-500",    emoji: "🚲", description: "Weekend warrior territory." },
      { label: "Club racer",        range: "220–280W",   color: "bg-green-500",   emoji: "🐅", description: "You're actually fast." },
      { label: "Serious amateur",   range: "280–350W",   color: "bg-yellow-500",  emoji: "🦅", description: "People notice you at the front." },
      { label: "Elite",             range: "350W+",      color: "bg-red-500",     emoji: "🚀", description: "Quit your job and race." },
    ],
  },
  CTL: {
    name: "CTL — Chronic Training Load",
    emoji: "💪",
    tagline: "Your fitness",
    what: "A 42-day rolling average of your daily training stress. It rises slowly when you train consistently and falls when you rest. Think of it as your aerobic engine size.",
    analogy: "Your fitness bank account balance. Takes months to build, weeks to spend.",
    formula: "42-day exponential moving average of TSS",
    zones: [
      { label: "Couch mode",     range: "0–20",   color: "bg-slate-500",   emoji: "😴", description: "Netflix is getting more exercise than you." },
      { label: "Recreational",   range: "20–50",  color: "bg-blue-500",    emoji: "🚶", description: "Regular rides, decent base." },
      { label: "Committed",      range: "50–70",  color: "bg-green-500",   emoji: "🚴", description: "You take training seriously." },
      { label: "Racer",          range: "70–90",  color: "bg-yellow-500",  emoji: "🏆", description: "Podium territory on local races." },
      { label: "Pro-level",      range: "90+",    color: "bg-red-500",     emoji: "🌟", description: "How's the Tour treating you?" },
    ],
  },
  ATL: {
    name: "ATL — Acute Training Load",
    emoji: "🔥",
    tagline: "Your fatigue",
    what: "A 7-day rolling average of training stress. Spikes fast after hard days and drops quickly with rest. High ATL means your legs are cooked right now.",
    analogy: "The bar tab you ran up this week. The higher it is, the worse you'll feel tomorrow.",
    formula: "7-day exponential moving average of TSS",
    zones: [
      { label: "Fresh",         range: "0–30",   color: "bg-emerald-500", emoji: "😎", description: "Ready for anything." },
      { label: "Normal load",   range: "30–60",  color: "bg-blue-500",    emoji: "💪", description: "Training is happening." },
      { label: "Hard block",    range: "60–90",  color: "bg-amber-500",   emoji: "😓", description: "Legs are heavy. Normal." },
      { label: "Danger zone",   range: "90+",    color: "bg-red-500",     emoji: "🚨", description: "Take a rest day. Seriously." },
    ],
  },
  TSB: {
    name: "TSB — Training Stress Balance",
    emoji: "⚖️",
    tagline: "Your form",
    what: "CTL minus ATL. Negative means you're training hard and tired. Positive means you're rested and ready to race. The sweet spot is slightly negative for training, positive for race day.",
    analogy: "The needle on your 'are you ready to smash it?' gauge.",
    formula: "TSB = CTL − ATL",
    zones: [
      { label: "Overreaching ⚠️", range: "< −30",    color: "bg-red-500",     emoji: "🪦", description: "Back off before you get injured." },
      { label: "Hard training",    range: "−30 to −10", color: "bg-amber-500",  emoji: "😤", description: "Accumulating fitness. Stay the course." },
      { label: "Normal training",  range: "−10 to 0",  color: "bg-blue-500",    emoji: "😊", description: "Good training zone." },
      { label: "Race ready",       range: "0 to +20",  color: "bg-emerald-500", emoji: "🚀", description: "Taper worked. Go fast." },
      { label: "Too rested",       range: "> +20",     color: "bg-slate-500",   emoji: "😴", description: "You've been resting too long." },
    ],
  },
  TSS: {
    name: "TSS — Training Stress Score",
    emoji: "📊",
    tagline: "Today's workout score",
    what: "A per-workout score that combines duration and intensity. 100 TSS = exactly 1 hour at your FTP. Below FTP scores less; above FTP scores more.",
    analogy: "The receipt for your training session. How much did you spend today?",
    formula: "TSS = (seconds × NP × IF) / (FTP × 3600) × 100",
    zones: [
      { label: "Recovery ride",   range: "< 50",    color: "bg-slate-500",   emoji: "🛁", description: "Barely counts. Good for active recovery." },
      { label: "Easy day",        range: "50–80",   color: "bg-blue-500",    emoji: "😌", description: "Aerobic base building." },
      { label: "Solid session",   range: "80–120",  color: "bg-green-500",   emoji: "💪", description: "Quality work done." },
      { label: "Hard day",        range: "120–180", color: "bg-amber-500",   emoji: "😰", description: "You'll feel this tomorrow." },
      { label: "Epic",            range: "180+",    color: "bg-red-500",     emoji: "💀", description: "Rest tomorrow. Not optional." },
    ],
  },
};

interface Props {
  metric: keyof typeof METRICS;
  currentValue?: number;
  open: boolean;
  onClose: () => void;
}

export function MetricExplainer({ metric, currentValue, open, onClose }: Props) {
  const info = METRICS[metric];
  if (!info || !open) return null;

  return (
    <>
      {open && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center p-4" onClick={onClose}>
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
          <div
            className="relative w-full max-w-lg bg-surface-card border border-surface-border rounded-2xl shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between p-5 border-b border-surface-border">
              <div>
                <div className="text-2xl mb-1">{info.emoji}</div>
                <h2 className="font-bold text-white text-sm">{info.name}</h2>
                <p className="text-xs text-brand-400 font-medium mt-0.5">{info.tagline}</p>
              </div>
              <button onClick={onClose} className="text-slate-500 hover:text-white mt-1">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-4">
              {/* What it is */}
              <div>
                <p className="text-sm text-slate-300 leading-relaxed">{info.what}</p>
              </div>

              {/* Analogy */}
              <div className="bg-surface border border-surface-border rounded-xl p-3">
                <p className="text-xs text-slate-400 italic">💡 {info.analogy}</p>
              </div>

              {/* Formula */}
              {info.formula && (
                <div className="text-xs text-slate-500 font-mono bg-surface px-3 py-1.5 rounded-lg border border-surface-border">
                  {info.formula}
                </div>
              )}

              {/* Zones */}
              <div>
                <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Ranges</p>
                <div className="space-y-2">
                  {info.zones.map((z) => {
                    const isCurrent = currentValue != null && isInZone(z, currentValue);
                    return (
                      <div
                        key={z.label}
                        className={cn(
                          "flex items-center gap-3 rounded-lg px-3 py-2 border transition-all",
                          isCurrent
                            ? "border-brand-500/40 bg-brand-500/5"
                            : "border-transparent bg-surface"
                        )}
                      >
                        <div className={cn("w-2 h-2 rounded-full shrink-0", z.color)} />
                        <span className="text-lg shrink-0">{z.emoji}</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-medium text-slate-300">{z.label}</span>
                            {isCurrent && (
                              <span className="text-[10px] text-brand-400 border border-brand-500/30 px-1.5 py-0.5 rounded-full">You are here</span>
                            )}
                          </div>
                          <p className="text-[11px] text-slate-500 truncate">{z.description}</p>
                        </div>
                        <span className="text-xs text-slate-500 font-mono shrink-0">{z.range}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function toNum(s: string): number {
  // normalise Unicode minus (−) and en-dash to ASCII hyphen before parseFloat
  return parseFloat(s.replace(/[−–]/g, "-").replace(/[^-\d.]/g, ""));
}

function isInZone(zone: Zone, value: number): boolean {
  const r = zone.range;
  if (r.startsWith("<"))  return value < toNum(r.slice(1));
  if (r.startsWith(">"))  return value > toNum(r.slice(1));
  const parts = r.split(/\s*(?:to|–|−)\s*/);
  if (parts.length === 2) return value >= toNum(parts[0]) && value <= toNum(parts[1]);
  return false;
}
