"use client";

import { useQuery, useMutation } from "@tanstack/react-query";
import { profileAPI } from "@/lib/api";
import { useState, useEffect } from "react";
import type { AthleteProfile, GoalType } from "@/types";
import { User, Save, Loader2, Link2 } from "lucide-react";
import toast from "react-hot-toast";
import GarminConnect from "@/components/integrations/GarminConnect";
import StravaConnect from "@/components/integrations/StravaConnect";
import OuraConnect from "@/components/integrations/OuraConnect";
import FitbitConnect from "@/components/integrations/FitbitConnect";

const GOALS: { value: GoalType; label: string }[] = [
  { value: "general_fitness",  label: "General Fitness" },
  { value: "ftp_improvement",  label: "Improve FTP" },
  { value: "weight_loss",      label: "Weight Loss" },
  { value: "event_specific",   label: "Specific Event (Race / Sportive)" },
  { value: "gran_fondo",       label: "Gran Fondo" },
  { value: "criterium",        label: "Criterium / Road Race" },
  { value: "climbing",         label: "Climbing / Mountainous Event" },
  { value: "triathlon",        label: "Triathlon" },
];

const EVENT_TYPES: { value: NonNullable<AthleteProfile["event_type"]>; label: string; hint: string }[] = [
  { value: "long_road",       label: "Long road race",         hint: "4–7h hard pack racing" },
  { value: "crit",            label: "Criterium",              hint: "45–90 min, repeated max efforts" },
  { value: "tt",              label: "Time trial",             hint: "Sustained threshold/VO2" },
  { value: "stage_race",      label: "Stage race",             hint: "Multi-day, recovery between stages" },
  { value: "gran_fondo",      label: "Gran fondo",             hint: "4–8h endurance + climbs" },
  { value: "climbing_camp",   label: "Alpine / climbing camp", hint: "Multi-day big climbing volume" },
  { value: "mtb_marathon",    label: "MTB marathon",           hint: "Off-road 3–6h, variable power" },
  { value: "ultra_endurance", label: "Ultra-endurance",        hint: "8h+ steady aerobic" },
  { value: "triathlon_70_3",  label: "Triathlon 70.3",         hint: "~2.5h cycling leg" },
  { value: "triathlon_140_6", label: "Triathlon Ironman",      hint: "~5h cycling leg" },
];

export default function ProfilePage() {
  const { data, isLoading } = useQuery({
    queryKey: ["profile"],
    queryFn: () => profileAPI.get(),
  });

  const [form, setForm] = useState<Partial<AthleteProfile>>({});

  useEffect(() => {
    if (data) setForm(data);
  }, [data]);

  const save = useMutation({
    mutationFn: () => profileAPI.update(form),
    onSuccess: () => toast.success("Profile saved"),
    onError: () => toast.error("Failed to save profile"),
  });

  function set<K extends keyof AthleteProfile>(key: K, value: AthleteProfile[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  if (isLoading) {
    return (
      <div className="p-6 flex items-center justify-center h-60">
        <Loader2 className="w-6 h-6 text-brand-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <User className="w-6 h-6 text-brand-500" />
            Athlete Profile
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            The AI coach uses this to personalize your plan
          </p>
        </div>
        <button
          onClick={() => save.mutate()}
          disabled={save.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-brand-500 hover:bg-brand-600 disabled:opacity-60 text-white text-sm font-medium rounded-xl transition-colors"
        >
          {save.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          Save
        </button>
      </div>

      <Section title="Physical">
        <Row label="Age" hint="years">
          <NumInput value={form.age} onChange={(v) => set("age", v)} min={10} max={100} />
        </Row>
        <Row label="Weight" hint="kg">
          <NumInput value={form.weight_kg} onChange={(v) => set("weight_kg", v)} min={30} max={200} step={0.1} />
        </Row>
        <Row label="Height" hint="cm">
          <NumInput value={form.height_cm} onChange={(v) => set("height_cm", v)} min={100} max={250} />
        </Row>
        <Row label="Sex">
          <select
            value={form.sex ?? ""}
            onChange={(e) => set("sex", e.target.value as AthleteProfile["sex"])}
            className="input-field"
          >
            <option value="">Select</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </Row>
      </Section>

      <Section title="Performance">
        <Row label="FTP" hint="watts — Functional Threshold Power">
          <NumInput value={form.ftp} onChange={(v) => set("ftp", v)} min={50} max={600} />
        </Row>
        <Row label="Max Heart Rate" hint="bpm">
          <NumInput value={form.max_hr} onChange={(v) => set("max_hr", v)} min={100} max={230} />
        </Row>
        <Row label="Resting Heart Rate" hint="bpm">
          <NumInput value={form.resting_hr} onChange={(v) => set("resting_hr", v)} min={30} max={100} />
        </Row>
        <Row label="Cycling experience" hint="years">
          <NumInput value={form.cycling_experience_years} onChange={(v) => set("cycling_experience_years", v)} min={0} max={50} />
        </Row>
      </Section>

      <Section title="Training">
        <Row label="Training days / week">
          <NumInput value={form.training_days_per_week} onChange={(v) => set("training_days_per_week", v)} min={1} max={7} />
        </Row>
        <Row label="Strength approach" hint="How to integrate gym / bodyweight work">
          <select
            value={form.strength_approach ?? "friel"}
            onChange={(e) => set("strength_approach", e.target.value as AthleteProfile["strength_approach"])}
            className="input-field"
          >
            <option value="friel">Friel Periodized — phases match your cycling block (recommended)</option>
            <option value="minimum_dose">Minimum Dose — 5 patterns, same session 2×/week, one kettlebell</option>
            <option value="grease_the_groove">Grease the Groove — 5 reps every 1-2h, zero fatigue cost</option>
            <option value="none">No strength training</option>
          </select>
        </Row>
        <Row label="Primary Goal">
          <select
            value={form.primary_goal ?? ""}
            onChange={(e) => set("primary_goal", e.target.value as GoalType)}
            className="input-field"
          >
            <option value="">Select goal</option>
            {GOALS.map((g) => (
              <option key={g.value} value={g.value}>{g.label}</option>
            ))}
          </select>
        </Row>
        {form.primary_goal === "event_specific" || form.primary_goal?.includes("gran") || form.primary_goal === "climbing" || form.primary_goal === "criterium" || form.primary_goal === "triathlon" ? (
          <>
            <Row label="Event name">
              <input
                type="text"
                value={form.goal_event_name ?? ""}
                onChange={(e) => set("goal_event_name", e.target.value)}
                className="input-field"
                placeholder="e.g. Alpe d'HuZes 2026"
              />
            </Row>
            <Row label="Event date">
              <input
                type="date"
                value={form.goal_event_date ?? ""}
                onChange={(e) => set("goal_event_date", e.target.value)}
                className="input-field"
              />
            </Row>
            <Row label="Event type" hint="Tunes the AI plan to event demands">
              <select
                value={form.event_type ?? ""}
                onChange={(e) =>
                  set("event_type", (e.target.value || undefined) as AthleteProfile["event_type"])
                }
                className="input-field"
              >
                <option value="">Select event type</option>
                {EVENT_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label} — {t.hint}
                  </option>
                ))}
              </select>
            </Row>
          </>
        ) : null}
      </Section>

      <div className="bg-surface-card border border-surface-border rounded-2xl p-6">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-5 flex items-center gap-2">
          <Link2 className="w-4 h-4" />
          Integrations
        </h2>
        <div className="space-y-4">
          <GarminConnect />
          <StravaConnect />
          <OuraConnect />
          <FitbitConnect />
        </div>
      </div>

      <style jsx>{`
        .input-field {
          width: 100%;
          padding: 0.6rem 0.75rem;
          background: #0f1117;
          border: 1px solid #1e2737;
          border-radius: 0.5rem;
          color: #e2e8f0;
          font-size: 0.875rem;
          outline: none;
          transition: border-color 0.15s;
        }
        .input-field:focus {
          border-color: #22c55e;
        }
      `}</style>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface-card border border-surface-border rounded-2xl p-6">
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-5">{title}</h2>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-4">
      <div className="w-48 flex-shrink-0">
        <div className="text-sm text-slate-300">{label}</div>
        {hint && <div className="text-xs text-slate-500 mt-0.5">{hint}</div>}
      </div>
      <div className="flex-1">{children}</div>
    </div>
  );
}

function NumInput({
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  value?: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <input
      type="number"
      value={value ?? ""}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      min={min}
      max={max}
      step={step}
      className="w-full px-3 py-2 bg-surface border border-surface-border rounded-lg text-white text-sm focus:outline-none focus:border-brand-500 transition-colors"
    />
  );
}
