"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { profileAPI } from "@/lib/api";
import toast from "react-hot-toast";

// Defined OUTSIDE the page component so React never unmounts/remounts it on re-render.
function InputField({
  label,
  type = "text",
  placeholder,
  min,
  max,
  value,
  onChange,
}: {
  label: string;
  type?: string;
  placeholder?: string;
  min?: number;
  max?: number;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        min={min}
        max={max}
        className="w-full bg-surface-card border border-surface-border rounded-lg px-3 py-2
                   text-white placeholder-gray-500 focus:outline-none focus:ring-2
                   focus:ring-brand-500 transition"
      />
    </div>
  );
}

const GOALS = [
  { value: "general_fitness", label: "General fitness & health" },
  { value: "event_specific", label: "Train for a specific event" },
  { value: "ftp_improvement", label: "Improve FTP / power" },
  { value: "gran_fondo", label: "Gran Fondo / endurance" },
  { value: "weight_loss", label: "Weight loss" },
];

const EXPERIENCE = [
  { value: "beginner", label: "Beginner (< 1 year)" },
  { value: "intermediate", label: "Intermediate (1–3 years)" },
  { value: "advanced", label: "Advanced (3–7 years)" },
  { value: "elite", label: "Elite (7+ years / racer)" },
];

export default function ProfileSetupPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  const [form, setForm] = useState({
    age: "30",
    weight_kg: "70",
    height_cm: "177",
    sex: "male" as "male" | "female" | "other",
    ftp: "200",
    max_hr: "185",
    resting_hr: "55",
    cycling_experience_years: "2",
    primary_goal: "general_fitness" as import("@/types").GoalType,
    training_days_per_week: "4",
    goal_event_name: "",
    goal_event_date: "",
  });

  const update = (field: string, value: string) =>
    setForm((f) => ({ ...f, [field]: value }));

  const field = (name: string) => ({
    value: (form as Record<string, string>)[name],
    onChange: (v: string) => update(name, v),
  });

  const mutation = useMutation({
    mutationFn: () =>
      profileAPI.update({
        ...form,
        age: form.age ? parseInt(form.age) : undefined,
        weight_kg: form.weight_kg ? parseFloat(form.weight_kg) : undefined,
        height_cm: form.height_cm ? parseInt(form.height_cm) : undefined,
        ftp: form.ftp ? parseInt(form.ftp) : undefined,
        max_hr: form.max_hr ? parseInt(form.max_hr) : undefined,
        resting_hr: form.resting_hr ? parseInt(form.resting_hr) : undefined,
        cycling_experience_years: form.cycling_experience_years
          ? parseInt(form.cycling_experience_years)
          : undefined,
        training_days_per_week: parseInt(form.training_days_per_week),
        goal_event_date: form.goal_event_date || undefined,
        goal_event_name: form.goal_event_name || undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["profile"] });
      toast.success("Profile saved!");
      router.push("/dashboard");
    },
    onError: () => toast.error("Failed to save profile"),
  });

  return (
    <div className="min-h-screen bg-surface-bg flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-white mb-2">
            Set up your athlete profile
          </h1>
          <p className="text-gray-400">
            The AI uses this to personalise your training plan. You can update
            everything later.
          </p>
        </div>

        <div className="bg-surface-card border border-surface-border rounded-2xl p-8 space-y-8">
          {/* Physical */}
          <section>
            <h2 className="text-lg font-semibold text-white mb-4">Physical</h2>
            <div className="grid grid-cols-2 gap-4">
              <InputField label="Age" {...field("age")} type="number" min={10} max={100} />
              <InputField label="Weight (kg)" {...field("weight_kg")} type="number" min={30} max={250} />
              <InputField label="Height (cm)" {...field("height_cm")} type="number" min={100} max={230} />
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Sex
                </label>
                <select
                  value={form.sex}
                  onChange={(e) => update("sex", e.target.value)}
                  className="w-full bg-surface-card border border-surface-border rounded-lg px-3 py-2
                             text-white focus:outline-none focus:ring-2 focus:ring-brand-500"
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other / prefer not to say</option>
                </select>
              </div>
            </div>
          </section>

          {/* Performance */}
          <section>
            <h2 className="text-lg font-semibold text-white mb-4">Performance</h2>
            <div className="grid grid-cols-3 gap-4">
              <InputField label="FTP (watts)" {...field("ftp")} type="number" min={50} max={600} />
              <InputField label="Max HR (bpm)" {...field("max_hr")} type="number" min={100} max={230} />
              <InputField label="Resting HR (bpm)" {...field("resting_hr")} type="number" min={30} max={120} />
            </div>
          </section>

          {/* Training */}
          <section>
            <h2 className="text-lg font-semibold text-white mb-4">Training</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Primary goal
                </label>
                <select
                  value={form.primary_goal}
                  onChange={(e) => update("primary_goal", e.target.value)}
                  className="w-full bg-surface-card border border-surface-border rounded-lg px-3 py-2
                             text-white focus:outline-none focus:ring-2 focus:ring-brand-500"
                >
                  {GOALS.map((g) => (
                    <option key={g.value} value={g.value}>
                      {g.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Training days / week
                </label>
                <select
                  value={form.training_days_per_week}
                  onChange={(e) => update("training_days_per_week", e.target.value)}
                  className="w-full bg-surface-card border border-surface-border rounded-lg px-3 py-2
                             text-white focus:outline-none focus:ring-2 focus:ring-brand-500"
                >
                  {[2, 3, 4, 5, 6, 7].map((n) => (
                    <option key={n} value={n}>
                      {n} days
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {(form.primary_goal === "event_specific" || form.primary_goal === "gran_fondo" || form.primary_goal === "criterium") && (
              <div className="grid grid-cols-2 gap-4 mt-4">
                <InputField label="Event name" {...field("goal_event_name")} placeholder="Gran Fondo Alps" />
                <InputField label="Event date" {...field("goal_event_date")} type="date" />
              </div>
            )}
          </section>

          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="w-full bg-brand-500 hover:bg-brand-600 disabled:opacity-50
                       text-white font-semibold py-3 rounded-xl transition"
          >
            {mutation.isPending ? "Saving…" : "Save & go to dashboard →"}
          </button>
        </div>
      </div>
    </div>
  );
}
