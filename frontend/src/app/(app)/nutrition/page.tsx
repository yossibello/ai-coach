"use client";

import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  FlaskConical,
  Upload,
  Trash2,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  Pill,
  TrendingUp,
  Info,
  Loader2,
  ClipboardList,
  Plus,
  Pencil,
  Square,
  ShieldAlert,
  CalendarDays,
  ChevronDown,
  ChevronUp,
  Activity,
} from "lucide-react";
import toast from "react-hot-toast";
import {
  nutritionAPI,
  trackingAPI,
  doseLogAPI,
  type BloodTest,
  type SupplementItem,
  type SupplementWarning,
  type MarkerTimeSeries,
  type SupplementIntakeRecord,
  type DoseLogRecord,
  type PerformanceTestRecord,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type Tab = "tests" | "supplements" | "trends" | "log";

export default function NutritionPage() {
  const [tab, setTab] = useState<Tab>("supplements");

  // Keep supplements query alive at the page level so that when a dose is deleted
  // from the Log tab, invalidateQueries(["supplements"]) triggers an *immediate*
  // refetch (active subscriber) instead of just marking stale until the tab remounts.
  useQuery({
    queryKey: ["supplements"],
    queryFn: () => nutritionAPI.getSupplements(),
    staleTime: 0,
  });

  return (
    <div className="p-6 space-y-6 animate-fade-in max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <FlaskConical className="w-6 h-6 text-brand-500" />
            Nutrition &amp; Supplements
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Evidence-based supplement guidance from your training load and (optional) blood markers.
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-surface-card rounded-lg border border-surface-border w-fit">
        <TabBtn active={tab === "supplements"} onClick={() => setTab("supplements")}>
          <Pill className="w-4 h-4" /> Today&apos;s Stack
        </TabBtn>
        <TabBtn active={tab === "tests"} onClick={() => setTab("tests")}>
          <Upload className="w-4 h-4" /> Blood Tests
        </TabBtn>
        <TabBtn active={tab === "trends"} onClick={() => setTab("trends")}>
          <TrendingUp className="w-4 h-4" /> Marker Trends
        </TabBtn>
        <TabBtn active={tab === "log"} onClick={() => setTab("log")}>
          <ClipboardList className="w-4 h-4" /> My Log
        </TabBtn>
      </div>

      {tab === "supplements" && <SupplementsTab />}
      {tab === "tests" && <BloodTestsTab />}
      {tab === "trends" && <TrendsTab />}
      {tab === "log" && <LogTab />}
    </div>
  );
}

function TabBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all",
        active
          ? "bg-brand-500/15 text-brand-400 border border-brand-500/20"
          : "text-slate-400 hover:text-white"
      )}
    >
      {children}
    </button>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Supplements tab
// ════════════════════════════════════════════════════════════════════════════
function SupplementsTab() {
  const qc = useQueryClient();
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["supplements"],
    queryFn: () => nutritionAPI.getSupplements(),
    staleTime: 0,   // always fetch fresh on mount so dose changes reflect immediately
  });
  // Refetch every 90 s so taken_today status stays live
  const { data: intakes } = useQuery({
    queryKey: ["intakes"],
    queryFn: () => trackingAPI.listIntakes(),
    refetchInterval: 90_000,
  });
  const activeKeys = new Set(
    (intakes ?? []).filter((i) => i.is_active).map((i) => i.supplement_key)
  );

  const refresh = useMutation({
    mutationFn: () => nutritionAPI.refreshSupplements(),
    onSuccess: (d) => {
      qc.setQueryData(["supplements"], d);
      qc.invalidateQueries({ queryKey: ["supplements"] });
      toast.success("Stack refreshed");
    },
    onError: () => toast.error("Could not refresh stack"),
  });

  if (isLoading) return <Loading label="Computing your stack…" />;
  if (!data) return <Empty msg="No supplement data yet." />;
  // isFetching=true means a background refetch is in progress (stale data showing)
  const isRefreshing = isFetching && !isLoading;

  const { stack, warnings, has_blood_test, is_cold_start, disclaimer } = data.payload;

  return (
    <div className="space-y-5">
      {/* Status banner */}
      <div className="flex items-center justify-between p-4 rounded-lg bg-surface-card border border-surface-border">
        <div className="flex items-center gap-3">
          {has_blood_test ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          ) : (
            <Info className="w-5 h-5 text-amber-400" />
          )}
          <div>
            <p className="text-sm font-medium text-white">
              {has_blood_test
                ? "Personalised using your latest blood test"
                : "General athlete recommendations (no blood test on file)"}
            </p>
            <p className="text-xs text-slate-400 mt-0.5">
              Engine: {data.engine_version} ·{" "}
              {is_cold_start ? "cold-start" : "data-driven"}
            </p>
          </div>
        </div>
        <button
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending || isRefreshing}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm bg-surface-muted text-slate-300 hover:text-white border border-surface-border disabled:opacity-50"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", (refresh.isPending || isRefreshing) && "animate-spin")} />
          {isRefreshing ? "Syncing…" : "Refresh"}
        </button>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((w) => (
            <WarningCard key={w.warning_key} warning={w} />
          ))}
        </div>
      )}

      {/* Stack */}
      {stack.length === 0 ? (
        <div className="p-8 rounded-lg bg-surface-card border border-surface-border text-center">
          <Pill className="w-8 h-8 text-slate-500 mx-auto mb-2" />
          <p className="text-slate-300">No supplements indicated right now.</p>
          <p className="text-slate-500 text-xs mt-1">
            Your training load and (where available) blood markers don&apos;t suggest needs above
            our inclusion threshold.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {Array.from(
            new Map(stack.map((s) => [s.supplement_key, s])).values()
          ).map((s) => (
            <SupplementCard
              key={s.supplement_key}
              item={s}
              alreadyLogged={activeKeys.has(s.supplement_key)}
            />
          ))}
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-xs text-slate-500 leading-relaxed pt-2 border-t border-surface-border">
        {disclaimer}
      </p>
    </div>
  );
}

// ─── Professional Supplement Card ────────────────────────────────────────────
function SupplementCard({
  item,
  alreadyLogged,
}: {
  item: SupplementItem;
  alreadyLogged: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [showDoseForm, setShowDoseForm] = useState(false);
  const [doseInput, setDoseInput] = useState(String(item.dose ?? ""));
  const [doseNotes, setDoseNotes] = useState("");
  const qc = useQueryClient();

  // Enrollment (start tracking) — creates intake record
  const enroll = useMutation({
    mutationFn: async () => {
      try {
        return await trackingAPI.createIntakeFromRecommendation(item.supplement_key);
      } catch {
        return await trackingAPI.createIntake({
          supplement_key: item.supplement_key,
          label: item.label,
          dose: item.dose,
          dose_unit: item.dose_unit,
          frequency: item.frequency,
          timing: item.timing,
        });
      }
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["intakes"] }),
    onError: () => {},   // handled below
  });

  // Log a dose — creates a dose_log entry
  const logDose = useMutation({
    mutationFn: async (dose: number) => {
      // If not yet enrolled, enroll first (ignore 409)
      if (!alreadyLogged) {
        try { await enroll.mutateAsync(); } catch { /* 409 OK */ }
      }
      return doseLogAPI.create({
        supplement_key: item.supplement_key,
        label: item.label,
        dose_taken: dose,
        dose_unit: item.dose_unit ?? undefined,
        notes: doseNotes || undefined,
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["intakes"] });
      qc.invalidateQueries({ queryKey: ["dose-logs"] });
      qc.invalidateQueries({ queryKey: ["supplements"] });
      toast.success(`Logged ${doseInput} ${item.dose_unit} ${item.label}`);
      setShowDoseForm(false);
      setDoseNotes("");
    },
    onError: () => toast.error("Could not log dose"),
  });

  const handleLogDose = () => {
    const n = parseFloat(doseInput);
    if (!n || n <= 0) { toast.error("Enter a valid dose amount"); return; }
    logDose.mutate(n);
  };

  const gradeColor =
    item.evidence_grade === "A"
      ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
      : item.evidence_grade === "B"
      ? "bg-sky-500/15 text-sky-400 border-sky-500/30"
      : "bg-amber-500/15 text-amber-400 border-amber-500/30";

  const takenPct =
    item.taken_today && item.dose
      ? Math.min(100, ((item.today_total_dose ?? 0) / item.dose) * 100)
      : 0;

  return (
    <div
      className={cn(
        "rounded-xl border bg-surface-card overflow-hidden",
        item.dose_exceeded
          ? "border-red-500/40"
          : item.taken_today
          ? "border-emerald-500/30"
          : "border-surface-border"
      )}
    >
      {/* Over-dose warning banner */}
      {item.dose_exceeded && item.dose_warning && (
        <div className="flex items-center gap-2 px-4 py-2 bg-red-500/10 border-b border-red-500/30">
          <ShieldAlert className="w-4 h-4 text-red-400 flex-shrink-0" />
          <p className="text-xs text-red-300">{item.dose_warning}</p>
        </div>
      )}

      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          {/* Left: name + metadata */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-semibold text-white">{item.label}</h3>
              <span className={cn("text-[10px] px-2 py-0.5 rounded border font-bold", gradeColor)}>
                GRADE {item.evidence_grade}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-surface-muted text-slate-400 border border-surface-border">
                {item.category}
              </span>
            </div>

            <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
              <Stat label="Dose" value={`${item.dose} ${item.dose_unit}`} />
              <Stat label="Frequency" value={item.frequency} />
              <Stat label="Timing" value={item.timing} />
              <Stat label="Duration" value={item.duration} />
            </div>

            {/* Today's progress bar */}
            {item.taken_today && (
              <div className="mt-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] uppercase tracking-wide text-slate-500">
                    Today&apos;s intake
                  </span>
                  <span
                    className={cn(
                      "text-xs font-medium",
                      item.dose_exceeded ? "text-red-400" : "text-emerald-400"
                    )}
                  >
                    {item.today_total_dose} / {item.dose} {item.dose_unit}
                  </span>
                </div>
                <div className="h-1.5 bg-surface-muted rounded-full overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all",
                      item.dose_exceeded ? "bg-red-500" : "bg-emerald-500"
                    )}
                    style={{ width: `${Math.min(takenPct, 100)}%` }}
                  />
                </div>
              </div>
            )}

            <p className="mt-3 text-sm text-slate-300 leading-relaxed">{item.rationale}</p>

            {item.warnings.length > 0 && (
              <div className="mt-3 space-y-1">
                {item.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-amber-400 flex items-start gap-1.5">
                    <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                    <span>{w}</span>
                  </p>
                ))}
              </div>
            )}

            <button
              onClick={() => setExpanded((v) => !v)}
              className="mt-3 text-xs text-brand-400 hover:text-brand-300 flex items-center gap-1"
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {expanded ? "Hide" : "Show"} citations &amp; trigger
            </button>
          </div>

          {/* Right: action button */}
          <div className="flex flex-col items-end gap-2 flex-shrink-0">
            {item.taken_today && !item.dose_exceeded ? (
              <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 whitespace-nowrap">
                <CheckCircle2 className="w-3.5 h-3.5" /> Taken today
              </span>
            ) : null}
            <button
              onClick={() => {
                setDoseInput(String(item.dose ?? ""));
                setShowDoseForm((v) => !v);
              }}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border whitespace-nowrap",
                item.dose_exceeded
                  ? "bg-amber-500/10 text-amber-400 border-amber-500/30 hover:bg-amber-500/20"
                  : "bg-brand-500/10 text-brand-400 border-brand-500/30 hover:bg-brand-500/20"
              )}
            >
              <Plus className="w-3.5 h-3.5" />
              {item.dose_exceeded ? "Log anyway" : "Log a dose"}
            </button>
          </div>
        </div>

        {/* Inline dose form */}
        {showDoseForm && (
          <div className="mt-3 p-3 rounded-lg bg-surface-muted border border-surface-border space-y-2">
            <p className="text-xs text-slate-400">
              Recommended: <strong className="text-white">{item.dose} {item.dose_unit}</strong> ·
              Adjust if you took more or less
            </p>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="0.1"
                min="0.01"
                value={doseInput}
                onChange={(e) => setDoseInput(e.target.value)}
                className="w-24 px-2 py-1.5 rounded bg-surface-card border border-surface-border text-white text-sm focus:outline-none focus:border-brand-500"
              />
              <span className="text-sm text-slate-400">{item.dose_unit}</span>
              <input
                type="text"
                placeholder="Notes (optional)"
                value={doseNotes}
                onChange={(e) => setDoseNotes(e.target.value)}
                className="flex-1 px-2 py-1.5 rounded bg-surface-card border border-surface-border text-white text-sm placeholder-slate-600 focus:outline-none focus:border-brand-500"
              />
              <button
                onClick={handleLogDose}
                disabled={logDose.isPending}
                className="px-3 py-1.5 rounded bg-brand-500 hover:bg-brand-400 text-white text-xs font-medium disabled:opacity-50 flex items-center gap-1"
              >
                {logDose.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
                Save
              </button>
              <button
                onClick={() => setShowDoseForm(false)}
                className="px-2 py-1.5 rounded text-slate-400 hover:text-white text-xs"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Citations */}
        {expanded && (
          <div className="mt-3 space-y-1.5 text-xs text-slate-400 border-t border-surface-border pt-3">
            <div>
              <span className="text-slate-500">Triggered by: </span>
              {item.triggered_by.join(", ")}
            </div>
            <div>
              <span className="text-slate-500">Score: </span>
              {item.score.toFixed(2)}
            </div>
            <div>
              <span className="text-slate-500">Citations:</span>
              <ul className="mt-0.5 ml-4 list-disc space-y-0.5">
                {item.citations.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-sm text-white">{value}</div>
    </div>
  );
}

function WarningCard({ warning }: { warning: SupplementWarning }) {
  return (
    <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/30">
      <div className="flex items-start gap-2.5">
        <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-sm text-amber-300 leading-relaxed">{warning.message}</p>
          <p className="text-[11px] text-slate-500 mt-1">
            Applies to: {warning.applies_to.join(", ")}
          </p>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Blood Tests tab
// ════════════════════════════════════════════════════════════════════════════
function BloodTestsTab() {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);

  const { data: tests, isLoading } = useQuery({
    queryKey: ["blood-tests"],
    queryFn: () => nutritionAPI.listBloodTests(),
  });

  const upload = useMutation({
    mutationFn: (f: File) => nutritionAPI.uploadBloodTestPDF(f),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["blood-tests"] });
      qc.invalidateQueries({ queryKey: ["supplements"] });
      toast.success("Blood test parsed and saved");
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      toast.error(e?.response?.data?.detail ?? "Upload failed"),
  });

  const del = useMutation({
    mutationFn: (id: string) => nutritionAPI.deleteBloodTest(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["blood-tests"] });
      qc.invalidateQueries({ queryKey: ["supplements"] });
      toast.success("Blood test deleted");
    },
  });

  return (
    <div className="space-y-5">
      {/* Upload area */}
      <div className="p-6 rounded-lg bg-surface-card border border-dashed border-surface-border">
        <div className="flex flex-col items-center text-center gap-3">
          <Upload className="w-8 h-8 text-slate-500" />
          <div>
            <p className="text-white font-medium">Upload a lab-report PDF</p>
            <p className="text-slate-400 text-xs mt-1">
              We&apos;ll extract markers automatically. Image-only / scanned PDFs aren&apos;t supported yet.
            </p>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) upload.mutate(f);
              if (fileRef.current) fileRef.current.value = "";
            }}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={upload.isPending}
            className="px-4 py-2 rounded-md text-sm font-medium bg-brand-500 hover:bg-brand-400 text-white disabled:opacity-50 flex items-center gap-2"
          >
            {upload.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            Choose PDF
          </button>
        </div>
      </div>

      {/* Tests list */}
      {isLoading ? (
        <Loading label="Loading tests…" />
      ) : !tests || tests.length === 0 ? (
        <Empty msg="No blood tests uploaded yet." />
      ) : (
        <div className="space-y-3">
          {tests.map((t) => (
            <BloodTestRow key={t.id} test={t} onDelete={() => del.mutate(t.id)} />
          ))}
        </div>
      )}
    </div>
  );
}

function BloodTestRow({ test, onDelete }: { test: BloodTest; onDelete: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const markerCount = Object.keys(test.markers).length;
  const flagged = Object.values(test.markers).filter(
    (m) => !["optimal", "unknown"].includes(m.status)
  ).length;

  return (
    <div className="rounded-lg bg-surface-card border border-surface-border overflow-hidden">
      <div className="p-4 flex items-center justify-between gap-3">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex-1 text-left flex items-center gap-3"
        >
          <FlaskConical className="w-4 h-4 text-brand-500" />
          <div>
            <p className="text-sm text-white font-medium">
              {new Date(test.test_date).toLocaleDateString()} ·{" "}
              {test.lab_name ?? "Unknown lab"}
            </p>
            <p className="text-xs text-slate-400 mt-0.5">
              {markerCount} markers · {flagged} flagged · {test.source}
              {test.parser_confidence != null && (
                <> · parser conf {(test.parser_confidence * 100).toFixed(0)}%</>
              )}
            </p>
          </div>
        </button>
        <button
          onClick={onDelete}
          className="p-2 rounded-md text-slate-500 hover:text-red-400 hover:bg-red-500/10"
          title="Delete"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {expanded && (
        <div className="border-t border-surface-border p-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {Object.entries(test.markers).map(([key, m]) => (
            <MarkerChip key={key} markerKey={key} marker={m} />
          ))}
        </div>
      )}
    </div>
  );
}

function MarkerChip({
  markerKey,
  marker,
}: {
  markerKey: string;
  marker: BloodTest["markers"][string];
}) {
  const color = statusColor(marker.status);
  return (
    <div className="p-2.5 rounded-md bg-surface-muted border border-surface-border">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-slate-400 truncate">{marker.label}</span>
        <span className={cn("text-[9px] uppercase font-bold", color)}>
          {marker.status.replace("_", " ")}
        </span>
      </div>
      <div className="mt-1 text-sm text-white">
        {marker.value} <span className="text-xs text-slate-500">{marker.unit}</span>
      </div>
      <div className="text-[10px] text-slate-500 mt-0.5">
        ref {marker.ref_low}–{marker.ref_high}
      </div>
    </div>
  );
}

function statusColor(status: string): string {
  switch (status) {
    case "optimal":
      return "text-emerald-400";
    case "suboptimal":
      return "text-amber-400";
    case "low":
    case "high":
      return "text-orange-400";
    case "critical_low":
    case "critical_high":
      return "text-red-400";
    default:
      return "text-slate-500";
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Trends tab
// ════════════════════════════════════════════════════════════════════════════
const TRENDABLE_MARKERS = [
  { key: "ferritin", label: "Ferritin" },
  { key: "vitamin_d", label: "Vitamin D" },
  { key: "vitamin_b12", label: "Vitamin B12" },
  { key: "magnesium", label: "Magnesium" },
  { key: "hemoglobin", label: "Hemoglobin" },
  { key: "testosterone_total", label: "Testosterone" },
  { key: "cortisol", label: "Cortisol" },
  { key: "tsh", label: "TSH" },
  { key: "crp", label: "CRP" },
  { key: "hba1c", label: "HbA1c" },
  { key: "ck", label: "Creatine Kinase" },
  { key: "homocysteine", label: "Homocysteine" },
];

function TrendsTab() {
  const [selected, setSelected] = useState<string>("ferritin");
  const { data, isLoading } = useQuery({
    queryKey: ["marker-series", selected],
    queryFn: () => nutritionAPI.getMarkerSeries(selected),
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {TRENDABLE_MARKERS.map((m) => (
          <button
            key={m.key}
            onClick={() => setSelected(m.key)}
            className={cn(
              "px-3 py-1.5 rounded-md text-xs font-medium border transition-all",
              selected === m.key
                ? "bg-brand-500/15 text-brand-400 border-brand-500/30"
                : "bg-surface-card text-slate-400 border-surface-border hover:text-white"
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <Loading label="Loading…" />
      ) : !data || data.points.length === 0 ? (
        <Empty msg="No data points for this marker yet. Upload a blood test that includes it." />
      ) : (
        <MarkerChart series={data} />
      )}
    </div>
  );
}

function MarkerChart({ series }: { series: MarkerTimeSeries }) {
  const { points, label, unit } = series;
  const values = points.map((p) => p.value);
  const refLows = points.map((p) => p.ref_low);
  const refHighs = points.map((p) => p.ref_high);
  const yMin = Math.min(...values, ...refLows) * 0.9;
  const yMax = Math.max(...values, ...refHighs) * 1.1;
  const w = 720;
  const h = 240;
  const pad = { l: 40, r: 16, t: 16, b: 28 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;

  const x = (i: number) =>
    pad.l + (points.length === 1 ? innerW / 2 : (i * innerW) / (points.length - 1));
  const y = (v: number) =>
    pad.t + innerH - ((v - yMin) / (yMax - yMin)) * innerH;

  const refLowY = y(refLows[0]);
  const refHighY = y(refHighs[0]);
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(p.value)}`).join(" ");

  return (
    <div className="p-4 rounded-lg bg-surface-card border border-surface-border">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-white">
          {label} <span className="text-slate-500 text-xs">({unit})</span>
        </h3>
        <p className="text-xs text-slate-500">
          {points.length} data point{points.length === 1 ? "" : "s"}
        </p>
      </div>

      <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="overflow-visible">
        {/* Reference range band */}
        <rect
          x={pad.l}
          y={refHighY}
          width={innerW}
          height={refLowY - refHighY}
          fill="rgb(16 185 129 / 0.08)"
        />
        <line
          x1={pad.l} x2={pad.l + innerW} y1={refLowY} y2={refLowY}
          stroke="rgb(100 116 139 / 0.4)" strokeDasharray="4 4"
        />
        <line
          x1={pad.l} x2={pad.l + innerW} y1={refHighY} y2={refHighY}
          stroke="rgb(100 116 139 / 0.4)" strokeDasharray="4 4"
        />

        {/* Y-axis labels */}
        <text x={pad.l - 6} y={refLowY + 4} fill="rgb(148 163 184)" fontSize="10" textAnchor="end">
          {refLows[0]}
        </text>
        <text x={pad.l - 6} y={refHighY + 4} fill="rgb(148 163 184)" fontSize="10" textAnchor="end">
          {refHighs[0]}
        </text>

        {/* Line */}
        <path d={linePath} stroke="rgb(34 211 238)" strokeWidth="2" fill="none" />

        {/* Points */}
        {points.map((p, i) => (
          <g key={i}>
            <circle
              cx={x(i)} cy={y(p.value)}
              r="4"
              fill={
                p.status === "optimal"
                  ? "rgb(16 185 129)"
                  : p.status === "suboptimal"
                  ? "rgb(245 158 11)"
                  : p.status.startsWith("critical")
                  ? "rgb(239 68 68)"
                  : "rgb(249 115 22)"
              }
              stroke="rgb(15 23 42)" strokeWidth="2"
            />
            <text
              x={x(i)} y={h - 8}
              fill="rgb(148 163 184)" fontSize="10" textAnchor="middle"
            >
              {new Date(p.test_date).toLocaleDateString(undefined, {
                year: "2-digit", month: "short",
              })}
            </text>
          </g>
        ))}
      </svg>

      <div className="mt-3 text-xs text-slate-500 flex items-center gap-3">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-emerald-500" /> optimal
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-amber-500" /> suboptimal
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-orange-500" /> low/high
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500" /> critical
        </span>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Shared
// ════════════════════════════════════════════════════════════════════════════
function Loading({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center h-40 text-slate-400 text-sm gap-2">
      <Loader2 className="w-4 h-4 animate-spin" /> {label}
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="p-8 rounded-lg bg-surface-card border border-surface-border text-center text-slate-400 text-sm">
      {msg}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Log tab — what I'm taking + my performance tests
// ════════════════════════════════════════════════════════════════════════════
const TEST_TYPES: { value: string; label: string; defaultUnit: string }[] = [
  { value: "ftp_20min",    label: "FTP (20-min test)", defaultUnit: "W" },
  { value: "ftp_ramp",     label: "FTP (ramp test)",   defaultUnit: "W" },
  { value: "ftp_8min",     label: "FTP (8-min test)",  defaultUnit: "W" },
  { value: "vo2max",       label: "VO2max",            defaultUnit: "ml/kg/min" },
  { value: "threshold_hr", label: "Threshold HR",      defaultUnit: "bpm" },
  { value: "resting_hr",   label: "Resting HR",        defaultUnit: "bpm" },
  { value: "weight",       label: "Body weight",       defaultUnit: "kg" },
];

function LogTab() {
  return (
    <div className="space-y-6">
      <ActiveStackSection />
      <PerformanceTestsSection />
    </div>
  );
}

// ── Active supplement intake + Dose Log ────────────────────────────────────
function ActiveStackSection() {
  const qc = useQueryClient();
  const { data: intakes, isLoading } = useQuery({
    queryKey: ["intakes"],
    queryFn: () => trackingAPI.listIntakes(),
  });
  const { data: doseLogs } = useQuery({
    queryKey: ["dose-logs"],
    queryFn: () => doseLogAPI.list({ since: new Date(Date.now() - 30 * 86400_000).toISOString() }),
    refetchInterval: 60_000,
  });

  const stop = useMutation({
    mutationFn: (id: string) =>
      trackingAPI.updateIntake(id, { stopped_at: new Date().toISOString() }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["intakes"] });
      qc.invalidateQueries({ queryKey: ["supplements"] });
      toast.success("Stopped");
    },
  });

  const setAdherence = useMutation({
    mutationFn: (args: { id: string; pct: number }) =>
      trackingAPI.updateIntake(args.id, { adherence_pct: args.pct }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["intakes"] }),
  });

  const del = useMutation({
    mutationFn: (id: string) => trackingAPI.deleteIntake(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["intakes"] });
      qc.invalidateQueries({ queryKey: ["supplements"] });
      toast.success("Deleted");
    },
  });

  const active = (intakes ?? []).filter((i) => i.is_active);
  const past = (intakes ?? []).filter((i) => !i.is_active);

  // Group dose logs by supplement_key for quick lookup
  const logsByKey = (doseLogs ?? []).reduce<Record<string, DoseLogRecord[]>>((acc, log) => {
    (acc[log.supplement_key] ??= []).push(log);
    return acc;
  }, {});

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Pill className="w-5 h-5 text-brand-500" /> Currently Taking
        </h2>
        <p className="text-xs text-slate-500">
          Use &quot;Log a dose&quot; on any stack card to start tracking.
        </p>
      </div>

      {isLoading ? (
        <Loading label="Loading log…" />
      ) : active.length === 0 ? (
        <Empty msg="Nothing logged as active yet. Go to Today's Stack and click 'Log a dose'." />
      ) : (
        <div className="space-y-3">
          {active.map((i) => (
            <IntakeCard
              key={i.id}
              intake={i}
              doseLogs={logsByKey[i.supplement_key] ?? []}
              onStop={() => stop.mutate(i.id)}
              onDelete={() => del.mutate(i.id)}
              onAdherence={(pct) => setAdherence.mutate({ id: i.id, pct })}
            />
          ))}
        </div>
      )}

      {past.length > 0 && (
        <details className="mt-4">
          <summary className="text-sm text-slate-400 cursor-pointer hover:text-white">
            Past supplements ({past.length})
          </summary>
          <div className="mt-2 space-y-2">
            {past.map((i) => (
              <IntakeCard
                key={i.id}
                intake={i}
                doseLogs={logsByKey[i.supplement_key] ?? []}
                onDelete={() => del.mutate(i.id)}
                onAdherence={(pct) => setAdherence.mutate({ id: i.id, pct })}
              />
            ))}
          </div>
        </details>
      )}
    </section>
  );
}

function IntakeCard({
  intake,
  doseLogs,
  onStop,
  onDelete,
  onAdherence,
}: {
  intake: SupplementIntakeRecord;
  doseLogs: DoseLogRecord[];
  onStop?: () => void;
  onDelete: () => void;
  onAdherence: (pct: number) => void;
}) {
  const qc = useQueryClient();
  const [editingLogId, setEditingLogId] = useState<string | null>(null);
  const [editDose, setEditDose] = useState("");
  const [showHistory, setShowHistory] = useState(false);

  const days = Math.max(
    1,
    Math.round(
      ((intake.stopped_at ? new Date(intake.stopped_at) : new Date()).getTime() -
        new Date(intake.started_at).getTime()) /
        86400000
    )
  );

  // Today's total from dose logs
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const todayLogs = doseLogs.filter((l) => new Date(l.taken_at) >= todayStart);
  const todayTotal = todayLogs.reduce((s, l) => s + l.dose_taken, 0);
  const recDose = intake.dose ?? 0;
  const todayPct = recDose > 0 ? Math.min(100, (todayTotal / recDose) * 100) : 0;

  const updateDose = useMutation({
    mutationFn: ({ id, dose_taken }: { id: string; dose_taken: number }) =>
      doseLogAPI.update(id, { dose_taken }),
    onSuccess: async () => {
      // Force-refetch so Today's Stack re-pulls enriched data from backend immediately
      await Promise.all([
        qc.refetchQueries({ queryKey: ["dose-logs"] }),
        qc.refetchQueries({ queryKey: ["supplements"] }),
        qc.refetchQueries({ queryKey: ["intakes"] }),
      ]);
      setEditingLogId(null);
      toast.success("Dose updated");
    },
    onError: () => toast.error("Could not update dose"),
  });

  const deleteLog = useMutation({
    mutationFn: (id: string) => doseLogAPI.delete(id),
    onSuccess: async () => {
      await Promise.all([
        qc.refetchQueries({ queryKey: ["dose-logs"] }),
        qc.refetchQueries({ queryKey: ["supplements"] }),
        qc.refetchQueries({ queryKey: ["intakes"] }),
      ]);
    },
  });

  // Group logs by calendar day for the history view
  const byDay = doseLogs.reduce<Record<string, DoseLogRecord[]>>((acc, log) => {
    const day = new Date(log.taken_at).toLocaleDateString(undefined, {
      weekday: "short", month: "short", day: "numeric",
    });
    (acc[day] ??= []).push(log);
    return acc;
  }, {});
  const days7 = Object.entries(byDay).slice(0, 7);

  return (
    <div className="p-4 rounded-xl bg-surface-card border border-surface-border space-y-3">
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm text-white font-semibold">{intake.label}</p>
          <p className="text-xs text-slate-400 mt-0.5">
            {intake.dose != null && `${intake.dose} ${intake.dose_unit ?? ""} · `}
            {intake.frequency ?? ""}{intake.timing ? ` · ${intake.timing}` : ""} ·{" "}
            {days}d {intake.is_active ? "and counting" : "total"}
            {intake.source === "recommended" ? " · from coach" : ""}
          </p>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {intake.is_active && onStop && (
            <button
              onClick={onStop}
              className="p-1.5 rounded text-slate-400 hover:text-amber-400 hover:bg-amber-500/10"
              title="Stop taking"
            >
              <Square className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={onDelete}
            className="p-1.5 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10"
            title="Delete"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Today's dose progress */}
      {intake.is_active && recDose > 0 && (
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">Today</span>
            <span
              className={cn(
                "text-xs font-medium",
                todayTotal > recDose * 1.5
                  ? "text-red-400"
                  : todayTotal >= recDose * 0.9
                  ? "text-emerald-400"
                  : "text-slate-400"
              )}
            >
              {todayTotal > 0
                ? `${todayTotal.toFixed(1)} / ${recDose} ${intake.dose_unit ?? ""}`
                : `Not taken yet — recommended ${recDose} ${intake.dose_unit ?? ""}`}
            </span>
          </div>
          {todayTotal > 0 && (
            <div className="h-1.5 bg-surface-muted rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all",
                  todayTotal > recDose * 1.5 ? "bg-red-500" : "bg-emerald-500"
                )}
                style={{ width: `${todayPct}%` }}
              />
            </div>
          )}
          {todayLogs.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {todayLogs.map((log) => (
                <DoseChip
                  key={log.id}
                  log={log}
                  isEditing={editingLogId === log.id}
                  editDose={editDose}
                  onEditStart={() => { setEditingLogId(log.id); setEditDose(String(log.dose_taken)); }}
                  onEditChange={setEditDose}
                  onEditSave={() => updateDose.mutate({ id: log.id, dose_taken: parseFloat(editDose) })}
                  onEditCancel={() => setEditingLogId(null)}
                  onDelete={() => deleteLog.mutate(log.id)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Adherence slider */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wide text-slate-500 whitespace-nowrap">
          Adherence
        </span>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          defaultValue={intake.adherence_pct ?? 100}
          onChange={(e) => onAdherence(Number(e.target.value))}
          className="flex-1 accent-brand-500"
        />
        <span className="text-xs text-slate-300 w-10 text-right">
          {intake.adherence_pct ?? 100}%
        </span>
      </div>

      {/* Dose history */}
      {days7.length > 0 && (
        <div>
          <button
            onClick={() => setShowHistory((v) => !v)}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-white"
          >
            <CalendarDays className="w-3 h-3" />
            {showHistory ? "Hide" : "Show"} dose history ({doseLogs.length} entries)
          </button>
          {showHistory && (
            <div className="mt-2 space-y-1.5">
              {days7.map(([day, logs]) => {
                const dayTotal = logs.reduce((s, l) => s + l.dose_taken, 0);
                return (
                  <div key={day} className="flex items-start gap-3 text-xs">
                    <span className="text-slate-500 w-28 flex-shrink-0">{day}</span>
                    <div className="flex flex-wrap gap-1">
                      {logs.map((log) => (
                        <DoseChip
                          key={log.id}
                          log={log}
                          isEditing={editingLogId === log.id}
                          editDose={editDose}
                          onEditStart={() => { setEditingLogId(log.id); setEditDose(String(log.dose_taken)); }}
                          onEditChange={setEditDose}
                          onEditSave={() => updateDose.mutate({ id: log.id, dose_taken: parseFloat(editDose) })}
                          onEditCancel={() => setEditingLogId(null)}
                          onDelete={() => deleteLog.mutate(log.id)}
                        />
                      ))}
                      <span className="text-slate-500 self-center ml-1">
                        = {dayTotal.toFixed(1)} {logs[0]?.dose_unit ?? ""}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Individual dose chip — show taken dose, click to edit inline
function DoseChip({
  log,
  isEditing,
  editDose,
  onEditStart,
  onEditChange,
  onEditSave,
  onEditCancel,
  onDelete,
}: {
  log: DoseLogRecord;
  isEditing: boolean;
  editDose: string;
  onEditStart: () => void;
  onEditChange: (v: string) => void;
  onEditSave: () => void;
  onEditCancel: () => void;
  onDelete: () => void;
}) {
  const time = new Date(log.taken_at).toLocaleTimeString(undefined, {
    hour: "2-digit", minute: "2-digit",
  });

  if (isEditing) {
    return (
      <span className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-brand-500/15 border border-brand-500/30">
        <input
          type="number"
          step="0.1"
          min="0.01"
          value={editDose}
          onChange={(e) => onEditChange(e.target.value)}
          className="w-14 bg-transparent text-white text-xs focus:outline-none"
          autoFocus
        />
        <span className="text-slate-400 text-[10px]">{log.dose_unit}</span>
        <button onClick={onEditSave} className="text-emerald-400 hover:text-emerald-300 text-[10px] ml-0.5">✓</button>
        <button onClick={onEditCancel} className="text-slate-500 hover:text-white text-[10px]">✕</button>
      </span>
    );
  }

  return (
    <span className="group flex items-center gap-1 px-2 py-0.5 rounded bg-surface-muted border border-surface-border text-xs text-slate-300">
      <span>{log.dose_taken} {log.dose_unit}</span>
      <span className="text-slate-500 text-[10px]">{time}</span>
      <button
        onClick={onEditStart}
        className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-white transition-opacity"
        title="Edit dose"
      >
        <Pencil className="w-2.5 h-2.5" />
      </button>
      <button
        onClick={onDelete}
        className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-400 transition-opacity"
        title="Delete"
      >
        <Trash2 className="w-2.5 h-2.5" />
      </button>
    </span>
  );
}

// ── Performance tests ───────────────────────────────────────────────────────
function PerformanceTestsSection() {
  const qc = useQueryClient();
  const { data: tests, isLoading } = useQuery({
    queryKey: ["perf-tests"],
    queryFn: () => trackingAPI.listPerformanceTests(),
  });

  const [showForm, setShowForm] = useState(false);

  const create = useMutation({
    mutationFn: (b: {
      test_date: string;
      test_type: string;
      value: number;
      unit: string;
      notes?: string;
    }) => trackingAPI.createPerformanceTest(b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["perf-tests"] });
      toast.success("Test logged");
      setShowForm(false);
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      toast.error(e?.response?.data?.detail ?? "Failed to log test"),
  });

  const del = useMutation({
    mutationFn: (id: string) => trackingAPI.deletePerformanceTest(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["perf-tests"] }),
  });

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-brand-500" /> Performance Tests
        </h2>
        <button
          onClick={() => setShowForm((v) => !v)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-brand-500 hover:bg-brand-400 text-white"
        >
          <Plus className="w-3.5 h-3.5" />
          {showForm ? "Cancel" : "Log a test"}
        </button>
      </div>

      {showForm && <PerfTestForm onSubmit={(b) => create.mutate(b)} pending={create.isPending} />}

      {isLoading ? (
        <Loading label="Loading…" />
      ) : !tests || tests.length === 0 ? (
        <Empty msg="No performance tests logged yet. Log an FTP test to start tracking." />
      ) : (
        <div className="space-y-2">
          {tests.map((t) => (
            <PerfTestRow key={t.id} test={t} onDelete={() => del.mutate(t.id)} />
          ))}
        </div>
      )}
    </section>
  );
}

function PerfTestRow({
  test,
  onDelete,
}: {
  test: PerformanceTestRecord;
  onDelete: () => void;
}) {
  const def = TEST_TYPES.find((t) => t.value === test.test_type);
  return (
    <div className="p-3 rounded-lg bg-surface-card border border-surface-border flex items-center justify-between">
      <div>
        <p className="text-sm text-white font-medium">
          {def?.label ?? test.test_type} ·{" "}
          <span className="text-brand-400">
            {test.value} {test.unit}
          </span>
        </p>
        <p className="text-xs text-slate-400 mt-0.5">
          {new Date(test.test_date).toLocaleDateString()}
          {test.notes ? ` · ${test.notes}` : ""}
        </p>
      </div>
      <button
        onClick={onDelete}
        className="p-1.5 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

function PerfTestForm({
  onSubmit,
  pending,
}: {
  onSubmit: (b: {
    test_date: string;
    test_type: string;
    value: number;
    unit: string;
    notes?: string;
  }) => void;
  pending: boolean;
}) {
  const [type, setType] = useState(TEST_TYPES[0].value);
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [value, setValue] = useState<string>("");
  const [unit, setUnit] = useState(TEST_TYPES[0].defaultUnit);
  const [notes, setNotes] = useState("");

  const handleTypeChange = (v: string) => {
    setType(v);
    const def = TEST_TYPES.find((t) => t.value === v);
    if (def) setUnit(def.defaultUnit);
  };

  const submit = () => {
    const num = Number(value);
    if (!num || num <= 0) {
      toast.error("Enter a positive value");
      return;
    }
    onSubmit({
      test_date: new Date(date).toISOString(),
      test_type: type,
      value: num,
      unit,
      notes: notes || undefined,
    });
  };

  return (
    <div className="p-4 rounded-lg bg-surface-card border border-surface-border grid sm:grid-cols-2 lg:grid-cols-5 gap-3">
      <label className="flex flex-col gap-1 text-xs text-slate-400">
        Type
        <select
          value={type}
          onChange={(e) => handleTypeChange(e.target.value)}
          className="px-2 py-1.5 rounded bg-surface-muted border border-surface-border text-sm text-white"
        >
          {TEST_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1 text-xs text-slate-400">
        Date
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="px-2 py-1.5 rounded bg-surface-muted border border-surface-border text-sm text-white"
        />
      </label>

      <label className="flex flex-col gap-1 text-xs text-slate-400">
        Value
        <input
          type="number"
          inputMode="decimal"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="px-2 py-1.5 rounded bg-surface-muted border border-surface-border text-sm text-white"
        />
      </label>

      <label className="flex flex-col gap-1 text-xs text-slate-400">
        Unit
        <input
          value={unit}
          onChange={(e) => setUnit(e.target.value)}
          className="px-2 py-1.5 rounded bg-surface-muted border border-surface-border text-sm text-white"
        />
      </label>

      <label className="flex flex-col gap-1 text-xs text-slate-400">
        Notes (optional)
        <input
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          className="px-2 py-1.5 rounded bg-surface-muted border border-surface-border text-sm text-white"
        />
      </label>

      <div className="sm:col-span-2 lg:col-span-5 flex justify-end">
        <button
          onClick={submit}
          disabled={pending}
          className="px-4 py-2 rounded-md text-sm font-medium bg-brand-500 hover:bg-brand-400 text-white disabled:opacity-50 flex items-center gap-2"
        >
          {pending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          Save test
        </button>
      </div>
    </div>
  );
}
