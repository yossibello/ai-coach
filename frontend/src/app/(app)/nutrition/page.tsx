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
  ExternalLink,
  Loader2,
} from "lucide-react";
import toast from "react-hot-toast";
import {
  nutritionAPI,
  type BloodTest,
  type SupplementItem,
  type SupplementWarning,
  type MarkerTimeSeries,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type Tab = "tests" | "supplements" | "trends";

export default function NutritionPage() {
  const [tab, setTab] = useState<Tab>("supplements");

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
      </div>

      {tab === "supplements" && <SupplementsTab />}
      {tab === "tests" && <BloodTestsTab />}
      {tab === "trends" && <TrendsTab />}
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
  const { data, isLoading } = useQuery({
    queryKey: ["supplements"],
    queryFn: () => nutritionAPI.getSupplements(),
  });
  const refresh = useMutation({
    mutationFn: () => nutritionAPI.refreshSupplements(),
    onSuccess: (d) => {
      qc.setQueryData(["supplements"], d);
      toast.success("Stack refreshed");
    },
    onError: () => toast.error("Could not refresh stack"),
  });

  if (isLoading) return <Loading label="Computing your stack…" />;
  if (!data) return <Empty msg="No supplement data yet." />;

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
                ? "Personalized using your latest blood test"
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
          disabled={refresh.isPending}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm bg-surface-muted text-slate-300 hover:text-white border border-surface-border disabled:opacity-50"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", refresh.isPending && "animate-spin")} />
          Refresh
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
          {stack.map((s) => (
            <SupplementCard key={s.supplement_key} item={s} />
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

function SupplementCard({ item }: { item: SupplementItem }) {
  const [expanded, setExpanded] = useState(false);
  const gradeColor =
    item.evidence_grade === "A"
      ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
      : item.evidence_grade === "B"
      ? "bg-sky-500/15 text-sky-400 border-sky-500/30"
      : "bg-amber-500/15 text-amber-400 border-amber-500/30";

  return (
    <div className="p-4 rounded-lg bg-surface-card border border-surface-border">
      <div className="flex items-start justify-between gap-3">
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
            className="mt-3 text-xs text-brand-400 hover:text-brand-300"
          >
            {expanded ? "Hide" : "Show"} citations &amp; trigger
          </button>

          {expanded && (
            <div className="mt-2 space-y-1.5 text-xs text-slate-400">
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
