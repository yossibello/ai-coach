"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { activitiesAPI, stravaAPI, garminAPI } from "@/lib/api";
import { formatDuration, formatDistance, formatDate, relativeDate } from "@/lib/utils";
import type { Activity, PaginatedResponse } from "@/types";
import {
  Trash2, ChevronLeft, ChevronRight, Activity as ActivityIcon,
  RefreshCw, X, Zap, Heart, Gauge, TrendingUp, BarChart2,
  CheckCircle2, Circle, AlertCircle, ExternalLink,
} from "lucide-react";
import toast from "react-hot-toast";
import Link from "next/link";

const WORKOUT_TYPE_COLORS: Record<string, string> = {
  recovery:   "bg-blue-900 text-blue-300",
  easy:       "bg-green-900 text-green-300",
  endurance:  "bg-teal-900 text-teal-300",
  tempo:      "bg-yellow-900 text-yellow-300",
  sweetspot:  "bg-orange-900 text-orange-300",
  threshold:  "bg-orange-900 text-orange-400",
  vo2max:     "bg-red-900 text-red-300",
  sprint:     "bg-purple-900 text-purple-300",
  race:       "bg-pink-900 text-pink-300",
  long_ride:  "bg-indigo-900 text-indigo-300",
};

const SOURCE_LABELS: Record<string, { label: string; color: string }> = {
  strava:  { label: "Strava",      color: "bg-orange-900 text-orange-300" },
  garmin:  { label: "Garmin",      color: "bg-blue-900 text-blue-300" },
  gpx:     { label: "GPX file",    color: "bg-teal-900 text-teal-300" },
  fit:     { label: "FIT file",    color: "bg-teal-900 text-teal-300" },
  manual:  { label: "Manual",      color: "bg-gray-800 text-gray-300" },
};

const QUALITY_COLORS: Record<string, string> = {
  high:     "text-green-400",
  medium:   "text-yellow-400",
  low:      "text-orange-400",
  rejected: "text-red-400",
};

const PAGE_SIZE = 20;

// ─── helper ──────────────────────────────────────────────────────────────────

function tssMethod(act: Activity): string {
  if (act.normalized_power && act.intensity_factor) return "Normalized Power (Coggan)";
  if (act.avg_power && !act.normalized_power)       return "Avg Power (NP unavailable)";
  if (act.avg_hr && !act.avg_power)                 return "Heart Rate estimate";
  return "Not calculated (no power or HR)";
}

function Field({
  label, value, unit, present,
}: { label: string; value?: string | number | null; unit?: string; present: boolean }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-surface-border last:border-0">
      <span className="flex items-center gap-1.5 text-slate-400 text-sm">
        {present
          ? <CheckCircle2 className="w-3.5 h-3.5 text-brand-500 flex-shrink-0" />
          : <Circle className="w-3.5 h-3.5 text-slate-700 flex-shrink-0" />}
        {label}
      </span>
      <span className={`text-sm font-medium ${present ? "text-white" : "text-slate-700"}`}>
        {present && value != null ? `${value}${unit ? " " + unit : ""}` : "—"}
      </span>
    </div>
  );
}

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-surface rounded-xl p-4">
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
        {icon}
        {title}
      </h3>
      {children}
    </div>
  );
}

// ─── Activity Detail Drawer ───────────────────────────────────────────────────

function ActivityDrawer({ activityId, onClose }: { activityId: string; onClose: () => void }) {
  const { data: act, isLoading } = useQuery<Activity>({
    queryKey: ["activity", activityId],
    queryFn: () => activitiesAPI.get(activityId),
  });

  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      {/* backdrop */}
      <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />

      {/* drawer */}
      <div className="fixed right-0 top-0 h-full w-full max-w-md bg-surface-card border-l border-surface-border z-50 overflow-y-auto shadow-2xl">
        {/* header */}
        <div className="sticky top-0 bg-surface-card border-b border-surface-border px-5 py-4 flex items-start justify-between z-10">
          <div>
            {isLoading ? (
              <div className="h-5 w-48 bg-slate-800 rounded animate-pulse" />
            ) : (
              <>
                <h2 className="font-semibold text-white text-base leading-tight">{act?.name}</h2>
                <p className="text-slate-400 text-xs mt-0.5">
                  {act ? formatDate(act.date) : ""}
                  {act?.trainer ? " · Indoor trainer" : ""}
                </p>
              </>
            )}
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white p-1 transition mt-0.5">
            <X className="w-5 h-5" />
          </button>
        </div>

        {isLoading ? (
          <div className="p-5 space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-16 bg-slate-800 rounded-xl animate-pulse" />
            ))}
          </div>
        ) : act ? (
          <div className="p-5 space-y-4">

            {/* Source badge + basics */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${SOURCE_LABELS[act.source]?.color ?? "bg-gray-800 text-gray-300"}`}>
                {SOURCE_LABELS[act.source]?.label ?? act.source}
              </span>
              {act.device_watts && (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-900/50 text-brand-400 border border-brand-500/30">
                  Power Meter
                </span>
              )}
              {act.workout_type && (
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${WORKOUT_TYPE_COLORS[act.workout_type] ?? "bg-gray-800 text-gray-300"}`}>
                  {act.workout_type.replace("_", " ")}
                </span>
              )}
              {act.quality_score && (
                <span className={`text-xs font-medium ${QUALITY_COLORS[act.quality_score]}`}>
                  Quality: {act.quality_score}
                </span>
              )}
              {act.external_id && act.source === "strava" && (
                <a
                  href={`https://www.strava.com/activities/${act.external_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-orange-400 hover:text-orange-300 text-xs flex items-center gap-0.5 ml-auto"
                >
                  View on Strava <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>

            {/* ── From Source ────────────────────────────────────────── */}
            <Section title={`From ${SOURCE_LABELS[act.source]?.label ?? act.source}`} icon={<ActivityIcon className="w-3.5 h-3.5" />}>
              <Field label="Duration"        value={formatDuration(act.duration_seconds)}  unit=""   present={!!act.duration_seconds} />
              <Field label="Distance"        value={formatDistance(act.distance_meters)}   unit=""   present={!!act.distance_meters} />
              <Field label="Elevation gain"  value={act.elevation_gain_meters != null ? Math.round(act.elevation_gain_meters) : null} unit="m" present={act.elevation_gain_meters != null} />
              <Field label="Avg power"       value={act.avg_power != null ? Math.round(act.avg_power) : null}            unit="W"   present={!!act.avg_power} />
              <Field label="Max power"       value={act.max_power != null ? Math.round(act.max_power) : null}            unit="W"   present={!!act.max_power} />
              <Field label="Norm. power (NP)"value={act.normalized_power != null ? Math.round(act.normalized_power) : null} unit="W" present={!!act.normalized_power} />
              <Field label="Avg heart rate"  value={act.avg_hr}   unit="bpm" present={!!act.avg_hr} />
              <Field label="Max heart rate"  value={act.max_hr}   unit="bpm" present={!!act.max_hr} />
              <Field label="Avg cadence"     value={act.avg_cadence} unit="rpm" present={!!act.avg_cadence} />
              <Field label="Temperature"     value={act.temperature_c != null ? act.temperature_c.toFixed(1) : null} unit="°C" present={act.temperature_c != null} />
            </Section>

            {/* ── Calculated ─────────────────────────────────────────── */}
            <Section title="Calculated by AI Coach" icon={<TrendingUp className="w-3.5 h-3.5" />}>
              <div className="mb-2 pb-2 border-b border-surface-border">
                <span className="text-xs text-slate-500">TSS method: </span>
                <span className="text-xs text-slate-300">{tssMethod(act)}</span>
              </div>
              <Field label="TSS"                 value={act.tss != null ? Math.round(act.tss) : null}                          unit=""     present={!!act.tss} />
              <Field label="Intensity Factor"    value={act.intensity_factor != null ? act.intensity_factor.toFixed(3) : null}  unit=""     present={!!act.intensity_factor} />
              <Field label="Variability Index"   value={act.variability_index != null ? act.variability_index.toFixed(3) : null} unit=""    present={!!act.variability_index} />
              <Field label="Aerobic efficiency"  value={act.aerobic_efficiency != null ? act.aerobic_efficiency.toFixed(2) : null} unit="W/bpm" present={!!act.aerobic_efficiency} />
              <Field label="HR drift (decoupling)" value={act.hr_drift != null ? act.hr_drift.toFixed(1) : null}                unit="%"    present={act.hr_drift != null} />
            </Section>

            {/* ── Power Curve ────────────────────────────────────────── */}
            {(act.pc_5s_wkg || act.pc_1min_wkg || act.pc_5min_wkg || act.pc_20min_wkg) && (
              <Section title="Power Curve Peaks" icon={<Zap className="w-3.5 h-3.5" />}>
                <Field label="5-second peak"   value={act.pc_5s_wkg?.toFixed(2)}   unit="W/kg" present={!!act.pc_5s_wkg} />
                <Field label="1-minute peak"   value={act.pc_1min_wkg?.toFixed(2)} unit="W/kg" present={!!act.pc_1min_wkg} />
                <Field label="5-minute peak"   value={act.pc_5min_wkg?.toFixed(2)} unit="W/kg" present={!!act.pc_5min_wkg} />
                <Field label="20-minute peak"  value={act.pc_20min_wkg?.toFixed(2)} unit="W/kg" present={!!act.pc_20min_wkg} />
              </Section>
            )}

            {/* ── Quality ────────────────────────────────────────────── */}
            {act.quality_reasons && act.quality_reasons.length > 0 && (
              <Section title="Data Quality Notes" icon={<AlertCircle className="w-3.5 h-3.5" />}>
                <ul className="space-y-1">
                  {act.quality_reasons.map((r: string) => (
                    <li key={r} className="text-xs text-slate-400 flex items-center gap-1.5">
                      <AlertCircle className="w-3 h-3 text-yellow-500 flex-shrink-0" />
                      {r.replace(/_/g, " ")}
                    </li>
                  ))}
                </ul>
              </Section>
            )}

            {/* ── Notes ──────────────────────────────────────────────── */}
            {act.notes && (
              <Section title="Notes" icon={<BarChart2 className="w-3.5 h-3.5" />}>
                <p className="text-sm text-slate-300 whitespace-pre-wrap">{act.notes}</p>
              </Section>
            )}

          </div>
        ) : (
          <div className="p-5 text-slate-400 text-sm">Activity not found.</div>
        )}
      </div>
    </>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function ActivitiesPage() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { data, isLoading } = useQuery<PaginatedResponse<Activity>>({
    queryKey: ["activities", page, search],
    queryFn: () => activitiesAPI.list({ page, limit: PAGE_SIZE, search }),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => activitiesAPI.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["activities"] });
      toast.success("Activity deleted");
    },
    onError: () => toast.error("Failed to delete activity"),
  });

  const [refreshing, setRefreshing] = useState(false);
  async function handleRefresh() {
    setRefreshing(true);
    await queryClient.invalidateQueries({ queryKey: ["activities"] });
    setRefreshing(false);
  }

  // ── Strava sync ───────────────────────────────────────────────────────
  const [stravaTaskId, setStravaTaskId] = useState<string | null>(null);
  const { data: stravaSync } = useQuery({
    queryKey: ["strava-sync", stravaTaskId],
    queryFn: () => stravaAPI.getSyncStatus(stravaTaskId!),
    enabled: !!stravaTaskId,
    refetchInterval: (q) =>
      q.state.data?.status === "completed" || q.state.data?.status === "failed" ? false : 3000,
  });
  useEffect(() => {
    if (stravaSync?.status === "completed") {
      queryClient.invalidateQueries({ queryKey: ["activities"] });
      toast.success(`Strava: ${stravaSync.progress} activities synced`);
      setStravaTaskId(null);
    } else if (stravaSync?.status === "failed") {
      toast.error("Strava sync failed");
      setStravaTaskId(null);
    }
  }, [stravaSync?.status, stravaSync?.progress, queryClient]);

  // ── Garmin sync ───────────────────────────────────────────────────────
  const [garminTaskId, setGarminTaskId] = useState<string | null>(null);
  const { data: garminSync } = useQuery({
    queryKey: ["garmin-sync", garminTaskId],
    queryFn: () => garminAPI.syncStatus(garminTaskId!),
    enabled: !!garminTaskId,
    refetchInterval: (q) =>
      q.state.data?.status === "completed" || q.state.data?.status === "failed" ? false : 3000,
  });
  useEffect(() => {
    if (garminSync?.status === "completed") {
      queryClient.invalidateQueries({ queryKey: ["activities"] });
      queryClient.invalidateQueries({ queryKey: ["fitness"] });
      queryClient.invalidateQueries({ queryKey: ["recommendation"] });
      queryClient.invalidateQueries({ queryKey: ["progression"] });
      const stats = garminSync.stats;
      const added = stats ? (stats.inserted ?? 0) : garminSync.progress ?? 0;
      toast.success(`Garmin: ${added} new activities`);
      setGarminTaskId(null);
    } else if (garminSync?.status === "failed") {
      toast.error(`Garmin sync failed: ${garminSync.error ?? "unknown error"}`);
      setGarminTaskId(null);
    }
  }, [garminSync?.status, garminSync?.progress, garminSync?.stats, queryClient]);

  // ── Sync All ──────────────────────────────────────────────────────────
  const [syncAllPending, setSyncAllPending] = useState(false);
  async function handleSyncAll() {
    setSyncAllPending(true);
    let any = false;
    try {
      const stravaStatus = await stravaAPI.status();
      if (stravaStatus.connected) {
        const { task_id } = await stravaAPI.syncHistory();
        setStravaTaskId(task_id);
        any = true;
      }
    } catch { /* Strava not connected */ }
    try {
      const gStatus = await garminAPI.status();
      if (gStatus.connected) {
        const { task_id } = await garminAPI.sync(60);
        setGarminTaskId(task_id);
        any = true;
      }
    } catch { /* Garmin not connected */ }
    if (!any) toast.error("No services connected. Go to Profile → Integrations.");
    setSyncAllPending(false);
  }

  const isSyncing =
    syncAllPending ||
    (!!stravaTaskId && stravaSync?.status !== "completed" && stravaSync?.status !== "failed") ||
    (!!garminTaskId && garminSync?.status !== "completed" && garminSync?.status !== "failed");

  const syncLabel = isSyncing
    ? `Syncing… ${
        [stravaTaskId && stravaSync
          ? `Strava ${stravaSync.progress ?? 0}/${stravaSync.total || "?"}`
          : "",
         garminTaskId && garminSync
          ? `Garmin ${garminSync.progress ?? 0}/${garminSync.total || "?"}`
          : ""]
          .filter(Boolean).join(" · ") || ""
      }`
    : "Sync All";

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Activities</h1>
          <p className="text-gray-400 text-sm mt-1">
            {data ? `${data.total} total rides` : "Loading…"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            title="Reload from server"
            className="p-2 bg-surface border border-surface-border hover:border-brand-500/50
                       text-slate-300 rounded-lg transition disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
          </button>
          <button
            onClick={handleSyncAll}
            disabled={isSyncing}
            title="Pull new rides from Garmin &amp; Strava"
            className="flex items-center gap-1.5 px-3 py-2 bg-brand-500 hover:bg-brand-600
                       text-white text-sm font-medium rounded-lg transition disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isSyncing ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">{syncLabel}</span>
            <span className="sm:hidden">Sync</span>
          </button>
          <Link
            href="/upload"
            className="bg-surface border border-surface-border hover:border-brand-500/50
                       text-slate-300 px-3 py-2 rounded-lg text-sm font-medium transition"
          >
            + Upload
          </Link>
        </div>
      </div>

      {/* Search */}
      <input
        type="text"
        placeholder="Search activities…"
        value={search}
        onChange={(e) => { setSearch(e.target.value); setPage(1); }}
        className="w-full bg-surface-card border border-surface-border rounded-lg px-4 py-2.5
                   text-white placeholder-gray-500 focus:outline-none focus:ring-2
                   focus:ring-brand-500 transition"
      />

      {/* Table */}
      <div className="bg-surface-card border border-surface-border rounded-2xl overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-gray-400">Loading activities…</div>
        ) : !data?.items?.length ? (
          <div className="p-12 text-center">
            <ActivityIcon className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No activities yet.</p>
            <Link href="/upload" className="text-brand-500 hover:underline text-sm mt-1 inline-block">
              Upload a GPX/FIT file or connect Strava →
            </Link>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="border-b border-surface-border">
              <tr className="text-gray-400">
                <th className="text-left px-4 py-3 font-medium">Activity</th>
                <th className="text-right px-3 py-3 font-medium">Dur</th>
                <th className="hidden sm:table-cell text-right px-3 py-3 font-medium">Dist</th>
                <th className="hidden sm:table-cell text-right px-3 py-3 font-medium">Power</th>
                <th className="text-right px-3 py-3 font-medium">TSS</th>
                <th className="text-left px-3 py-3 font-medium">Type</th>
                <th className="px-3 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border">
              {data.items.map((act) => (
                <tr
                  key={act.id}
                  onClick={() => setSelectedId(act.id)}
                  className="hover:bg-surface-border/20 transition cursor-pointer"
                >
                  <td className="px-4 py-3">
                    <div className="font-medium text-white truncate max-w-[140px] sm:max-w-none">{act.name}</div>
                    <div className="text-gray-500 text-xs mt-0.5 flex items-center gap-1.5">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${SOURCE_LABELS[act.source]?.color ?? "bg-gray-800 text-gray-400"}`}>
                        {SOURCE_LABELS[act.source]?.label ?? act.source}
                      </span>
                      {relativeDate(act.date)}
                    </div>
                  </td>
                  <td className="px-3 py-3 text-right text-gray-300 whitespace-nowrap">
                    {act.duration_seconds ? formatDuration(act.duration_seconds) : "—"}
                  </td>
                  <td className="hidden sm:table-cell px-3 py-3 text-right text-gray-300 whitespace-nowrap">
                    {act.distance_meters ? formatDistance(act.distance_meters) : "—"}
                  </td>
                  <td className="hidden sm:table-cell px-3 py-3 text-right text-gray-300 whitespace-nowrap">
                    {act.avg_power ? `${Math.round(act.avg_power)}W` : "—"}
                  </td>
                  <td className="px-3 py-3 text-right text-gray-300 whitespace-nowrap">
                    {act.tss ? Math.round(act.tss) : "—"}
                  </td>
                  <td className="px-3 py-3">
                    {act.workout_type ? (
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium whitespace-nowrap ${WORKOUT_TYPE_COLORS[act.workout_type] ?? "bg-gray-800 text-gray-300"}`}>
                        {act.workout_type.replace("_", " ")}
                      </span>
                    ) : null}
                  </td>
                  <td className="px-3 py-3 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm("Delete this activity?")) deleteMutation.mutate(act.id);
                      }}
                      className="text-gray-600 hover:text-red-400 transition"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-sm">
            Page {page} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-2 rounded-lg bg-surface-card border border-surface-border
                         text-gray-300 disabled:opacity-40 hover:border-brand-500 transition"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="p-2 rounded-lg bg-surface-card border border-surface-border
                         text-gray-300 disabled:opacity-40 hover:border-brand-500 transition"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Detail drawer */}
      {selectedId && (
        <ActivityDrawer activityId={selectedId} onClose={() => setSelectedId(null)} />
      )}
    </div>
  );
}
