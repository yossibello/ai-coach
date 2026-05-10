"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { activitiesAPI, stravaAPI, garminAPI } from "@/lib/api";
import { formatDuration, formatDistance, formatDate, relativeDate } from "@/lib/utils";
import type { Activity, PaginatedResponse } from "@/types";
import { Trash2, ChevronLeft, ChevronRight, Activity as ActivityIcon, RefreshCw } from "lucide-react";
import toast from "react-hot-toast";
import Link from "next/link";

const WORKOUT_TYPE_COLORS: Record<string, string> = {
  recovery: "bg-blue-900 text-blue-300",
  easy: "bg-green-900 text-green-300",
  endurance: "bg-teal-900 text-teal-300",
  tempo: "bg-yellow-900 text-yellow-300",
  sweetspot: "bg-orange-900 text-orange-300",
  threshold: "bg-orange-900 text-orange-400",
  vo2max: "bg-red-900 text-red-300",
  sprint: "bg-purple-900 text-purple-300",
  race: "bg-pink-900 text-pink-300",
  long_ride: "bg-indigo-900 text-indigo-300",
};

const PAGE_SIZE = 20;

export default function ActivitiesPage() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");

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

  // Manual refresh: just re-fetch the activities list.
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
        [stravaTaskId && stravaSync ? `Strava ${stravaSync.progress ?? 0}/${stravaSync.total ?? "?"}` : "",
         garminTaskId && garminSync ? `Garmin ${garminSync.progress ?? 0}/${garminSync.total ?? "?"}` : ""]
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
            className="flex items-center gap-1.5 px-3 py-2 bg-surface border border-surface-border
                       hover:border-brand-500/50 text-slate-300 text-sm rounded-lg transition disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            onClick={handleSyncAll}
            disabled={isSyncing}
            title="Pull new rides from Garmin &amp; Strava"
            className="flex items-center gap-1.5 px-3 py-2 bg-brand-500 hover:bg-brand-600
                       text-white text-sm font-medium rounded-lg transition disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isSyncing ? "animate-spin" : ""}`} />
            {syncLabel}
          </button>
          <Link
            href="/upload"
            className="bg-brand-500 hover:bg-brand-600 text-white px-4 py-2 rounded-lg
                       text-sm font-medium transition"
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
                <th className="text-left px-6 py-3 font-medium">Activity</th>
                <th className="text-right px-4 py-3 font-medium">Duration</th>
                <th className="text-right px-4 py-3 font-medium">Distance</th>
                <th className="text-right px-4 py-3 font-medium">Power</th>
                <th className="text-right px-4 py-3 font-medium">TSS</th>
                <th className="text-left px-4 py-3 font-medium">Type</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border">
              {data.items.map((act) => (
                <tr key={act.id} className="hover:bg-surface-border/20 transition">
                  <td className="px-6 py-4">
                    <div className="font-medium text-white">{act.name}</div>
                    <div className="text-gray-500 text-xs mt-0.5">
                      {relativeDate(act.date)} · {act.source}
                    </div>
                  </td>
                  <td className="px-4 py-4 text-right text-gray-300">
                    {act.duration_seconds ? formatDuration(act.duration_seconds) : "—"}
                  </td>
                  <td className="px-4 py-4 text-right text-gray-300">
                    {act.distance_meters ? formatDistance(act.distance_meters) : "—"}
                  </td>
                  <td className="px-4 py-4 text-right text-gray-300">
                    {act.avg_power ? `${Math.round(act.avg_power)}W` : "—"}
                  </td>
                  <td className="px-4 py-4 text-right text-gray-300">
                    {act.tss ? Math.round(act.tss) : "—"}
                  </td>
                  <td className="px-4 py-4">
                    {act.workout_type ? (
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          WORKOUT_TYPE_COLORS[act.workout_type] ?? "bg-gray-800 text-gray-300"
                        }`}
                      >
                        {act.workout_type.replace("_", " ")}
                      </span>
                    ) : null}
                  </td>
                  <td className="px-4 py-4 text-right">
                    <button
                      onClick={() => {
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
    </div>
  );
}
