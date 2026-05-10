"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { stravaAPI } from "@/lib/api";
import toast from "react-hot-toast";
import {
  Wifi,
  WifiOff,
  RefreshCw,
  Trash2,
  Loader2,
  CheckCircle2,
} from "lucide-react";

export default function StravaConnect() {
  const qc = useQueryClient();
  const [syncTaskId, setSyncTaskId] = useState<string | null>(null);

  // ── Status ─────────────────────────────────────────────────────────────────
  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ["strava-status"],
    queryFn: stravaAPI.status,
  });

  // ── Connect (redirect to OAuth) ────────────────────────────────────────────
  async function connectStrava() {
    try {
      const url = await stravaAPI.getAuthURL();
      window.location.href = url;
    } catch {
      toast.error("Could not start Strava auth");
    }
  }

  // ── Disconnect ─────────────────────────────────────────────────────────────
  const disconnect = useMutation({
    mutationFn: () => stravaAPI.disconnect(),
    onSuccess: () => {
      toast.success("Strava disconnected");
      setSyncTaskId(null);
      qc.invalidateQueries({ queryKey: ["strava-status"] });
    },
    onError: () => toast.error("Failed to disconnect"),
  });

  // ── Sync ───────────────────────────────────────────────────────────────────
  const sync = useMutation({
    mutationFn: () => stravaAPI.syncHistory(),
    onSuccess: (data) => {
      setSyncTaskId(data.task_id);
      toast.success("Strava sync started — this may take a few minutes");
    },
    onError: () => toast.error("Strava sync failed. Is your account connected?"),
  });

  // ── Poll sync status ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!syncTaskId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const s = await stravaAPI.getSyncStatus(syncTaskId);
        if (cancelled) return;
        if (s.status === "completed") {
          setSyncTaskId(null);
          toast.success(`Strava sync done — ${s.total} activities processed`);
          qc.invalidateQueries({ queryKey: ["activities"] });
        } else if (s.status === "failed") {
          setSyncTaskId(null);
          toast.error("Strava sync failed");
        } else {
          setTimeout(poll, 3000);
        }
      } catch {
        if (!cancelled) setSyncTaskId(null);
      }
    };

    poll();
    return () => { cancelled = true; };
  }, [syncTaskId, qc]);

  const isSyncing = !!syncTaskId || sync.isPending;
  const connected = status?.connected ?? false;

  return (
    <div className="rounded-2xl border border-surface-border bg-surface-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-surface-border">
        <div className="w-7 h-7 bg-orange-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
          <span className="text-orange-400 font-bold text-sm">S</span>
        </div>
        <div className="flex-1">
          <div className="text-sm font-semibold text-white">Strava</div>
          <div className="text-xs text-slate-500 mt-0.5">Sync your activity history via OAuth</div>
        </div>

        {statusLoading ? (
          <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
        ) : connected ? (
          <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Connected
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-xs text-slate-500">
            <WifiOff className="w-3.5 h-3.5" />
            Not connected
          </span>
        )}
      </div>

      {/* Body */}
      <div className="px-6 py-5">
        {connected ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Wifi className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <span>Strava account connected (athlete #{status?.athlete_id})</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => sync.mutate()}
                disabled={isSyncing}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-orange-500 hover:bg-orange-600
                           disabled:opacity-50 text-white text-sm font-medium transition-colors"
              >
                {isSyncing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4" />
                )}
                {isSyncing ? "Syncing…" : "Sync history"}
              </button>
              <button
                onClick={() => disconnect.mutate()}
                disabled={disconnect.isPending || isSyncing}
                className="flex items-center gap-2 px-3 py-2 rounded-xl border border-surface-border
                           hover:border-red-500/50 hover:text-red-400 text-slate-400
                           disabled:opacity-50 text-sm transition-colors"
              >
                {disconnect.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4" />
                )}
                Disconnect
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-slate-400">
              Connect your Strava account to automatically import rides and activities.
            </p>
            <button
              onClick={connectStrava}
              className="flex items-center gap-2 px-4 py-2 bg-orange-500 hover:bg-orange-600
                         text-white text-sm font-medium rounded-xl transition-colors"
            >
              Connect Strava
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
