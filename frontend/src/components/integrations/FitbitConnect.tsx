"use client";

import { useState, useEffect, Suspense } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { fitbitAPI } from "@/lib/api";
import toast from "react-hot-toast";
import {
  WifiOff,
  RefreshCw,
  Trash2,
  Loader2,
  CheckCircle2,
  Info,
  Moon,
  Heart,
  Activity,
  Zap,
  LogIn,
} from "lucide-react";

const FITBIT_TASK_KEY = "fitbit_sync_task_id";

function FitbitConnectInner() {
  const qc = useQueryClient();
  const searchParams = useSearchParams();
  const [syncTaskId, setSyncTaskIdState] = useState<string | null>(() =>
    typeof window !== "undefined" ? localStorage.getItem(FITBIT_TASK_KEY) : null
  );
  const [syncStats, setSyncStats] = useState<Record<string, number> | null>(null);

  function setSyncTaskId(id: string | null) {
    if (id) localStorage.setItem(FITBIT_TASK_KEY, id);
    else localStorage.removeItem(FITBIT_TASK_KEY);
    setSyncTaskIdState(id);
  }

  // Show success toast when redirected back from Fitbit OAuth
  useEffect(() => {
    const result = searchParams.get("fitbit");
    if (result === "connected") {
      toast.success("Fitbit connected!");
      qc.invalidateQueries({ queryKey: ["fitbit-status"] });
    } else if (result === "error") {
      toast.error("Fitbit connection failed — please try again");
    }
  }, [searchParams, qc]);

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ["fitbit-status"],
    queryFn: fitbitAPI.status,
  });

  const startOAuth = useMutation({
    mutationFn: fitbitAPI.authUrl,
    onSuccess: (data) => {
      window.location.href = data.url;
    },
    onError: () => toast.error("Could not get Fitbit authorization URL"),
  });

  const disconnect = useMutation({
    mutationFn: fitbitAPI.disconnect,
    onSuccess: () => {
      toast.success("Fitbit disconnected");
      setSyncTaskId(null);
      setSyncStats(null);
      qc.invalidateQueries({ queryKey: ["fitbit-status"] });
    },
    onError: () => toast.error("Failed to disconnect"),
  });

  const sync = useMutation({
    mutationFn: () => fitbitAPI.sync(60),
    onSuccess: (data) => {
      setSyncTaskId(data.task_id);
      setSyncStats(null);
    },
    onError: () => toast.error("Sync failed to start"),
  });

  useEffect(() => {
    if (!syncTaskId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const s = await fitbitAPI.syncStatus(syncTaskId);
        if (cancelled) return;
        if (s.status === "completed") {
          setSyncTaskId(null);
          setSyncStats(s.stats);
          toast.success(
            `Sync done — ${s.stats?.health_days_added ?? 0} new days, ${s.stats?.health_days_updated ?? 0} updated`
          );
          qc.invalidateQueries({ queryKey: ["fitbit-status"] });
        } else if (s.status === "failed") {
          setSyncTaskId(null);
          toast.error(`Sync failed: ${s.error}`);
        } else {
          setTimeout(poll, 2500);
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
      <div className="flex items-center gap-3 px-6 py-4 border-b border-surface-border">
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center flex-shrink-0">
          <Activity className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1">
          <div className="text-sm font-semibold text-white">Fitbit</div>
          <div className="text-xs text-slate-500 mt-0.5">Sleep, heart rate &amp; HRV</div>
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

      <div className="px-6 py-5 space-y-4">
        {connected ? (
          <>
            {status?.user_id && (
              <div className="text-sm text-slate-300">
                Fitbit ID: <strong className="text-white">{status.user_id}</strong>
              </div>
            )}

            {syncStats && (
              <div className="grid grid-cols-2 gap-2 text-xs">
                <Stat label="New days" value={syncStats.health_days_added} />
                <Stat label="Updated days" value={syncStats.health_days_updated} />
                <Stat label="Errors" value={syncStats.errors} />
              </div>
            )}

            <div className="flex gap-2 pt-1">
              <button
                onClick={() => sync.mutate()}
                disabled={isSyncing}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-500 hover:bg-brand-600
                           disabled:opacity-50 text-white text-sm font-medium transition-colors"
              >
                {isSyncing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4" />
                )}
                {isSyncing ? "Syncing…" : "Sync Now"}
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
          </>
        ) : (
          <>
            <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-4 space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wide">
                <Info className="w-3.5 h-3.5 text-brand-400" />
                Why connect Fitbit?
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { icon: <Moon className="w-3.5 h-3.5" />, label: "Sleep stages", desc: "Deep, REM & light (min → sec)" },
                  { icon: <Heart className="w-3.5 h-3.5" />, label: "Resting HR", desc: "Daily baseline" },
                  { icon: <Zap className="w-3.5 h-3.5" />, label: "HRV (Premium)", desc: "Daily RMSSD" },
                  { icon: <Activity className="w-3.5 h-3.5" />, label: "Sleep efficiency", desc: "Used as sleep score" },
                ].map(({ icon, label, desc }) => (
                  <div key={label} className="flex items-start gap-2 text-xs">
                    <span className="text-brand-400 mt-0.5">{icon}</span>
                    <div>
                      <div className="text-slate-200 font-medium">{label}</div>
                      <div className="text-slate-500">{desc}</div>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                Uses Fitbit's official OAuth2 API. Your data stays on your device — we only read health
                metrics to inform your training plan.
              </p>
            </div>

            <button
              onClick={() => startOAuth.mutate()}
              disabled={startOAuth.isPending}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-500 hover:bg-brand-600
                         disabled:opacity-50 text-white text-sm font-medium transition-colors"
            >
              {startOAuth.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <LogIn className="w-4 h-4" />
              )}
              Connect with Fitbit
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default function FitbitConnect() {
  return (
    <Suspense fallback={null}>
      <FitbitConnectInner />
    </Suspense>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-surface-800/50 rounded-lg px-3 py-2">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-base font-semibold text-white">{value}</div>
    </div>
  );
}
