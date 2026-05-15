"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ouraAPI } from "@/lib/api";
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
  ExternalLink,
} from "lucide-react";

const OURA_TASK_KEY = "oura_sync_task_id";

export default function OuraConnect() {
  const qc = useQueryClient();
  const [token, setToken] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [syncTaskId, setSyncTaskIdState] = useState<string | null>(() =>
    typeof window !== "undefined" ? localStorage.getItem(OURA_TASK_KEY) : null
  );
  const [syncStats, setSyncStats] = useState<Record<string, number> | null>(null);

  function setSyncTaskId(id: string | null) {
    if (id) localStorage.setItem(OURA_TASK_KEY, id);
    else localStorage.removeItem(OURA_TASK_KEY);
    setSyncTaskIdState(id);
  }

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ["oura-status"],
    queryFn: ouraAPI.status,
  });

  const connect = useMutation({
    mutationFn: () => ouraAPI.connect(token),
    onSuccess: () => {
      toast.success("Oura Ring connected!");
      setShowForm(false);
      setToken("");
      qc.invalidateQueries({ queryKey: ["oura-status"] });
      sync.mutate();
    },
    onError: (err: any) =>
      toast.error(err?.response?.data?.detail || "Connection failed — check your token"),
  });

  const disconnect = useMutation({
    mutationFn: ouraAPI.disconnect,
    onSuccess: () => {
      toast.success("Oura Ring disconnected");
      setSyncTaskId(null);
      setSyncStats(null);
      qc.invalidateQueries({ queryKey: ["oura-status"] });
    },
    onError: () => toast.error("Failed to disconnect"),
  });

  const sync = useMutation({
    mutationFn: () => ouraAPI.sync(60),
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
        const s = await ouraAPI.syncStatus(syncTaskId);
        if (cancelled) return;
        if (s.status === "completed") {
          setSyncTaskId(null);
          setSyncStats(s.stats);
          toast.success(
            `Sync done — ${s.stats?.health_days_added ?? 0} new days, ${s.stats?.health_days_updated ?? 0} updated`
          );
          qc.invalidateQueries({ queryKey: ["oura-status"] });
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
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-rose-500 to-orange-400 flex items-center justify-center flex-shrink-0">
          <Moon className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1">
          <div className="text-sm font-semibold text-white">Oura Ring</div>
          <div className="text-xs text-slate-500 mt-0.5">Sleep, HRV &amp; readiness</div>
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
                Why connect Oura?
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { icon: <Moon className="w-3.5 h-3.5" />, label: "Sleep stages", desc: "Deep, REM & light" },
                  { icon: <Heart className="w-3.5 h-3.5" />, label: "HRV + RHR", desc: "Overnight averages" },
                  { icon: <Zap className="w-3.5 h-3.5" />, label: "Readiness score", desc: "Daily recovery index" },
                  { icon: <Activity className="w-3.5 h-3.5" />, label: "Sleep score", desc: "Quality rating" },
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
                Oura's overnight HRV and readiness score are fed directly into the AI model — so your
                plan adapts to how recovered you actually are.
              </p>
            </div>

            <div>
              <button
                onClick={() => setShowForm((v) => !v)}
                className="flex items-center gap-1.5 text-sm text-brand-400 hover:text-brand-300 transition-colors"
              >
                {showForm ? "Hide form" : "Enter personal access token"}
              </button>

              {showForm && (
                <form
                  className="mt-4 space-y-3"
                  onSubmit={(e) => { e.preventDefault(); connect.mutate(); }}
                >
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">
                      Oura personal access token
                    </label>
                    <input
                      type="text"
                      required
                      value={token}
                      onChange={(e) => setToken(e.target.value)}
                      placeholder="Paste your token here"
                      className="w-full bg-[#0f1117] border border-surface-border rounded-lg px-3 py-2
                                 text-white text-sm placeholder-slate-600
                                 focus:outline-none focus:ring-2 focus:ring-brand-500 transition"
                    />
                    <p className="text-xs text-slate-500 mt-1.5 flex items-center gap-1">
                      Generate one at{" "}
                      <a
                        href="https://cloud.ouraring.com/personal-access-tokens"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-brand-400 hover:text-brand-300 inline-flex items-center gap-0.5"
                      >
                        cloud.ouraring.com
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </p>
                  </div>
                  <button
                    type="submit"
                    disabled={connect.isPending || !token}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-500
                               hover:bg-brand-600 disabled:opacity-50 text-white text-sm
                               font-medium transition-colors"
                  >
                    {connect.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
                    Connect &amp; Sync
                  </button>
                </form>
              )}
            </div>
          </>
        )}
      </div>
    </div>
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
