"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { garminAPI } from "@/lib/api";
import toast from "react-hot-toast";
import {
  Wifi,
  WifiOff,
  KeyRound,
  Lock,
  RefreshCw,
  Trash2,
  Loader2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

type Method = "credentials" | "oauth";

const GARMIN_TASK_KEY = "garmin_sync_task_id";

export default function GarminConnect() {
  const qc = useQueryClient();
  const [method, setMethod] = useState<Method>("credentials");
  const [showForm, setShowForm] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [syncTaskId, setSyncTaskIdState] = useState<string | null>(() =>
    typeof window !== "undefined" ? localStorage.getItem(GARMIN_TASK_KEY) : null
  );
  const [syncStats, setSyncStats] = useState<Record<string, number> | null>(null);

  function setSyncTaskId(id: string | null) {
    if (id) localStorage.setItem(GARMIN_TASK_KEY, id);
    else localStorage.removeItem(GARMIN_TASK_KEY);
    setSyncTaskIdState(id);
  }

  // ── Status ─────────────────────────────────────────────────────────────────
  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ["garmin-status"],
    queryFn: garminAPI.status,
    refetchInterval: syncTaskId ? 3000 : false,
  });

  // ── Connect (credentials) ──────────────────────────────────────────────────
  const connect = useMutation({
    mutationFn: () => garminAPI.connectCredentials(username, password),
    onSuccess: () => {
      toast.success("Garmin connected!");
      setShowForm(false);
      setPassword("");
      qc.invalidateQueries({ queryKey: ["garmin-status"] });
      // Auto-start sync
      sync.mutate();
    },
    onError: () => toast.error("Connection failed — check your credentials"),
  });

  // ── Disconnect ─────────────────────────────────────────────────────────────
  const disconnect = useMutation({
    mutationFn: garminAPI.disconnect,
    onSuccess: () => {
      toast.success("Garmin disconnected");
      setSyncTaskId(null);
      setSyncStats(null);
      qc.invalidateQueries({ queryKey: ["garmin-status"] });
    },
    onError: () => toast.error("Failed to disconnect"),
  });

  // ── Sync ───────────────────────────────────────────────────────────────────
  const sync = useMutation({
    mutationFn: () => garminAPI.sync(60),
    onSuccess: (data) => {
      setSyncTaskId(data.task_id);
      setSyncStats(null);
    },
    onError: () => toast.error("Sync failed to start"),
  });

  // ── Poll sync status ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!syncTaskId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const s = await garminAPI.syncStatus(syncTaskId);
        if (cancelled) return;
        if (s.status === "completed") {
          setSyncTaskId(null);
          setSyncStats(s.stats);
          toast.success(
            `Sync done — ${s.stats?.activities_added ?? 0} new activities, ${s.stats?.health_days_added ?? 0} health days`
          );
          qc.invalidateQueries({ queryKey: ["garmin-status"] });
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

  // ── Render ─────────────────────────────────────────────────────────────────
  const isSyncing = !!syncTaskId || sync.isPending;
  const connected = status?.connected ?? false;

  return (
    <div className="rounded-2xl border border-surface-border bg-surface-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-surface-border">
        <img
          src="/garmin-logo.svg"
          alt="Garmin"
          className="w-7 h-7 object-contain"
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
        <div className="flex-1">
          <div className="text-sm font-semibold text-white">Garmin Connect</div>
          <div className="text-xs text-slate-500 mt-0.5">Sync activities and health data</div>
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
      <div className="px-6 py-5 space-y-4">
        {connected ? (
          <>
            {/* Connected state */}
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Wifi className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <span>
                Logged in as <strong className="text-white">{status?.username}</strong>
              </span>
            </div>

            {syncStats && (
              <div className="grid grid-cols-2 gap-2 text-xs">
                <Stat label="New activities" value={syncStats.activities_added} />
                <Stat label="Health days" value={syncStats.health_days_added} />
                <Stat label="Duplicates skipped" value={syncStats.duplicates_resolved} />
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
            {/* Method picker */}
            <div className="grid grid-cols-2 gap-3">
              <MethodCard
                icon={<KeyRound className="w-5 h-5" />}
                title="Username & Password"
                description="Connect directly with your Garmin credentials"
                selected={method === "credentials"}
                onClick={() => { setMethod("credentials"); setShowForm(true); }}
              />
              <MethodCard
                icon={<Lock className="w-5 h-5" />}
                title="OAuth (Official)"
                description="Requires Garmin partner approval — coming soon"
                selected={method === "oauth"}
                disabled
                badge="Coming Soon"
                onClick={() => {}}
              />
            </div>

            {/* Credentials form */}
            {method === "credentials" && (
              <div>
                <button
                  onClick={() => setShowForm((v) => !v)}
                  className="flex items-center gap-1.5 text-sm text-brand-400 hover:text-brand-300 transition-colors"
                >
                  {showForm ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  {showForm ? "Hide form" : "Enter credentials"}
                </button>

                {showForm && (
                  <form
                    className="mt-4 space-y-3"
                    onSubmit={(e) => { e.preventDefault(); connect.mutate(); }}
                  >
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        Garmin Connect email
                      </label>
                      <input
                        type="email"
                        required
                        autoComplete="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="you@example.com"
                        className="w-full bg-[#0f1117] border border-surface-border rounded-lg px-3 py-2
                                   text-white text-sm placeholder-slate-600
                                   focus:outline-none focus:ring-2 focus:ring-brand-500 transition"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        Password
                      </label>
                      <input
                        type="password"
                        required
                        autoComplete="current-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••"
                        className="w-full bg-[#0f1117] border border-surface-border rounded-lg px-3 py-2
                                   text-white text-sm placeholder-slate-600
                                   focus:outline-none focus:ring-2 focus:ring-brand-500 transition"
                      />
                      <p className="text-xs text-slate-500 mt-1.5">
                        Stored encrypted (AES-256). Never sent to third parties.
                      </p>
                    </div>
                    <button
                      type="submit"
                      disabled={connect.isPending || !username || !password}
                      className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-500
                                 hover:bg-brand-600 disabled:opacity-50 text-white text-sm
                                 font-medium transition-colors"
                    >
                      {connect.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
                      Connect & Sync
                    </button>
                  </form>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function MethodCard({
  icon,
  title,
  description,
  selected,
  disabled,
  badge,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  selected: boolean;
  disabled?: boolean;
  badge?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`
        relative text-left rounded-xl border p-4 transition-all
        ${disabled
          ? "border-surface-border opacity-50 cursor-not-allowed"
          : selected
          ? "border-brand-500 bg-brand-500/10"
          : "border-surface-border hover:border-slate-500"
        }
      `}
    >
      {badge && (
        <span className="absolute top-2 right-2 text-[10px] font-semibold px-1.5 py-0.5
                         bg-slate-700 text-slate-300 rounded-full">
          {badge}
        </span>
      )}
      <div className={`mb-2 ${selected ? "text-brand-400" : "text-slate-400"}`}>{icon}</div>
      <div className="text-sm font-medium text-white leading-tight">{title}</div>
      <div className="text-xs text-slate-500 mt-1 leading-snug">{description}</div>
    </button>
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
