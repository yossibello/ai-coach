"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fitnessAPI, activitiesAPI, coachAPI, stravaAPI } from "@/lib/api";
import { FitnessChart } from "@/components/dashboard/FitnessChart";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { RecentActivities } from "@/components/dashboard/RecentActivities";
import { CoachTip } from "@/components/dashboard/CoachTip";
import { HealthWidget } from "@/components/dashboard/HealthWidget";
import HealthTrendsChart from "@/components/dashboard/HealthTrendsChart";
import { healthAPI } from "@/lib/api";
import { getTSBStatus, formatPower } from "@/lib/utils";
import { TrendingUp, Zap, Heart, Activity, RefreshCw, Sparkles } from "lucide-react";
import { toast } from "react-hot-toast";

export default function DashboardPage() {
  const qc = useQueryClient();

  const { data: fitness } = useQuery({
    queryKey: ["fitness"],
    queryFn: () => fitnessAPI.getProgression(16),
  });

  const { data: activities } = useQuery({
    queryKey: ["activities", 1],
    queryFn: () => activitiesAPI.list({ page: 1, limit: 5 }),
  });

  const { data: coach } = useQuery({
    queryKey: ["coach-recommendation"],
    queryFn: () => coachAPI.getRecommendation(),
  });

  const { data: healthData } = useQuery({
    queryKey: ["health-recent-30"],
    queryFn: () => healthAPI.recent(30),
    staleTime: 5 * 60 * 1000,
  });

  const refreshRec = useMutation({
    mutationFn: () => coachAPI.refreshRecommendation(),
    onSuccess: (data) => {
      qc.setQueryData(["coach-recommendation"], data);
      toast.success("Recommendation refreshed");
    },
    onError: () => toast.error("Could not refresh recommendation"),
  });

  const estimateFTP = useMutation({
    mutationFn: () => stravaAPI.estimateFTP(),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["fitness"] });
      toast.success(data.message);
    },
    onError: () => toast.error("Could not estimate FTP"),
  });

  const current = fitness?.current;
  const tsbStatus = current ? getTSBStatus(current.tsb) : null;

  const ftpMeta = estimateFTP.data ?? (current?.ftp_meta ? {
    method: current.ftp_meta.method,
    confidence: current.ftp_meta.confidence,
    sample_count: current.ftp_meta.sample_count,
    trend: current.ftp_meta.trend,
    best_ride_age_days: current.ftp_meta.best_ride_age_days,
    last_test_age_days: current.ftp_meta.last_test_age_days,
    confidence_low: current.ftp_meta.confidence_low,
    confidence_high: current.ftp_meta.confidence_high,
  } : null);

  const METHOD_LABELS: Record<string, string> = {
    power_weighted:  "Power curve",
    blended:         "Blended w/ profile",
    manual_fallback: "Profile FTP (no rides)",
    test_blend:      "Test blend",
    test_anchored:   "Test anchored",
  };

  const ftpSub = (() => {
    if (!ftpMeta) return "Functional Threshold Power";
    const m = ftpMeta.method ?? "";
    return m.startsWith("verified_test") ? "Verified test" : (METHOD_LABELS[m] ?? m);
  })();

  const ftpDetail = (() => {
    if (!ftpMeta) return undefined;
    const parts: string[] = [];
    if (ftpMeta.sample_count > 0) parts.push(`${ftpMeta.sample_count} rides`);
    if (ftpMeta.last_test_age_days != null) parts.push(`test ${ftpMeta.last_test_age_days}d ago`);
    else if (ftpMeta.best_ride_age_days != null) parts.push(`best ride ${ftpMeta.best_ride_age_days}d ago`);
    if (ftpMeta.confidence_low && ftpMeta.confidence_high)
      parts.push(`range ${ftpMeta.confidence_low}–${ftpMeta.confidence_high}W`);
    if (ftpMeta.trend && ftpMeta.trend !== "stable") parts.push(ftpMeta.trend);
    return parts.length > 0 ? parts.join(" · ") : undefined;
  })();

  const ftpConfidence = ftpMeta?.confidence ?? undefined;

  return (
    <div className="p-6 space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-slate-400 text-sm mt-1">Your training at a glance</p>
        </div>
        <div className="flex gap-2 flex-shrink-0">
          <button
            onClick={() => estimateFTP.mutate()}
            disabled={estimateFTP.isPending}
            title="Auto-detect FTP from your best power on long rides"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-surface-card border border-surface-border text-slate-300 hover:text-white disabled:opacity-50"
          >
            <Sparkles className={`w-3.5 h-3.5 text-brand-400 ${estimateFTP.isPending ? "animate-pulse" : ""}`} />
            Estimate FTP
          </button>
          <button
            onClick={() => refreshRec.mutate()}
            disabled={refreshRec.isPending}
            title="Re-run the AI model with your latest data"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-surface-card border border-surface-border text-slate-300 hover:text-white disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshRec.isPending ? "animate-spin" : ""}`} />
            Refresh AI
          </button>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={Zap}
          label="FTP"
          value={current ? formatPower(current.ftp) : "—"}
          sub={ftpSub}
          detail={ftpDetail}
          confidence={ftpConfidence}
          color="text-brand-400"
        />
        <MetricCard
          icon={TrendingUp}
          label="Fitness (CTL)"
          value={current ? `${Math.round(current.ctl)}` : "—"}
          sub={`ATL ${current ? Math.round(current.atl) : "—"} · 42-day load`}
          color="text-blue-400"
        />
        <MetricCard
          icon={Activity}
          label="Form (TSB)"
          value={current ? `${Math.round(current.tsb) > 0 ? "+" : ""}${Math.round(current.tsb)}` : "—"}
          sub={tsbStatus?.label ?? "Training Stress Balance"}
          color={tsbStatus?.color ?? "text-slate-400"}
        />
        <MetricCard
          icon={Heart}
          label="Today's TSS"
          value={current ? `${Math.round(current.tss)}` : "0"}
          sub="Training Stress Score"
          color="text-pink-400"
        />
      </div>

      {/* Coach tip */}
      {coach && <CoachTip recommendation={coach} />}

      {/* Recovery & Readiness (HRV / RHR / Sleep / Body Battery) */}
      <HealthWidget />

      {/* Health trend charts */}
      {healthData?.days && healthData.days.length > 0 && (
        <HealthTrendsChart days={healthData.days} />
      )}

      {/* PMC Chart */}
      <div className="bg-surface-card border border-surface-border rounded-2xl p-6">
        <h2 className="font-semibold text-white mb-4">Performance Management Chart</h2>
        {fitness?.history ? (
          <FitnessChart data={fitness.history} />
        ) : (
          <div className="h-56 flex items-center justify-center text-slate-500 text-sm">
            No data yet — upload your first ride or sync Strava
          </div>
        )}
      </div>

      {/* Recent activities */}
      <div className="bg-surface-card border border-surface-border rounded-2xl p-6">
        <h2 className="font-semibold text-white mb-4">Recent Activities</h2>
        <RecentActivities activities={activities?.items ?? []} />
      </div>
    </div>
  );
}
