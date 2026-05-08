"use client";

import { useQuery } from "@tanstack/react-query";
import { fitnessAPI, activitiesAPI, coachAPI } from "@/lib/api";
import { FitnessChart } from "@/components/dashboard/FitnessChart";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { RecentActivities } from "@/components/dashboard/RecentActivities";
import { CoachTip } from "@/components/dashboard/CoachTip";
import { getTSBStatus, formatPower } from "@/lib/utils";
import { TrendingUp, Zap, Heart, Activity } from "lucide-react";

export default function DashboardPage() {
  const { data: fitness } = useQuery({
    queryKey: ["fitness"],
    queryFn: () => fitnessAPI.getProgression(16),
  });

  const { data: activities } = useQuery({
    queryKey: ["activities", 1],
    queryFn: () => activitiesAPI.list(1, 5),
  });

  const { data: coach } = useQuery({
    queryKey: ["coach-recommendation"],
    queryFn: () => coachAPI.getRecommendation(),
  });

  const current = fitness?.current;
  const tsbStatus = current ? getTSBStatus(current.tsb) : null;

  return (
    <div className="p-6 space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1">Your training at a glance</p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={Zap}
          label="FTP"
          value={current ? formatPower(current.ftp) : "—"}
          sub="Functional Threshold Power"
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
