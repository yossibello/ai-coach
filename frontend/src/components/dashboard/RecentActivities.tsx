"use client";

import type { Activity } from "@/types";
import { formatDuration, formatDistance, relativeDate } from "@/lib/utils";
import Link from "next/link";
import { Bike } from "lucide-react";

const TYPE_COLORS: Record<string, string> = {
  easy:       "bg-blue-500/10 text-blue-400",
  endurance:  "bg-green-500/10 text-green-400",
  tempo:      "bg-yellow-500/10 text-yellow-400",
  sweetspot:  "bg-orange-400/10 text-orange-400",
  threshold:  "bg-orange-500/10 text-orange-500",
  vo2max:     "bg-red-500/10 text-red-400",
  sprint:     "bg-pink-500/10 text-pink-400",
  race:       "bg-purple-500/10 text-purple-400",
  recovery:   "bg-slate-500/10 text-slate-400",
  long_ride:  "bg-teal-500/10 text-teal-400",
};

interface Props {
  activities: Activity[];
}

export function RecentActivities({ activities }: Props) {
  if (!activities.length) {
    return (
      <div className="text-center py-12 text-slate-500 text-sm">
        No activities yet —{" "}
        <Link href="/upload" className="text-brand-400 hover:underline">
          upload a ride
        </Link>{" "}
        or{" "}
        <Link href="/upload" className="text-brand-400 hover:underline">
          connect Strava
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {activities.map((a) => (
        <Link
          key={a.id}
          href={`/activities/${a.id}`}
          className="flex items-center gap-4 p-3 rounded-xl hover:bg-surface-muted transition-colors group"
        >
          <div className="w-9 h-9 rounded-lg bg-brand-500/10 flex items-center justify-center flex-shrink-0">
            <Bike className="w-4 h-4 text-brand-400" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-white truncate group-hover:text-brand-300 transition-colors">
              {a.name}
            </div>
            <div className="text-xs text-slate-500">{relativeDate(a.date)}</div>
          </div>
          <div className="text-right flex-shrink-0 space-y-0.5">
            <div className="text-sm text-slate-300">{formatDuration(a.duration_seconds)}</div>
            <div className="text-xs text-slate-500">{formatDistance(a.distance_meters)}</div>
          </div>
          {a.workout_type && (
            <span
              className={`hidden sm:inline-flex text-xs px-2 py-0.5 rounded-full font-medium capitalize ${
                TYPE_COLORS[a.workout_type] ?? "bg-slate-500/10 text-slate-400"
              }`}
            >
              {a.workout_type.replace("_", " ")}
            </span>
          )}
          {a.tss !== undefined && (
            <div className="text-xs text-slate-400 hidden md:block w-16 text-right">
              TSS {Math.round(a.tss)}
            </div>
          )}
        </Link>
      ))}
    </div>
  );
}
