import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function formatDistance(meters: number): string {
  const km = meters / 1000;
  return km >= 10 ? `${km.toFixed(0)} km` : `${km.toFixed(1)} km`;
}

export function formatPower(watts: number): string {
  return `${Math.round(watts)} W`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function relativeDate(iso: string): string {
  const act = new Date(iso);
  const today = new Date();
  // Compare calendar dates so "2 days ago" matches what Strava shows regardless
  // of the exact time of day the activity was recorded.
  const actDay  = new Date(act.getFullYear(),   act.getMonth(),   act.getDate());
  const todayDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const days = Math.round((todayDay.getTime() - actDay.getTime()) / 86_400_000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
  return formatDate(iso);
}

/** Coggan power zones based on FTP */
export function getPowerZone(watts: number, ftp: number): 1 | 2 | 3 | 4 | 5 | 6 | 7 {
  const pct = watts / ftp;
  if (pct < 0.55) return 1;
  if (pct < 0.75) return 2;
  if (pct < 0.90) return 3;
  if (pct < 1.05) return 4;
  if (pct < 1.20) return 5;
  if (pct < 1.50) return 6;
  return 7;
}

export const ZONE_LABELS: Record<number, string> = {
  1: "Recovery",
  2: "Endurance",
  3: "Tempo",
  4: "Threshold",
  5: "VO2max",
  6: "Anaerobic",
  7: "Neuromuscular",
};

export const ZONE_COLORS: Record<number, string> = {
  1: "#60a5fa",
  2: "#4ade80",
  3: "#facc15",
  4: "#fb923c",
  5: "#f87171",
  6: "#c084fc",
  7: "#f9a8d4",
};

/** TSB interpretation */
export function getTSBStatus(tsb: number): { label: string; color: string } {
  if (tsb > 25)  return { label: "Very Fresh",   color: "#60a5fa" };
  if (tsb > 5)   return { label: "Fresh",         color: "#4ade80" };
  if (tsb >= -10) return { label: "Optimal",      color: "#22c55e" };
  if (tsb >= -30) return { label: "Tired",        color: "#facc15" };
  return           { label: "Very Tired",          color: "#f87171" };
}
