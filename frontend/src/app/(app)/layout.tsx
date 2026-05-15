"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  LayoutDashboard,
  Brain,
  Upload,
  User,
  Zap,
  LogOut,
  Activity,
  FlaskConical,
} from "lucide-react";
import { authAPI } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard",  label: "Dashboard",   icon: LayoutDashboard },
  { href: "/coach",      label: "AI Coach",    icon: Brain },
  { href: "/activities", label: "Activities",  icon: Activity },
  { href: "/nutrition",  label: "Nutrition",   icon: FlaskConical },
  { href: "/upload",     label: "Upload",      icon: Upload },
  { href: "/profile",    label: "Profile",     icon: User },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  function handleLogout() {
    authAPI.logout();
    router.push("/login");
  }

  return (
    <div className="min-h-screen bg-surface">

      {/* ── Desktop sidebar (md and up) ───────────────────────────────────── */}
      <aside className="hidden md:flex fixed inset-y-0 left-0 w-60 flex-col bg-surface-card border-r border-surface-border z-40">
        <div className="h-16 flex items-center px-6 border-b border-surface-border">
          <Link href="/dashboard" className="flex items-center gap-2">
            <Zap className="w-6 h-6 text-brand-500" />
            <span className="font-bold text-white">AI Coach</span>
          </Link>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
                  active
                    ? "bg-brand-500/15 text-brand-400 border border-brand-500/20"
                    : "text-slate-400 hover:text-white hover:bg-surface-muted"
                )}
              >
                <Icon className="w-4 h-4" />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="px-3 py-4 border-t border-surface-border">
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm text-slate-400 hover:text-red-400 hover:bg-red-400/10 transition-all"
          >
            <LogOut className="w-4 h-4" />
            Sign out
          </button>
        </div>
      </aside>

      {/* ── Mobile top bar (below md) ──────────────────────────────────────── */}
      <header className="md:hidden fixed top-0 left-0 right-0 h-14 bg-surface-card border-b border-surface-border z-40 flex items-center px-4 gap-3">
        <Zap className="w-5 h-5 text-brand-500 shrink-0" />
        <span className="font-bold text-white flex-1">AI Coach</span>
        <button
          onClick={handleLogout}
          className="p-2 text-slate-400 hover:text-red-400 transition-colors"
          aria-label="Sign out"
        >
          <LogOut className="w-5 h-5" />
        </button>
      </header>

      {/* ── Main content ───────────────────────────────────────────────────── */}
      {/* Desktop: offset by sidebar width. Mobile: full width, padded for top/bottom bars. */}
      <main className="md:ml-60 min-h-screen pt-14 md:pt-0 pb-20 md:pb-0">
        {children}
      </main>

      {/* ── Mobile bottom tab bar (below md) ──────────────────────────────── */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-surface-card border-t border-surface-border z-40 flex">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex-1 flex flex-col items-center justify-center py-2 gap-0.5 transition-colors",
                active ? "text-brand-400" : "text-slate-500 hover:text-slate-300"
              )}
            >
              <Icon className="w-5 h-5" />
              <span className="text-[10px] font-medium leading-tight">{label}</span>
            </Link>
          );
        })}
      </nav>

    </div>
  );
}
