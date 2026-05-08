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
} from "lucide-react";
import { authAPI } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard",        label: "Dashboard",      icon: LayoutDashboard },
  { href: "/coach",            label: "AI Coach",       icon: Brain },
  { href: "/activities",       label: "Activities",     icon: Activity },
  { href: "/upload",           label: "Upload / Sync",  icon: Upload },
  { href: "/profile",          label: "Profile",        icon: User },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  function handleLogout() {
    authAPI.logout();
    router.push("/login");
  }

  return (
    <div className="min-h-screen bg-surface flex">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-60 flex flex-col bg-surface-card border-r border-surface-border z-40">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-surface-border">
          <Link href="/dashboard" className="flex items-center gap-2">
            <Zap className="w-6 h-6 text-brand-500" />
            <span className="font-bold text-white">AI Coach</span>
          </Link>
        </div>

        {/* Nav */}
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

        {/* Logout */}
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

      {/* Main content */}
      <main className="flex-1 ml-60 min-h-screen">{children}</main>
    </div>
  );
}
