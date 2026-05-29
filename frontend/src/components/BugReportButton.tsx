"use client";

import { useState, useRef } from "react";
import { usePathname } from "next/navigation";
import { Bug, X, Paperclip, Send, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";

const PAGE_LABELS: Record<string, string> = {
  "/dashboard":  "Dashboard",
  "/coach":      "AI Coach",
  "/activities": "Activities",
  "/nutrition":  "Nutrition",
  "/upload":     "Upload",
  "/profile":    "Profile",
};

function pageLabel(pathname: string): string {
  for (const [prefix, label] of Object.entries(PAGE_LABELS)) {
    if (pathname.startsWith(prefix)) return label;
  }
  return pathname;
}

export default function BugReportButton() {
  const pathname  = usePathname();
  const [open, setOpen]         = useState(false);
  const [desc, setDesc]         = useState("");
  const [severity, setSeverity] = useState<"low" | "medium" | "high">("medium");
  const [file, setFile]         = useState<File | null>(null);
  const [loading, setLoading]   = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const page = pageLabel(pathname);

  async function submit() {
    if (!desc.trim()) { toast.error("Please describe the bug"); return; }
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("page",        page);
      fd.append("description", desc.trim());
      fd.append("severity",    severity);
      if (file) fd.append("screenshot", file);

      const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
      const res = await fetch("/api/v1/bugs", {
        method:  "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body:    fd,
      });
      if (!res.ok) throw new Error("Failed");
      toast.success("Bug reported — thank you!");
      setOpen(false);
      setDesc("");
      setFile(null);
      setSeverity("medium");
    } catch {
      toast.error("Could not submit report — try again");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      {/* Floating trigger button */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-24 right-4 md:bottom-6 z-50 flex items-center gap-1.5 px-3 py-2 rounded-full bg-surface-card border border-surface-border text-slate-400 hover:text-red-400 hover:border-red-500/40 shadow-lg transition-all text-xs font-medium"
        title="Report a bug"
      >
        <Bug className="w-3.5 h-3.5" />
        <span className="hidden sm:inline">Report bug</span>
      </button>

      {/* Modal overlay */}
      {open && (
        <div className="fixed inset-0 z-[60] flex items-end sm:items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)} />
          <div className="relative w-full max-w-md bg-surface-card border border-surface-border rounded-2xl p-5 space-y-4 shadow-2xl">

            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Bug className="w-4 h-4 text-red-400" />
                <span className="font-semibold text-white">Report a bug</span>
              </div>
              <button onClick={() => setOpen(false)} className="text-slate-500 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Page (readonly) */}
            <div>
              <label className="text-xs text-slate-500 mb-1 block">Page</label>
              <div className="text-sm text-slate-300 bg-surface border border-surface-border rounded-lg px-3 py-2">
                {page}
              </div>
            </div>

            {/* Description */}
            <div>
              <label className="text-xs text-slate-500 mb-1 block">What happened? <span className="text-red-400">*</span></label>
              <textarea
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                rows={4}
                placeholder="Describe the bug — what did you expect vs. what happened?"
                className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-brand-500/50 resize-none"
              />
            </div>

            {/* Severity + screenshot row */}
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <label className="text-xs text-slate-500 mb-1 block">Severity</label>
                <div className="flex gap-2">
                  {(["low", "medium", "high"] as const).map((s) => (
                    <button
                      key={s}
                      onClick={() => setSeverity(s)}
                      className={cn(
                        "flex-1 text-xs py-1.5 rounded-lg border capitalize transition-all",
                        severity === s
                          ? s === "high"   ? "border-red-500/50 bg-red-500/10 text-red-300"
                          : s === "medium" ? "border-amber-500/50 bg-amber-500/10 text-amber-300"
                          :                  "border-blue-500/50 bg-blue-500/10 text-blue-300"
                          : "border-surface-border text-slate-500 hover:text-slate-300"
                      )}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              {/* Screenshot */}
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Screenshot</label>
                <button
                  onClick={() => fileRef.current?.click()}
                  className={cn(
                    "flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all",
                    file ? "border-brand-500/50 bg-brand-500/10 text-brand-300" : "border-surface-border text-slate-500 hover:text-slate-300"
                  )}
                >
                  <Paperclip className="w-3 h-3" />
                  {file ? file.name.slice(0, 12) + "…" : "Attach"}
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </div>
            </div>

            {/* Submit */}
            <button
              onClick={submit}
              disabled={loading || !desc.trim()}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-white text-sm font-semibold transition-all"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              {loading ? "Sending…" : "Submit report"}
            </button>

          </div>
        </div>
      )}
    </>
  );
}
