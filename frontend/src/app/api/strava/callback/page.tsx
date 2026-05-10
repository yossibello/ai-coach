"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { stravaAPI } from "@/lib/api";

type Phase = "connecting" | "syncing" | "done" | "error";

export default function StravaCallbackPage() {
  const router = useRouter();
  const params = useSearchParams();
  const [phase, setPhase] = useState<Phase>("connecting");
  const [message, setMessage] = useState("Connecting your Strava account…");
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const taskIdRef = useRef<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const ranRef = useRef(false);

  useEffect(() => {
    if (ranRef.current) return;
    ranRef.current = true;

    const code = params.get("code");
    const error = params.get("error");

    if (error) {
      setPhase("error");
      setMessage(`Strava authorization was denied (${error}).`);
      return;
    }
    if (!code) {
      setPhase("error");
      setMessage("Missing authorization code from Strava.");
      return;
    }
    if (typeof window !== "undefined" && !localStorage.getItem("access_token")) {
      router.replace(`/login?next=/api/strava/callback?code=${encodeURIComponent(code)}`);
      return;
    }

    (async () => {
      // Step 1: exchange OAuth code → tokens saved
      try {
        await stravaAPI.exchangeCode(code);
      } catch (e: unknown) {
        setPhase("error");
        const msg = e instanceof Error ? e.message : "Unknown error";
        setMessage(`Failed to connect Strava: ${msg}`);
        return;
      }

      // Step 2: kick off history sync
      setPhase("syncing");
      setMessage("Importing your Strava activities…");

      let taskId: string;
      try {
        const res = await stravaAPI.syncHistory();
        taskId = res.task_id;
        taskIdRef.current = taskId;
      } catch {
        // Sync failed to start — still connected, just go to dashboard
        setPhase("done");
        router.replace("/dashboard?strava=connected");
        return;
      }

      // Step 3: poll progress
      let pollErrors = 0;
      pollRef.current = setInterval(async () => {
        try {
          const s = await stravaAPI.getSyncStatus(taskId);
          pollErrors = 0;
          setProgress(s.progress);
          setTotal(s.total);
          if (s.status === "completed" || s.status === "failed") {
            clearInterval(pollRef.current!);
            setPhase("done");
            setMessage(
              s.status === "completed"
                ? `Imported ${s.progress} activities! Redirecting…`
                : "Sync finished with some errors. Redirecting…"
            );
            setTimeout(() => router.replace("/dashboard?strava=connected"), 1200);
          }
        } catch {
          // If task not found (backend restarted) or repeated errors → give up
          pollErrors++;
          if (pollErrors >= 3) {
            clearInterval(pollRef.current!);
            setPhase("done");
            setMessage("Strava connected! Redirecting…");
            setTimeout(() => router.replace("/upload?strava=connected"), 1200);
          }
        }
      }, 2000);
    })();

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [params, router]);

  const pct = total > 0 ? Math.round((progress / total) * 100) : null;

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-950 p-6">
      <div className="max-w-sm w-full rounded-xl bg-gray-900 border border-gray-800 p-8 text-center">
        {/* Strava logo */}
        <div className="w-12 h-12 bg-orange-500/15 rounded-full flex items-center justify-center mx-auto mb-5">
          <span className="text-orange-400 font-bold text-xl">S</span>
        </div>

        <h1 className="text-lg font-semibold text-white mb-2">
          {phase === "connecting" && "Connecting Strava…"}
          {phase === "syncing"    && "Importing Activities"}
          {phase === "done"       && "All Done!"}
          {phase === "error"      && "Connection Failed"}
        </h1>

        <p className={`text-sm mb-5 ${phase === "error" ? "text-red-400" : "text-gray-400"}`}>
          {message}
        </p>

        {phase === "syncing" && (
          <div className="space-y-2">
            <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
              <div
                className="h-2 bg-orange-500 rounded-full transition-all duration-500"
                style={{ width: pct !== null ? `${pct}%` : "30%", animation: pct === null ? "pulse 1.5s infinite" : undefined }}
              />
            </div>
            <p className="text-xs text-gray-500">
              {pct !== null ? `${progress} / ${total} activities` : "Starting…"}
            </p>
          </div>
        )}

        {phase === "error" && (
          <button
            onClick={() => router.push("/dashboard")}
            className="mt-2 rounded-lg bg-orange-500 px-4 py-2 text-sm text-white hover:bg-orange-600"
          >
            Back to Dashboard
          </button>
        )}
      </div>
    </div>
  );
}
