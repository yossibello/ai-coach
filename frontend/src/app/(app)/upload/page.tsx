"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { activitiesAPI, stravaAPI } from "@/lib/api";
import { Upload, FileText, CheckCircle, XCircle, Loader2, RefreshCw, Link2 } from "lucide-react";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";
import type { UploadResult } from "@/types";

interface UploadJob {
  file: File;
  progress: number;
  status: "pending" | "uploading" | "done" | "error";
  result?: UploadResult;
}

export default function UploadPage() {
  const [jobs, setJobs] = useState<UploadJob[]>([]);
  const [syncTaskId, setSyncTaskId] = useState<string | null>(null);
  const qc = useQueryClient();

  const onDrop = useCallback(async (accepted: File[]) => {
    const newJobs: UploadJob[] = accepted.map((f) => ({
      file: f,
      progress: 0,
      status: "pending",
    }));
    setJobs((prev) => [...prev, ...newJobs]);

    for (let i = 0; i < newJobs.length; i++) {
      const job = newJobs[i];
      setJobs((prev) =>
        prev.map((j) => (j.file === job.file ? { ...j, status: "uploading" } : j))
      );
      try {
        const result = await activitiesAPI.uploadFile(job.file, (pct) => {
          setJobs((prev) =>
            prev.map((j) => (j.file === job.file ? { ...j, progress: pct } : j))
          );
        });
        setJobs((prev) =>
          prev.map((j) =>
            j.file === job.file ? { ...j, status: "done", progress: 100, result } : j
          )
        );
        if (result.status === "success") {
          toast.success(`Uploaded: ${job.file.name}`);
          qc.invalidateQueries({ queryKey: ["activities"] });
          qc.invalidateQueries({ queryKey: ["fitness"] });
        }
      } catch {
        setJobs((prev) =>
          prev.map((j) => (j.file === job.file ? { ...j, status: "error" } : j))
        );
        toast.error(`Failed: ${job.file.name}`);
      }
    }
  }, [qc]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/gpx+xml": [".gpx"],
      "application/octet-stream": [".fit"],
      "application/xml": [".gpx", ".tcx"],
    },
    maxSize: 50 * 1024 * 1024,  // 50 MB
  });

  async function handleStravaSync() {
    try {
      const { task_id } = await stravaAPI.syncHistory();
      setSyncTaskId(task_id);
      toast.success("Strava sync started — this may take a few minutes");
    } catch {
      toast.error("Strava sync failed. Is your account connected?");
    }
  }

  async function connectStrava() {
    const url = await stravaAPI.getAuthURL();
    window.location.href = url;
  }

  return (
    <div className="p-6 space-y-8 animate-fade-in max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Upload & Sync</h1>
        <p className="text-slate-400 text-sm mt-1">
          Import rides from Strava, Garmin, or upload GPX / FIT files directly
        </p>
      </div>

      {/* Strava section */}
      <div className="bg-surface-card border border-surface-border rounded-2xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 bg-orange-500/10 rounded-lg flex items-center justify-center">
            <span className="text-orange-400 font-bold text-sm">S</span>
          </div>
          <div>
            <h2 className="font-semibold text-white">Strava</h2>
            <p className="text-xs text-slate-400">Import your full activity history</p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={connectStrava}
            className="flex items-center gap-2 px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white text-sm font-medium rounded-xl transition-colors"
          >
            <Link2 className="w-4 h-4" />
            Connect Strava
          </button>
          <button
            onClick={handleStravaSync}
            disabled={!!syncTaskId}
            className="flex items-center gap-2 px-4 py-2 bg-surface border border-surface-border hover:border-orange-500/50 text-slate-300 text-sm rounded-xl transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn("w-4 h-4", syncTaskId && "animate-spin")} />
            {syncTaskId ? "Syncing…" : "Sync history"}
          </button>
        </div>
        {syncTaskId && <SyncProgress taskId={syncTaskId} />}
      </div>

      {/* File upload */}
      <div>
        <h2 className="font-semibold text-white mb-3">Upload Files</h2>
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all",
            isDragActive
              ? "border-brand-500 bg-brand-500/10"
              : "border-surface-border hover:border-brand-500/50 hover:bg-surface-card"
          )}
        >
          <input {...getInputProps()} />
          <Upload className={cn("w-8 h-8 mx-auto mb-3", isDragActive ? "text-brand-400" : "text-slate-500")} />
          <p className="text-white font-medium mb-1">
            {isDragActive ? "Drop to upload" : "Drag & drop rides here"}
          </p>
          <p className="text-sm text-slate-400">Supports .gpx, .fit, .tcx · Max 50 MB per file</p>
        </div>
      </div>

      {/* Upload queue */}
      {jobs.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-slate-400">Upload queue</h3>
          {jobs.map((job, i) => (
            <UploadJobRow key={i} job={job} />
          ))}
        </div>
      )}
    </div>
  );
}

function UploadJobRow({ job }: { job: UploadJob }) {
  const STATUS_ICON = {
    pending:   <FileText className="w-4 h-4 text-slate-400" />,
    uploading: <Loader2 className="w-4 h-4 text-brand-400 animate-spin" />,
    done:      <CheckCircle className="w-4 h-4 text-brand-400" />,
    error:     <XCircle className="w-4 h-4 text-red-400" />,
  };

  return (
    <div className="flex items-center gap-3 bg-surface-card border border-surface-border rounded-xl px-4 py-3 text-sm">
      {STATUS_ICON[job.status]}
      <span className="flex-1 truncate text-slate-300">{job.file.name}</span>
      {job.status === "uploading" && (
        <span className="text-xs text-slate-400">{job.progress}%</span>
      )}
      {job.result?.status === "duplicate" && (
        <span className="text-xs text-slate-500">Already exists</span>
      )}
      {job.status === "error" && (
        <span className="text-xs text-red-400">Failed</span>
      )}
    </div>
  );
}

function SyncProgress({ taskId }: { taskId: string }) {
  const { data } = useQuery({
    queryKey: ["strava-sync", taskId],
    queryFn: () => stravaAPI.getSyncStatus(taskId),
    refetchInterval: (query) => {
      const d = query.state.data;
      return d?.status === "completed" || d?.status === "failed" ? false : 3000;
    },
  });

  if (!data) return null;

  const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
        <span className="capitalize">{data.status}</span>
        <span>{data.progress} / {data.total} activities</span>
      </div>
      <div className="h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className="h-full bg-brand-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
