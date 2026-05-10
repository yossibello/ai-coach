"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useQueryClient } from "@tanstack/react-query";
import { activitiesAPI } from "@/lib/api";
import { Upload, FileText, CheckCircle, XCircle, Loader2, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";
import Link from "next/link";
import type { UploadResult } from "@/types";

interface UploadJob {
  file: File;
  progress: number;
  status: "pending" | "uploading" | "done" | "error";
  result?: UploadResult;
}

export default function UploadPage() {
  const [jobs, setJobs] = useState<UploadJob[]>([]);
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
    maxSize: 50 * 1024 * 1024,
  });

  return (
    <div className="p-6 space-y-8 animate-fade-in max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Upload</h1>
        <p className="text-slate-400 text-sm mt-1">
          Import rides manually by uploading GPX, FIT, or TCX files.
        </p>
      </div>

      <div className="flex items-center gap-3 px-4 py-3 bg-surface-card border border-surface-border rounded-xl text-sm text-slate-400">
        <Settings className="w-4 h-4 flex-shrink-0 text-brand-400" />
        <span>
          To sync from Garmin or Strava, go to{" "}
          <Link href="/profile" className="text-brand-400 hover:text-brand-300 underline underline-offset-2">
            Profile &rarr; Integrations
          </Link>
          .
        </span>
      </div>

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
          <p className="text-sm text-slate-400">Supports .gpx, .fit, .tcx &middot; Max 50 MB per file</p>
        </div>
      </div>

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
