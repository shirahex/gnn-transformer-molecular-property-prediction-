/** Live training monitor with loss curves and status. */

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { molecularApi } from "@/services/api";
import type { TrainingJob } from "@/types";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingDown,
  Activity,
} from "lucide-react";

interface TrainingMonitorProps {
  jobId: string;
  onComplete?: () => void;
}

export default function TrainingMonitor({ jobId, onComplete }: TrainingMonitorProps) {
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const data = await molecularApi.getTrainingStatus(jobId);
        setJob(data);

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
          if (data.status === "completed" && onComplete) {
            onComplete();
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch status");
      }
    }, 3000);

    // Initial fetch
    molecularApi.getTrainingStatus(jobId).then(setJob).catch(console.error);

    return () => clearInterval(interval);
  }, [jobId]);

  if (!job) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 animate-spin text-cyan-500" />
      </div>
    );
  }

  const isRunning = job.status === "running";
  const isCompleted = job.status === "completed";
  const isFailed = job.status === "failed";

  // Prepare chart data
  const lossData =
    job.result?.history?.train_loss?.map((loss: number, i: number) => ({
      epoch: i + 1,
      train_loss: loss,
      val_loss: job.result?.history?.val_loss?.[i] ?? null,
    })) || [];

  return (
    <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl overflow-hidden">
      {/* Status Header */}
      <div className="px-4 py-3 border-b border-slate-800/40 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isRunning && <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />}
          {isCompleted && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
          {isFailed && <XCircle className="w-4 h-4 text-rose-400" />}
          {!isRunning && !isCompleted && !isFailed && (
            <Clock className="w-4 h-4 text-amber-400" />
          )}
          <div>
            <span className="text-sm font-medium text-slate-200">
              Job {job.job_id}
            </span>
            <span
              className={`ml-2 text-[10px] font-bold px-2 py-0.5 rounded-full ${
                isRunning
                  ? "bg-cyan-500/10 text-cyan-400"
                  : isCompleted
                  ? "bg-emerald-500/10 text-emerald-400"
                  : isFailed
                  ? "bg-rose-500/10 text-rose-400"
                  : "bg-amber-500/10 text-amber-400"
              }`}
            >
              {job.status.toUpperCase()}
            </span>
          </div>
        </div>
        <span className="text-xs text-slate-500">
          {job.model_name} / {job.dataset_name}
        </span>
      </div>

      {/* Progress */}
      {isRunning && (
        <div className="px-4 py-2 border-b border-slate-800/40">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-slate-400">{job.message}</span>
            <span className="text-[10px] text-slate-500">
              {Math.round((job.progress || 0) * 100)}%
            </span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500"
              style={{ width: `${(job.progress || 0) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Loss Chart */}
      {lossData.length > 0 && (
        <div className="p-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-medium text-slate-300">Loss Curves</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="epoch" tick={{ fill: "#64748b", fontSize: 9 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  fontSize: "11px",
                }}
              />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={false}
                name="Train Loss"
              />
              {lossData.some((d) => d.val_loss !== null) && (
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#a78bfa"
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Test Metrics */}
      {job.result?.test_metrics && Object.keys(job.result.test_metrics).length > 0 && (
        <div className="px-4 py-3 border-t border-slate-800/40 bg-slate-950/30">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-medium text-slate-300">Test Metrics</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(job.result.test_metrics).map(([key, value]) => (
              <div
                key={key}
                className="bg-slate-900/60 rounded-lg p-2 text-center"
              >
                <div className="text-xs font-bold text-emerald-400 font-mono">
                  {typeof value === "number" ? value.toFixed(4) : value}
                </div>
                <div className="text-[10px] text-slate-500 uppercase">{key}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="px-4 py-2 text-xs text-rose-400 bg-rose-500/5 border-t border-rose-500/10">
          {error}
        </div>
      )}
    </div>
  );
}
