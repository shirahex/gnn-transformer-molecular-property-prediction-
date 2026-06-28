/** Training page for starting and monitoring training jobs. */

import { useState, useEffect, useCallback } from "react";
import {
  TrendingUp,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Clock,
  Zap,
  Settings,
} from "lucide-react";
import { molecularApi } from "@/services/api";
import type { TrainingJob } from "@/types";
import TrainingMonitor from "@/components/TrainingMonitor";

const MODELS = [
  { value: "gcn", label: "GCN", desc: "Graph Convolutional Network — fast baseline" },
  { value: "gat", label: "GAT", desc: "Graph Attention — interpretable attention weights" },
  { value: "chemberta", label: "ChemBERTa-2", desc: "Transformer — rich SMILES representations" },
];

export default function Training() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("gat");
  const [selectedDataset, setSelectedDataset] = useState("esol");
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [starting, setStarting] = useState(false);
  const [activeJobs, setActiveJobs] = useState<string[]>([]);
  const [jobHistory, setJobHistory] = useState<TrainingJob[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    molecularApi
      .getDatasets()
      .then((d) => {
        const keys = Object.keys(d.datasets);
        setDatasets(keys);
        if (keys.length > 0) setSelectedDataset(keys[0]);
      })
      .catch(() => setDatasets(["esol", "bbbp", "freesolv", "clintox"]));

    // Poll for job history
    const interval = setInterval(() => {
      molecularApi
        .listTrainingJobs()
        .then((j) => setJobHistory(j.jobs))
        .catch(() => {});
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      const response = await molecularApi.startTraining(
        selectedModel,
        selectedDataset,
        epochs,
        batchSize,
        lr
      );
      setActiveJobs((prev) => [...prev, response.job_id]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start training");
    } finally {
      setStarting(false);
    }
  };

  const handleJobComplete = useCallback(() => {
    // Refresh history
    molecularApi
      .listTrainingJobs()
      .then((j) => setJobHistory(j.jobs))
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/10 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
          </div>
          Model Training
        </h1>
        <p className="text-sm text-slate-400 mt-1 ml-12">
          Train GNN and Transformer models on MoleculeNet datasets.
        </p>
      </div>

      {/* Training Config */}
      <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <Settings className="w-4 h-4 text-cyan-400" />
          Training Configuration
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {/* Model */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Model
            </label>
            <div className="space-y-2">
              {MODELS.map((m) => (
                <button
                  key={m.value}
                  onClick={() => setSelectedModel(m.value)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm border transition-all ${
                    selectedModel === m.value
                      ? "bg-cyan-500/10 border-cyan-500/40 text-cyan-400"
                      : "bg-slate-950/30 border-slate-800/40 text-slate-400 hover:border-slate-700"
                  }`}
                >
                  <div className="font-medium">{m.label}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">{m.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Dataset */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Dataset
            </label>
            <div className="space-y-2">
              {datasets.map((d) => (
                <button
                  key={d}
                  onClick={() => setSelectedDataset(d)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm border capitalize transition-all ${
                    selectedDataset === d
                      ? "bg-purple-500/10 border-purple-500/40 text-purple-400"
                      : "bg-slate-950/30 border-slate-800/40 text-slate-400 hover:border-slate-700"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Hyperparameters */}
          <div className="space-y-4">
            <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Hyperparameters
            </label>

            <div>
              <label className="text-xs text-slate-500">Epochs</label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Math.max(1, Math.min(500, Number(e.target.value))))}
                className="w-full mt-1 bg-slate-950/30 border border-slate-800/40 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500/40"
              />
            </div>

            <div>
              <label className="text-xs text-slate-500">Batch Size</label>
              <select
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full mt-1 bg-slate-950/30 border border-slate-800/40 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none"
              >
                {[8, 16, 32, 64, 128].map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-slate-500">Learning Rate</label>
              <select
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
                className="w-full mt-1 bg-slate-950/30 border border-slate-800/40 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none"
              >
                {[0.1, 0.01, 0.001, 0.0001, 0.00001].map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Start Button */}
        <button
          onClick={handleStart}
          disabled={starting || datasets.length === 0}
          className="flex items-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg shadow-cyan-500/20"
        >
          {starting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Starting...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Start Training
            </>
          )}
        </button>

        {error && (
          <div className="mt-4 flex items-center gap-2 text-xs text-rose-400 bg-rose-500/5 border border-rose-500/10 rounded-lg px-3 py-2">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}
      </div>

      {/* Active Jobs */}
      {activeJobs.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            Active Training Jobs
          </h2>
          <div className="space-y-4">
            {activeJobs.map((jobId) => (
              <TrainingMonitor
                key={jobId}
                jobId={jobId}
                onComplete={handleJobComplete}
              />
            ))}
          </div>
        </div>
      )}

      {/* Job History */}
      {jobHistory.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            Training History
          </h2>
          <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-950/60">
                <tr>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Job ID</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Model</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Dataset</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Status</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Message</th>
                </tr>
              </thead>
              <tbody>
                {jobHistory.map((job) => (
                  <tr key={job.job_id} className="border-t border-slate-800/30">
                    <td className="px-4 py-2.5 text-xs font-mono text-slate-300">{job.job_id}</td>
                    <td className="px-4 py-2.5 text-xs uppercase text-slate-300">{job.model_name}</td>
                    <td className="px-4 py-2.5 text-xs uppercase text-slate-300">{job.dataset_name}</td>
                    <td className="px-4 py-2.5">
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                          job.status === "completed"
                            ? "bg-emerald-500/10 text-emerald-400"
                            : job.status === "failed"
                            ? "bg-rose-500/10 text-rose-400"
                            : job.status === "running"
                            ? "bg-cyan-500/10 text-cyan-400"
                            : "bg-amber-500/10 text-amber-400"
                        }`}
                      >
                        {job.status}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-xs text-slate-400 truncate max-w-[200px]">
                      {job.message}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
