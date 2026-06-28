/** Results page with benchmark comparison table and charts. */

import { useState, useEffect } from "react";
import {
  FileText,
  Trophy,
  AlertCircle,
  Download,
  BarChart3,
  Radar,
} from "lucide-react";
import { molecularApi } from "@/services/api";
import type { BenchmarkResponse, BenchmarkResult } from "@/types";
import BenchmarkChart from "@/components/BenchmarkChart";

export default function Results() {
  const [results, setResults] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartType, setChartType] = useState<"bar" | "radar">("bar");
  const [selectedDataset, setSelectedDataset] = useState<string>("all");

  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    setLoading(true);
    try {
      const data = await molecularApi.getResults();
      setResults(data);
    } catch (e) {
      setError("Failed to load results. Ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const filteredResults =
    selectedDataset === "all"
      ? results?.results || []
      : (results?.results || []).filter((r) => r.dataset === selectedDataset);

  const exportCSV = () => {
    if (!results?.results.length) return;

    const headers = ["dataset", "model", "task_type", "status", ...Object.keys(results.results[0]?.metrics || {})];
    const rows = results.results.map((r) =>
      [
        r.dataset,
        r.model,
        r.task_type,
        r.status || "trained",
        ...Object.values(r.metrics || {}),
      ].join(",")
    );

    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "benchmark_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const getPrimaryMetric = (r: BenchmarkResult) => {
    if (r.task_type === "regression") {
      return { label: "RMSE", value: r.metrics?.rmse ?? "-" };
    }
    return { label: "ROC-AUC", value: r.metrics?.roc_auc ?? "-" };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 flex items-center justify-center">
              <FileText className="w-5 h-5 text-cyan-400" />
            </div>
            Benchmark Results
          </h1>
          <p className="text-sm text-slate-400 mt-1 ml-12">
            Compare model performance across datasets and metrics.
          </p>
        </div>
        <button
          onClick={exportCSV}
          disabled={!results?.results?.length}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 text-sm rounded-lg transition-colors border border-slate-700"
        >
          <Download className="w-4 h-4" />
          Export CSV
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-xs text-rose-400 bg-rose-500/5 border border-rose-500/10 rounded-lg px-4 py-3">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center gap-3">
        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none"
        >
          <option value="all">All Datasets</option>
          {results?.datasets?.map((d) => (
            <option key={d} value={d} className="capitalize">
              {d.toUpperCase()}
            </option>
          ))}
        </select>

        <div className="flex bg-slate-900/60 border border-slate-700 rounded-lg overflow-hidden">
          <button
            onClick={() => setChartType("bar")}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors ${
              chartType === "bar"
                ? "bg-cyan-500/10 text-cyan-400"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Bar
          </button>
          <button
            onClick={() => setChartType("radar")}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors ${
              chartType === "radar"
                ? "bg-cyan-500/10 text-cyan-400"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Radar className="w-3.5 h-3.5" />
            Radar
          </button>
        </div>
      </div>

      {/* Chart */}
      <BenchmarkChart results={filteredResults} chartType={chartType} />

      {/* Results Table */}
      <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-800/40 flex items-center gap-2">
          <Trophy className="w-4 h-4 text-amber-400" />
          <h2 className="text-sm font-semibold text-slate-200">Detailed Results</h2>
        </div>

        {loading ? (
          <div className="p-8 text-center text-slate-500 animate-pulse">Loading results...</div>
        ) : filteredResults.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-sm text-slate-500 mb-2">No results available yet.</p>
            <p className="text-xs text-slate-600">
              Train models on the Training page to generate benchmark data.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-950/60">
                <tr>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Dataset</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Model</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Type</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">Status</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-slate-400 uppercase">Primary Metric</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">All Metrics</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map((r, i) => {
                  const primary = getPrimaryMetric(r);
                  return (
                    <tr
                      key={`${r.dataset}-${r.model}-${i}`}
                      className="border-t border-slate-800/30 hover:bg-slate-800/20 transition-colors"
                    >
                      <td className="px-4 py-3 text-xs font-bold uppercase text-slate-200">
                        {r.dataset}
                      </td>
                      <td className="px-4 py-3 text-xs uppercase text-cyan-400 font-medium">
                        {r.model}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                            r.task_type === "regression"
                              ? "bg-blue-500/10 text-blue-400"
                              : "bg-purple-500/10 text-purple-400"
                          }`}
                        >
                          {r.task_type}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                            r.status === "not_trained"
                              ? "bg-slate-500/10 text-slate-400"
                              : "bg-emerald-500/10 text-emerald-400"
                          }`}
                        >
                          {r.status || "trained"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-sm font-bold font-mono text-slate-200">
                          {typeof primary.value === "number" ? primary.value.toFixed(4) : primary.value}
                        </span>
                        <span className="text-[10px] text-slate-500 ml-1">{primary.label}</span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(r.metrics || {}).map(([k, v]) => (
                            <span
                              key={k}
                              className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800/50 text-slate-400 font-mono"
                            >
                              {k}: {typeof v === "number" ? v.toFixed(3) : v}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
