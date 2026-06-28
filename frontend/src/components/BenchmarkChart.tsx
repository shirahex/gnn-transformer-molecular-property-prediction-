/** Benchmark comparison charts for model evaluation results. */

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import type { BenchmarkResult } from "@/types";

interface BenchmarkChartProps {
  results: BenchmarkResult[];
  chartType?: "bar" | "radar";
}

const MODEL_COLORS: Record<string, string> = {
  gcn: "#22d3ee",
  gat: "#a78bfa",
  chemberta: "#f472b6",
};

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-xl">
      <p className="text-xs font-semibold text-slate-200 mb-2">{label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
          <span className="text-slate-400 capitalize">{entry.name}:</span>
          <span className="text-slate-200 font-mono">{entry.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

export default function BenchmarkChart({ results, chartType = "bar" }: BenchmarkChartProps) {
  const chartData = useMemo(() => {
    // Group by dataset
    const byDataset: Record<string, Record<string, number>> = {};
    const allModels = new Set<string>();

    for (const r of results) {
      if (r.status === "not_trained" || !r.metrics || Object.keys(r.metrics).length === 0) continue;
      if (!byDataset[r.dataset]) byDataset[r.dataset] = {};

      // Pick the primary metric
      const primaryMetric =
        r.task_type === "regression"
          ? r.metrics.rmse ?? r.metrics.mae ?? 0
          : r.metrics.roc_auc ?? r.metrics.accuracy ?? 0;

      byDataset[r.dataset][r.model] = primaryMetric;
      allModels.add(r.model);
    }

    if (chartType === "radar") {
      // Radar chart data: metrics as axes
      const metricKeys = new Set<string>();
      for (const r of results) {
        if (r.metrics) Object.keys(r.metrics).forEach((k) => metricKeys.add(k));
      }

      return {
        radar: Array.from(metricKeys).map((metric) => {
          const point: Record<string, string | number> = { metric: metric.toUpperCase() };
          for (const r of results) {
            if (r.metrics?.[metric] !== undefined) {
              point[r.model] = r.metrics[metric];
            }
          }
          return point;
        }),
        models: Array.from(allModels),
      };
    }

    // Bar chart data
    return {
      bar: Object.entries(byDataset).map(([dataset, modelScores]) => ({
        dataset: dataset.toUpperCase(),
        ...modelScores,
      })),
      models: Array.from(allModels),
    };
  }, [results, chartType]);

  if (results.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-slate-900/40 border border-slate-800/60 rounded-xl">
        <p className="text-sm text-slate-500">No benchmark data available</p>
      </div>
    );
  }

  if (chartType === "radar" && chartData.radar) {
    return (
      <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-200 mb-4">Metrics Radar</h3>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={chartData.radar}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 10 }} />
            <PolarRadiusAxis tick={{ fill: "#64748b", fontSize: 9 }} />
            {chartData.models.map((model) => (
              <Radar
                key={model}
                name={model.toUpperCase()}
                dataKey={model}
                stroke={MODEL_COLORS[model] || "#94a3b8"}
                fill={MODEL_COLORS[model] || "#94a3b8"}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            ))}
            <Legend
              wrapperStyle={{ fontSize: "11px" }}
              formatter={(value: string) => <span className="text-slate-300">{value}</span>}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-slate-200 mb-4">Model Comparison</h3>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={chartData.bar} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="dataset" tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "11px" }}
            formatter={(value: string) => <span className="text-slate-300 capitalize">{value}</span>}
          />
          {chartData.models.map((model) => (
            <Bar
              key={model}
              dataKey={model}
              fill={MODEL_COLORS[model] || "#94a3b8"}
              radius={[4, 4, 0, 0]}
              name={model}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
