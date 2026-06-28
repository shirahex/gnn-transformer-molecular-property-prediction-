/** Dataset statistics card component. */

import { Target, Hash } from "lucide-react";
import type { DatasetInfo } from "@/types";

interface DatasetCardProps {
  datasetKey: string;
  info: DatasetInfo;
}

export default function DatasetCard({ datasetKey, info }: DatasetCardProps) {
  const isRegression = info.task_type === "regression";
  const totalSize = (info.train_size || 0) + (info.val_size || 0) + (info.test_size || 0);

  const stats = [
    {
      label: "Train",
      value: info.train_size || 0,
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
    },
    {
      label: "Val",
      value: info.val_size || 0,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
    },
    {
      label: "Test",
      value: info.test_size || 0,
      color: "text-rose-400",
      bg: "bg-rose-500/10",
    },
  ];

  return (
    <div className="group relative bg-slate-900/50 border border-slate-800/60 rounded-xl overflow-hidden hover:border-cyan-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/5">
      {/* Top accent bar */}
      <div
        className={`h-1 w-full ${isRegression ? "bg-gradient-to-r from-blue-500 to-cyan-500" : "bg-gradient-to-r from-purple-500 to-pink-500"}`}
      />

      <div className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-base font-semibold text-slate-100 group-hover:text-cyan-400 transition-colors">
              {info.name}
            </h3>
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
              {datasetKey}
            </span>
          </div>
          <span
            className={`text-[10px] font-bold px-2 py-1 rounded-full ${
              isRegression
                ? "bg-blue-500/10 text-blue-400"
                : "bg-purple-500/10 text-purple-400"
            }`}
          >
            {isRegression ? "Regression" : "Classification"}
          </span>
        </div>

        {/* Description */}
        <p className="text-xs text-slate-400 mb-4 leading-relaxed line-clamp-2">
          {info.description}
        </p>

        {/* Stats */}
        <div className="flex items-center gap-2 mb-4">
          {stats.map((s) => (
            <div
              key={s.label}
              className={`flex-1 ${s.bg} rounded-lg p-2 text-center`}
            >
              <div className={`text-sm font-bold ${s.color}`}>
                {s.value.toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500 font-medium">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between text-[10px] text-slate-500 pt-3 border-t border-slate-800/40">
          <div className="flex items-center gap-1">
            <Hash className="w-3 h-3" />
            <span>Total: {totalSize.toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-1">
            <Target className="w-3 h-3" />
            <span className="uppercase">{info.metric}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
