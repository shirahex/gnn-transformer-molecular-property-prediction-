/** Auto poster generator for competition poster layout. */

import { useRef, useCallback } from "react";
import {
  Download,
  FileImage,
  Atom,
  Network,
  BrainCircuit,
  Target,
  TrendingUp,
  Shield,
} from "lucide-react";

interface PosterTemplateProps {
  results: Array<{
    dataset: string;
    model: string;
    task_type: string;
    metrics: Record<string, number>;
  }>;
  onExport?: () => void;
}

export default function PosterTemplate({ results, onExport }: PosterTemplateProps) {
  const posterRef = useRef<HTMLDivElement>(null);

  const handleExport = useCallback(() => {
    if (posterRef.current) {
      // Use html2canvas approach or trigger print
      window.print();
    }
    onExport?.();
  }, [onExport]);

  const bestResults = results.filter((r) => Object.keys(r.metrics).length > 0);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <FileImage className="w-5 h-5 text-cyan-400" />
          Competition Poster
        </h2>
        <button
          onClick={handleExport}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <Download className="w-4 h-4" />
          Export (Print)
        </button>
      </div>

      {/* Poster Preview */}
      <div
        ref={posterRef}
        className="bg-white text-slate-900 rounded-xl overflow-hidden border-4 border-slate-800 shadow-2xl"
        style={{ aspectRatio: "1/1.414", maxWidth: "800px", margin: "0 auto" }}
      >
        {/* Poster Header */}
        <div className="bg-gradient-to-r from-cyan-700 to-blue-800 text-white p-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center">
              <Atom className="w-10 h-10" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">
                Molecular Property Prediction
              </h1>
              <p className="text-cyan-200 text-lg">
                Comparing GNNs and Transformers with Explainable AI for Drug Discovery
              </p>
            </div>
          </div>
          <div className="flex gap-6 text-sm text-cyan-100">
            <span className="flex items-center gap-1">
              <Network className="w-4 h-4" /> Graph Neural Networks
            </span>
            <span className="flex items-center gap-1">
              <BrainCircuit className="w-4 h-4" /> Transformers
            </span>
            <span className="flex items-center gap-1">
              <Target className="w-4 h-4" /> Explainable AI
            </span>
          </div>
        </div>

        <div className="p-8 space-y-6">
          {/* Architecture Section */}
          <div>
            <h2 className="text-xl font-bold text-slate-800 mb-3 flex items-center gap-2">
              <Network className="w-5 h-5 text-cyan-600" />
              Model Architectures
            </h2>
            <div className="grid grid-cols-3 gap-4">
              {[
                {
                  name: "GCN",
                  desc: "Graph Convolutional Network",
                  details: "3 layers + BatchNorm + Global Pooling",
                  icon: Network,
                },
                {
                  name: "GAT",
                  desc: "Graph Attention Network",
                  details: "4-head Multi-Attention + Readout",
                  icon: Target,
                },
                {
                  name: "ChemBERTa-2",
                  desc: "Molecular Transformer",
                  details: "RoBERTa + LoRA Fine-tuning",
                  icon: BrainCircuit,
                },
              ].map((arch) => (
                <div
                  key={arch.name}
                  className="border border-slate-200 rounded-lg p-4"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <arch.icon className="w-5 h-5 text-cyan-600" />
                    <h3 className="font-bold text-slate-700">{arch.name}</h3>
                  </div>
                  <p className="text-sm text-slate-600">{arch.desc}</p>
                  <p className="text-xs text-slate-500 mt-1">{arch.details}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Results Table */}
          <div>
            <h2 className="text-xl font-bold text-slate-800 mb-3 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-cyan-600" />
              Benchmark Results
            </h2>
            <div className="border border-slate-200 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-100">
                  <tr>
                    <th className="text-left px-4 py-2 font-semibold text-slate-700">Dataset</th>
                    <th className="text-left px-4 py-2 font-semibold text-slate-700">Model</th>
                    <th className="text-left px-4 py-2 font-semibold text-slate-700">Type</th>
                    <th className="text-right px-4 py-2 font-semibold text-slate-700">Primary Metric</th>
                  </tr>
                </thead>
                <tbody>
                  {bestResults.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="text-center py-8 text-slate-500">
                        Train models to see benchmark results
                      </td>
                    </tr>
                  ) : (
                    bestResults.map((r, i) => {
                      const primaryMetric =
                        r.task_type === "regression"
                          ? r.metrics.rmse ?? r.metrics.mae ?? "-"
                          : r.metrics.roc_auc ?? r.metrics.accuracy ?? "-";
                      return (
                        <tr
                          key={i}
                          className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}
                        >
                          <td className="px-4 py-2 font-medium uppercase">{r.dataset}</td>
                          <td className="px-4 py-2 uppercase">{r.model}</td>
                          <td className="px-4 py-2">
                            <span
                              className={`text-xs px-2 py-0.5 rounded-full ${
                                r.task_type === "regression"
                                  ? "bg-blue-100 text-blue-700"
                                  : "bg-purple-100 text-purple-700"
                              }`}
                            >
                              {r.task_type}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right font-mono font-bold text-cyan-700">
                            {typeof primaryMetric === "number"
                              ? primaryMetric.toFixed(4)
                              : primaryMetric}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* XAI Section */}
          <div>
            <h2 className="text-xl font-bold text-slate-800 mb-3 flex items-center gap-2">
              <Shield className="w-5 h-5 text-cyan-600" />
              Explainability & Uncertainty
            </h2>
            <div className="grid grid-cols-2 gap-4 text-sm text-slate-600">
              <div className="border border-slate-200 rounded-lg p-4">
                <h4 className="font-bold text-slate-700 mb-2">Attention Visualization</h4>
                <p>
                  GAT attention weights reveal which atoms and bonds drive predictions.
                  Integrated gradients attribute predictions to input features.
                </p>
              </div>
              <div className="border border-slate-200 rounded-lg p-4">
                <h4 className="font-bold text-slate-700 mb-2">Uncertainty Quantification</h4>
                <p>
                  Monte Carlo Dropout provides confidence intervals for each prediction,
                  critical for high-stakes drug discovery decisions.
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center text-xs text-slate-500 pt-4 border-t border-slate-200">
            Molecular AI Platform — Drug Discovery with Explainable Deep Learning
          </div>
        </div>
      </div>
    </div>
  );
}
