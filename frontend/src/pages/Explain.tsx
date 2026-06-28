/** Explainability page for interactive molecule analysis. */

import { useState } from "react";
import {
  Microscope,
  Search,
  AlertTriangle,
  Loader2,
  Atom,
  Zap,
  Info,
} from "lucide-react";
import { molecularApi } from "@/services/api";
import type { ExplainResponse } from "@/types";
import MoleculeViewer from "@/components/MoleculeViewer";

const SAMPLE_SMILES = [
  { label: "Aspirin", value: "CC(=O)OC1=CC=CC=C1C(=O)O" },
  { label: "Caffeine", value: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" },
  { label: "Paracetamol", value: "CC(=O)Nc1ccc(O)cc1" },
];

const MODELS = [
  { value: "gat", label: "GAT", desc: "Attention-based" },
  { value: "gcn", label: "GCN", desc: "Gradient-based" },
  { value: "chemberta", label: "ChemBERTa-2", desc: "Token attention" },
];

export default function Explain() {
  const [smiles, setSmiles] = useState("CC(=O)OC1=CC=CC=C1C(=O)O");
  const [model, setModel] = useState("gat");
  const [dataset, setDataset] = useState("esol");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExplain = async () => {
    if (!smiles.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await molecularApi.explain(smiles.trim(), model, dataset);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Explanation failed");
    } finally {
      setLoading(false);
    }
  };

  // Get atom scores from attention or integrated gradients
  const atomScores =
    result?.attention?.atom_scores ||
    result?.integrated_gradients?.atom_scores ||
    {};

  // Sort atoms by score for the table
  const sortedAtoms = Object.entries(atomScores)
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 10);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/10 flex items-center justify-center">
            <Microscope className="w-5 h-5 text-cyan-400" />
          </div>
          Explainability Analysis
        </h1>
        <p className="text-sm text-slate-400 mt-1 ml-12">
          Visualize which atoms and substructures drive model predictions.
        </p>
      </div>

      {/* Input */}
      <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Search className="w-4 h-4 text-cyan-400" />
          Molecule Input
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <label className="text-xs text-slate-400 mb-1 block">SMILES String</label>
            <div className="relative">
              <input
                type="text"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Enter SMILES..."
                className="w-full bg-slate-950/30 border border-slate-700 rounded-lg px-4 py-3 text-sm text-slate-100 font-mono focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500/60"
              />
            </div>
            <div className="flex gap-2 mt-2">
              {SAMPLE_SMILES.map((s) => (
                <button
                  key={s.value}
                  onClick={() => setSmiles(s.value)}
                  className="text-[10px] px-2 py-1 rounded bg-slate-800/50 text-slate-400 hover:text-cyan-400 hover:bg-slate-800 transition-colors"
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-400 mb-1 block">Model</label>
              <div className="grid grid-cols-3 gap-1">
                {MODELS.map((m) => (
                  <button
                    key={m.value}
                    onClick={() => setModel(m.value)}
                    className={`px-2 py-2 rounded-lg text-xs font-medium border transition-all ${
                      model === m.value
                        ? "bg-cyan-500/10 border-cyan-500/40 text-cyan-400"
                        : "bg-slate-950/30 border-slate-800/40 text-slate-400 hover:border-slate-600"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs text-slate-400 mb-1 block">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full bg-slate-950/30 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 capitalize"
              >
                {["esol", "bbbp", "freesolv", "clintox"].map((d) => (
                  <option key={d} value={d}>
                    {d.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleExplain}
              disabled={loading || !smiles.trim()}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-medium py-2.5 rounded-lg transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Analyze
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-xs text-rose-400 bg-rose-500/5 border border-rose-500/10 rounded-lg px-4 py-3">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Molecule with Heatmap */}
          <div>
            <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
              <Atom className="w-4 h-4 text-cyan-400" />
              Attention Heatmap
            </h3>
            <MoleculeViewer
              smiles={smiles}
              model={model}
              dataset={dataset}
              showHeatmap={true}
              width={500}
              height={500}
            />
            {result.demo_mode && (
              <p className="text-xs text-amber-400 mt-2">
                <Info className="w-3 h-3 inline mr-1" />
                {result.message}
              </p>
            )}
          </div>

          {/* Atom Importance Table */}
          <div>
            <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
              <Microscope className="w-4 h-4 text-cyan-400" />
              Atom Importance Ranking
            </h3>

            {sortedAtoms.length > 0 ? (
              <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-950/60">
                    <tr>
                      <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">
                        Rank
                      </th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">
                        Atom Index
                      </th>
                      <th className="text-right px-4 py-3 text-xs font-medium text-slate-400 uppercase">
                        Score
                      </th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-slate-400 uppercase">
                        Visualization
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedAtoms.map(([idx, score], i) => {
                      const scoreVal = score as number;
                      const barColor =
                        scoreVal > 0.7
                          ? "bg-rose-500"
                          : scoreVal > 0.4
                          ? "bg-amber-500"
                          : "bg-cyan-500";
                      return (
                        <tr
                          key={idx}
                          className="border-t border-slate-800/30 hover:bg-slate-800/20"
                        >
                          <td className="px-4 py-2.5 text-xs text-slate-500">
                            #{i + 1}
                          </td>
                          <td className="px-4 py-2.5 text-xs font-bold text-slate-200">
                            Atom {idx}
                          </td>
                          <td className="px-4 py-2.5 text-right">
                            <span className="text-xs font-mono font-bold text-slate-200">
                              {scoreVal.toFixed(4)}
                            </span>
                          </td>
                          <td className="px-4 py-2.5">
                            <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${barColor} rounded-full transition-all`}
                                style={{ width: `${scoreVal * 100}%` }}
                              />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-8 text-center text-slate-500 text-sm">
                No atom scores available. Train the model and try again.
              </div>
            )}

            {/* Method info */}
            {result.attention?.method && (
              <div className="mt-4 bg-slate-900/30 border border-slate-800/40 rounded-lg p-4">
                <h4 className="text-xs font-semibold text-slate-300 mb-2">
                  Explanation Method
                </h4>
                <p className="text-xs text-slate-400">
                  Using{" "}
                  <span className="text-cyan-400 font-medium">
                    {result.attention.method === "gradient"
                      ? "Gradient-based Importance"
                      : "Multi-head Attention Weights"}
                  </span>
                  {result.attention.num_layers
                    ? ` from ${result.attention.num_layers} GAT layers`
                    : ""}
                  . Higher scores indicate atoms that contributed more to the prediction.
                </p>
              </div>
            )}

            {/* Integrated Gradients info */}
            {result.integrated_gradients && (
              <div className="mt-3 bg-slate-900/30 border border-slate-800/40 rounded-lg p-4">
                <h4 className="text-xs font-semibold text-slate-300 mb-2">
                  Integrated Gradients
                </h4>
                <p className="text-xs text-slate-400">
                  Path-integrated gradients from baseline to input, attributing predictions
                  to individual atom features across {result.integrated_gradients.num_atoms} atoms.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
