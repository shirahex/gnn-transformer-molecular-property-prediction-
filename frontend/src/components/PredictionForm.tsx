/** SMILES input form with model/dataset selection. */

import { useState } from "react";
import { FlaskConical, AlertCircle, Loader2, Sparkles, ChevronDown } from "lucide-react";
import type { PredictRequest } from "@/types";

interface PredictionFormProps {
  datasets: string[];
  onPredict: (request: PredictRequest) => void;
  loading: boolean;
  demoMode?: boolean;
}

const SAMPLE_SMILES = [
  { label: "Ethanol", value: "CCO" },
  { label: "Aspirin", value: "CC(=O)OC1=CC=CC=C1C(=O)O" },
  { label: "Caffeine", value: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" },
  { label: "Ibuprofen", value: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
  { label: "Paracetamol", value: "CC(=O)Nc1ccc(O)cc1" },
  { label: "Glucose", value: "C(C1C(C(C(C(O1)O)O)O)O)O" },
];

const MODELS = [
  { value: "gcn", label: "GCN", desc: "Graph Convolutional Network" },
  { value: "gat", label: "GAT", desc: "Graph Attention Network" },
  { value: "chemberta", label: "ChemBERTa-2", desc: "Transformer-based" },
];

export default function PredictionForm({ datasets, onPredict, loading, demoMode }: PredictionFormProps) {
  const [smiles, setSmiles] = useState("CCO");
  const [model, setModel] = useState("gat");
  const [dataset, setDataset] = useState(datasets[0] || "esol");
  const [showSamples, setShowSamples] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!smiles.trim()) return;
    onPredict({ smiles: smiles.trim(), model, dataset });
  };

  const applySample = (value: string) => {
    setSmiles(value);
    setShowSamples(false);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* SMILES Input */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          SMILES String
        </label>
        <div className="relative">
          <input
            type="text"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            placeholder="Enter SMILES (e.g., CCO for ethanol)"
            className="w-full bg-slate-900/60 border border-slate-700 rounded-lg px-4 py-3 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500/60 transition-all font-mono"
          />
          {/* Sample selector */}
          <div className="absolute right-2 top-1/2 -translate-y-1/2">
            <button
              type="button"
              onClick={() => setShowSamples(!showSamples)}
              className="text-xs text-slate-500 hover:text-cyan-400 px-2 py-1 rounded hover:bg-slate-800 transition-colors flex items-center gap-1"
            >
              Samples <ChevronDown className="w-3 h-3" />
            </button>
            {showSamples && (
              <div className="absolute right-0 top-full mt-1 w-56 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-20 overflow-hidden">
                {SAMPLE_SMILES.map((s) => (
                  <button
                    key={s.value}
                    type="button"
                    onClick={() => applySample(s.value)}
                    className="w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-slate-800 hover:text-cyan-400 transition-colors flex justify-between"
                  >
                    <span>{s.label}</span>
                    <span className="text-slate-600 font-mono truncate max-w-[120px]">{s.value}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model & Dataset Selectors */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Model</label>
          <div className="grid grid-cols-3 gap-2">
            {MODELS.map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setModel(m.value)}
                className={`px-3 py-2 rounded-lg text-xs font-medium transition-all border ${
                  model === m.value
                    ? "bg-cyan-500/10 border-cyan-500/40 text-cyan-400"
                    : "bg-slate-900/40 border-slate-700/50 text-slate-400 hover:border-slate-600"
                }`}
                title={m.desc}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Dataset</label>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 capitalize"
          >
            {datasets.map((d) => (
              <option key={d} value={d} className="bg-slate-900 capitalize">
                {d}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading || !smiles.trim()}
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-3 rounded-lg transition-all shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30"
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Predicting...
          </>
        ) : (
          <>
            <FlaskConical className="w-4 h-4" />
            Predict Property
          </>
        )}
      </button>

      {demoMode && (
        <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>Models not yet trained. Train a model first for real predictions.</span>
        </div>
      )}
    </form>
  );
}
