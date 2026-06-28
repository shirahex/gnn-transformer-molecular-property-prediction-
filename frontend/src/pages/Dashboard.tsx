/** Dashboard page with project overview and dataset statistics. */

import { useEffect, useState } from "react";
import {
  FlaskConical,
  Database,
  BrainCircuit,
  TrendingUp,
  Activity,
  Layers,
  ArrowRight,
} from "lucide-react";
import { molecularApi } from "@/services/api";
import type { DatasetResponse, HealthResponse } from "@/types";
import DatasetCard from "@/components/DatasetCard";
import { Link } from "react-router-dom";

const FEATURES = [
  {
    icon: BrainCircuit,
    title: "Graph Neural Networks",
    desc: "GCN and GAT models process molecular graphs with atom-level features and bond connectivity.",
    color: "text-cyan-400",
    bg: "bg-cyan-500/10",
  },
  {
    icon: Layers,
    title: "Transformer Models",
    desc: "ChemBERTa-2 leverages pre-trained RoBERTa on 10M SMILES for rich molecular representations.",
    color: "text-purple-400",
    bg: "bg-purple-500/10",
  },
  {
    icon: Activity,
    title: "Explainable AI",
    desc: "Attention visualization and integrated gradients reveal which substructures drive predictions.",
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
  },
  {
    icon: TrendingUp,
    title: "Uncertainty Quantification",
    desc: "Monte Carlo Dropout provides confidence intervals for reliable decision-making.",
    color: "text-amber-400",
    bg: "bg-amber-500/10",
  },
];

const SAMPLE_SMILES = [
  { name: "Ethanol", smiles: "CCO", value: "-1.31 logS" },
  { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O", value: "-2.05 logS" },
  { name: "Caffeine", smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", value: "-1.07 logS" },
];

export default function Dashboard() {
  const [datasets, setDatasets] = useState<DatasetResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [ds, h] = await Promise.all([
          molecularApi.getDatasets(),
          molecularApi.getHealth(),
        ]);
        setDatasets(ds);
        setHealth(h);
      } catch (e) {
        setError("Backend connection failed. Start the backend server on port 8000.");
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-slate-700/50 p-8">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_rgba(6,182,212,0.08),_transparent_50%)]" />
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <FlaskConical className="w-6 h-6 text-white" />
            </div>
            <span className="text-xs font-bold tracking-widest uppercase text-cyan-400">
              Drug Discovery Platform
            </span>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">
            Molecular Property Prediction
          </h1>
          <p className="text-slate-400 max-w-2xl leading-relaxed">
            Compare Graph Neural Networks (GCN, GAT) and Transformer-based representations
            (ChemBERTa-2) with Explainable AI for predicting molecular properties in drug discovery.
          </p>

          {health && (
            <div className="mt-4 flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5 text-emerald-400">
                <Activity className="w-3 h-3" />
                Backend: {health.status}
              </span>
              <span className="text-slate-500">Device: {health.device}</span>
              <span className="text-slate-500">CUDA: {health.cuda_available ? "Yes" : "No"}</span>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4 text-sm text-amber-400">
          {error}
        </div>
      )}

      {/* Features */}
      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <BrainCircuit className="w-5 h-5 text-cyan-400" />
          Platform Capabilities
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="group bg-slate-900/40 border border-slate-800/60 rounded-xl p-5 hover:border-cyan-500/30 transition-all"
            >
              <div className={`w-10 h-10 rounded-lg ${f.bg} flex items-center justify-center mb-3`}>
                <f.icon className={`w-5 h-5 ${f.color}`} />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-1">{f.title}</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Datasets */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
            <Database className="w-5 h-5 text-cyan-400" />
            MoleculeNet Datasets
          </h2>
          <Link
            to="/training"
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 transition-colors"
          >
            Start Training <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-pulse">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-48 bg-slate-900/40 rounded-xl" />
            ))}
          </div>
        ) : datasets?.datasets ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(datasets.datasets).map(([key, info]) => (
              <DatasetCard key={key} datasetKey={key} info={info} />
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">No datasets available</p>
        )}
      </div>

      {/* Quick Predict */}
      <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-cyan-400" />
          Quick Predictions
        </h2>
        <p className="text-xs text-slate-400 mb-4">
          Try predicting properties for these well-known molecules:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {SAMPLE_SMILES.map((m) => (
            <Link
              key={m.smiles}
              to={`/predictor?smiles=${encodeURIComponent(m.smiles)}`}
              className="group bg-slate-950/40 border border-slate-800/40 rounded-lg p-3 hover:border-cyan-500/30 transition-all"
            >
              <div className="text-sm font-medium text-slate-200 group-hover:text-cyan-400 transition-colors">
                {m.name}
              </div>
              <div className="text-[10px] text-slate-500 font-mono truncate mt-1">{m.smiles}</div>
              <div className="text-xs text-cyan-400 mt-1">{m.value}</div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
