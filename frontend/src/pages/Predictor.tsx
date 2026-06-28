/** Predictor page for molecular property prediction. */

import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  FlaskConical,
  AlertTriangle,
  Info,
  ChevronRight,
  Beaker,
} from "lucide-react";
import { molecularApi } from "@/services/api";
import type { PredictResponse } from "@/types";
import PredictionForm from "@/components/PredictionForm";
import MoleculeViewer from "@/components/MoleculeViewer";

export default function Predictor() {
  const [searchParams] = useSearchParams();
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<string[]>([]);

  useEffect(() => {
    molecularApi
      .getDatasets()
      .then((d) => setDatasets(Object.keys(d.datasets)))
      .catch(() => setDatasets(["esol", "bbbp", "freesolv", "clintox"]));
  }, []);

  // Handle ?smiles= from query params
  useEffect(() => {
    const smilesParam = searchParams.get("smiles");
    if (smilesParam) {
      handlePredict({ smiles: smilesParam, model: "gat", dataset: "esol" });
    }
  }, [searchParams]);

  const handlePredict = async (request: { smiles: string; model: string; dataset: string }) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await molecularApi.predict(request);
      setResult(response);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/10 flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-cyan-400" />
          </div>
          Molecular Property Predictor
        </h1>
        <p className="text-sm text-slate-400 mt-1 ml-12">
          Enter a SMILES string to predict molecular properties with confidence intervals.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Input Form */}
        <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <Beaker className="w-4 h-4 text-cyan-400" />
            Input
          </h2>
          <PredictionForm
            datasets={datasets}
            onPredict={handlePredict}
            loading={loading}
            demoMode={result?.demo_mode}
          />
        </div>

        {/* Right: Molecule Viewer */}
        <div>
          <MoleculeViewer
            smiles={result?.smiles || "CCO"}
            model={result?.model || "gat"}
            dataset={result?.dataset || "esol"}
            showHeatmap={!!result}
          />
        </div>
      </div>

      {/* Results */}
      {error && (
        <div className="bg-rose-500/5 border border-rose-500/20 rounded-xl p-4 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <div>
            <p className="text-sm text-rose-400">{error}</p>
            <p className="text-xs text-slate-500 mt-1">
              Make sure the backend is running and models are trained.
            </p>
          </div>
        </div>
      )}

      {result && !error && (
        <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-6 space-y-6">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Info className="w-4 h-4 text-cyan-400" />
            Prediction Results
          </h2>

          {/* Main prediction */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-slate-950/50 rounded-lg p-4 text-center border border-slate-800/40">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-1">
                Prediction
              </div>
              <div className="text-2xl font-bold text-cyan-400 font-mono">
                {typeof result.prediction === "number"
                  ? result.prediction.toFixed(4)
                  : result.prediction}
              </div>
              <div className="text-[10px] text-slate-500 mt-1">{result.task_type}</div>
            </div>

            <div className="bg-slate-950/50 rounded-lg p-4 text-center border border-slate-800/40">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-1">
                Uncertainty (Std)
              </div>
              <div className="text-2xl font-bold text-amber-400 font-mono">
                {typeof result.uncertainty === "number"
                  ? result.uncertainty.toFixed(4)
                  : "-"}
              </div>
              <div className="text-[10px] text-slate-500 mt-1">MC Dropout</div>
            </div>

            <div className="bg-slate-950/50 rounded-lg p-4 text-center border border-slate-800/40">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-1">
                CI Lower (95%)
              </div>
              <div className="text-lg font-bold text-emerald-400 font-mono">
                {typeof result.ci_lower === "number" ? result.ci_lower.toFixed(4) : "-"}
              </div>
            </div>

            <div className="bg-slate-950/50 rounded-lg p-4 text-center border border-slate-800/40">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-1">
                CI Upper (95%)
              </div>
              <div className="text-lg font-bold text-rose-400 font-mono">
                {typeof result.ci_upper === "number" ? result.ci_upper.toFixed(4) : "-"}
              </div>
            </div>
          </div>

          {/* Molecule Info */}
          {result.molecule_info && Object.keys(result.molecule_info).length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3">
                Molecule Properties
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(result.molecule_info).map(([key, value]) => (
                  <div
                    key={key}
                    className="bg-slate-950/30 rounded-lg p-3 border border-slate-800/30"
                  >
                    <div className="text-[10px] uppercase text-slate-500 font-medium">{key}</div>
                    <div className="text-sm font-semibold text-slate-200 font-mono mt-0.5">
                      {value}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model info */}
          <div className="flex items-center gap-2 text-xs text-slate-500 pt-2 border-t border-slate-800/40">
            <ChevronRight className="w-3 h-3" />
            <span className="uppercase font-medium">{result.model}</span>
            <span>/</span>
            <span className="uppercase">{result.dataset}</span>
            {result.demo_mode && (
              <span className="text-amber-400 ml-2">(Demo mode — model not trained)</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
