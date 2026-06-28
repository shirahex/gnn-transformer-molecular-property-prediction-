/** Poster generator page for competition-ready poster layout. */

import { useState, useEffect } from "react";
import { FileImage, Loader2 } from "lucide-react";
import { molecularApi } from "@/services/api";
import type { BenchmarkResponse } from "@/types";
import PosterTemplate from "@/components/PosterTemplate";

export default function Poster() {
  const [results, setResults] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    molecularApi
      .getResults()
      .then(setResults)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/10 flex items-center justify-center">
            <FileImage className="w-5 h-5 text-cyan-400" />
          </div>
          Poster Generator
        </h1>
        <p className="text-sm text-slate-400 mt-1 ml-12">
          Generate a competition-ready poster layout with your results.
        </p>
      </div>

      <p className="text-xs text-slate-500 bg-slate-900/40 border border-slate-800/60 rounded-lg px-4 py-3">
        This poster layout includes model architectures, benchmark results, and explainability
        highlights. Click "Export (Print)" to save as PDF using your browser's print dialog.
        For best results, use Chrome and select "Save as PDF" with A0 page size.
      </p>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-cyan-500" />
        </div>
      ) : (
        <PosterTemplate
          results={results?.results || []}
        />
      )}
    </div>
  );
}
