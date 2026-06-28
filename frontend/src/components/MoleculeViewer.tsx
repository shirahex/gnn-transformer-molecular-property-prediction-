/** Interactive molecule viewer with attention heatmap overlay. */

import { useState, useEffect } from "react";
import { molecularApi } from "@/services/api";
import { Loader2, AlertCircle, RefreshCw, ImageIcon } from "lucide-react";

interface MoleculeViewerProps {
  smiles: string;
  model?: string;
  dataset?: string;
  showHeatmap?: boolean;
  width?: number;
  height?: number;
}

export default function MoleculeViewer({
  smiles,
  model = "gat",
  dataset = "esol",
  showHeatmap = false,
  width = 400,
  height = 400,
}: MoleculeViewerProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!smiles) return;

    let cancelled = false;

    const loadImage = async () => {
      setLoading(true);
      setError(null);
      try {
        let blob: Blob;
        if (showHeatmap) {
          blob = await molecularApi.getHeatmap(smiles, model, dataset, width, height);
        } else {
          blob = await molecularApi.renderMolecule(smiles, width, height);
        }
        if (!cancelled) {
          setImageUrl(URL.createObjectURL(blob));
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to render molecule");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    loadImage();

    return () => {
      cancelled = true;
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [smiles, model, dataset, showHeatmap, width, height]);

  if (!smiles) {
    return (
      <div className="flex items-center justify-center h-64 bg-slate-900/40 border border-slate-800/60 rounded-xl">
        <p className="text-sm text-slate-500">Enter a SMILES string to view molecule</p>
      </div>
    );
  }

  return (
    <div className="relative bg-slate-900/40 border border-slate-800/60 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800/40 bg-slate-900/60">
        <div className="flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-medium text-slate-300">2D Structure</span>
          {showHeatmap && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Attention
            </span>
          )}
        </div>
        <button
          onClick={() => {
            setImageUrl(null);
            setError(null);
          }}
          className="p-1 rounded hover:bg-slate-800 text-slate-500 hover:text-cyan-400 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-3 h-3" />
        </button>
      </div>

      {/* Content */}
      <div className="flex items-center justify-center p-4 min-h-[300px]">
        {loading && (
          <div className="flex flex-col items-center gap-2 text-slate-500">
            <Loader2 className="w-8 h-8 animate-spin text-cyan-500" />
            <span className="text-xs">Rendering molecule...</span>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center gap-2 text-rose-400">
            <AlertCircle className="w-8 h-8" />
            <span className="text-xs text-center max-w-[200px]">{error}</span>
            <p className="text-[10px] text-slate-500 font-mono mt-1">{smiles}</p>
          </div>
        )}

        {imageUrl && !loading && (
          <img
            src={imageUrl}
            alt={`Molecule: ${smiles}`}
            className="max-w-full h-auto rounded-lg"
            style={{ maxHeight: height }}
          />
        )}
      </div>

      {/* SMILES display */}
      <div className="px-4 py-2 border-t border-slate-800/40 bg-slate-950/40">
        <p className="text-[10px] text-slate-500 font-mono truncate" title={smiles}>
          {smiles}
        </p>
      </div>
    </div>
  );
}
