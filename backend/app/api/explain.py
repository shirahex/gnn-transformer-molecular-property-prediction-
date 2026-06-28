"""Explainability endpoints for attention visualization and molecular rendering."""

import logging
from typing import Any, Optional

import torch
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app import config
from app.data.preprocessor import smiles_to_graph
from app.explainability.attention_viz import AttentionVisualizer
from app.explainability.molecule_renderer import MoleculeRenderer
from app.explainability.integrated_gradients import IntegratedGradientsExplainer
from app.models.gat import GATModel
from app.models.gcn import GCNModel
from app.models.chemberta import ChemBERTaModel
from app.utils.helpers import validate_smiles

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/explain", tags=["explainability"])


def _load_model(model_name: str, dataset_name: str):
    """Load a trained model from checkpoint."""
    from app.api.predict import _load_model as load_fn
    return load_fn(model_name, dataset_name)


@router.get("/{smiles}")
async def explain_molecule(smiles: str, model: str = "gat", dataset: str = "esol") -> dict[str, Any]:
    """Get attention/importance visualization data for a molecule.

    Args:
        smiles: SMILES string.
        model: Model to use for explanation.
        dataset: Dataset the model was trained on.

    Returns:
        Dictionary with atom scores and attention data.
    """
    is_valid, error = validate_smiles(smiles)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {error}")

    model_instance, error = _load_model(model, dataset)
    if error:
        # Return empty structure for demo
        return {
            "smiles": smiles,
            "model": model,
            "dataset": dataset,
            "demo_mode": True,
            "message": error,
            "attention": {
                "atom_scores": {},
                "method": "none",
            },
        }

    try:
        visualizer = AttentionVisualizer(model, config.DEVICE)
        attention_data = visualizer.get_attention(model_instance, smiles)

        # Also compute integrated gradients
        ig_explainer = IntegratedGradientsExplainer(model_instance, config.DEVICE)
        ig_data = ig_explainer.explain(smiles)

        return {
            "smiles": smiles,
            "model": model,
            "dataset": dataset,
            "attention": attention_data,
            "integrated_gradients": ig_data,
        }
    except Exception as e:
        logger.error("Explanation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/{smiles}/render")
async def render_molecule(
    smiles: str,
    width: int = 400,
    height: int = 400,
    highlight_atoms: Optional[str] = None,
) -> Response:
    """Render a molecule as PNG image.

    Args:
        smiles: SMILES string.
        width: Image width in pixels.
        height: Image height in pixels.
        highlight_atoms: Comma-separated atom indices to highlight.

    Returns:
        PNG image response.
    """
    is_valid, error = validate_smiles(smiles)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {error}")

    atom_scores = None
    if highlight_atoms:
        try:
            indices = [int(x) for x in highlight_atoms.split(",")]
            # Uniform highlighting if no scores provided
            atom_scores = {i: 0.7 for i in indices}
        except ValueError:
            pass

    try:
        renderer = MoleculeRenderer(width, height)
        png_bytes = renderer.render(smiles, atom_scores=atom_scores)

        if png_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to render molecule")

        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Render error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")


@router.get("/{smiles}/heatmap")
async def get_attention_heatmap(
    smiles: str,
    model: str = "gat",
    dataset: str = "esol",
    width: int = 500,
    height: int = 500,
) -> Response:
    """Render a molecule with attention heatmap overlay as PNG.

    Args:
        smiles: SMILES string.
        model: Model to use for attention extraction.
        dataset: Dataset the model was trained on.
        width: Image width.
        height: Image height.

    Returns:
        PNG image with heatmap overlay.
    """
    is_valid, error = validate_smiles(smiles)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {error}")

    model_instance, load_error = _load_model(model, dataset)

    atom_scores = None
    if model_instance is not None:
        try:
            visualizer = AttentionVisualizer(model, config.DEVICE)
            attention_data = visualizer.get_attention(model_instance, smiles)
            atom_scores = attention_data.get("atom_scores")
        except Exception as e:
            logger.warning("Could not extract attention: %s", e)

    try:
        renderer = MoleculeRenderer(width, height)
        png_bytes = renderer.render(
            smiles,
            atom_scores=atom_scores,
            highlight_functional_groups=True,
        )

        if png_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to render molecule")

        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Heatmap render error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Heatmap rendering failed: {str(e)}")
