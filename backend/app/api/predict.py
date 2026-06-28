"""Prediction endpoints for molecular property prediction."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.data.preprocessor import smiles_to_graph, smiles_to_tokens
from app.models.gat import GATModel
from app.models.gcn import GCNModel
from app.models.chemberta import ChemBERTaModel
from app.utils.helpers import validate_smiles

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["prediction"])


class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule")
    model: str = Field(default="gcn", description="Model to use: gcn, gat, chemberta")
    dataset: str = Field(default="esol", description="Dataset the model was trained on")


class PredictResponse(BaseModel):
    smiles: str
    model: str
    dataset: str
    prediction: float
    uncertainty: float
    ci_lower: float
    ci_upper: float
    task_type: str
    molecule_info: dict


def _load_model(model_name: str, dataset_name: str, device: str = config.DEVICE):
    """Load a trained model from checkpoint."""
    model_dir = config.MODEL_DIR / dataset_name
    checkpoint_path = model_dir / f"{model_name}_best.pt"

    if not checkpoint_path.exists():
        checkpoint_path = model_dir / f"{model_name}_latest.pt"

    if not checkpoint_path.exists():
        return None, f"No trained model found at {checkpoint_path}. Please train first."

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        task_type = checkpoint.get("task_type", "regression")
        num_classes = 2 if task_type == "classification" else None

        if model_name == "gcn":
            model = GCNModel(task_type=task_type, num_classes=num_classes)
        elif model_name == "gat":
            model = GATModel(task_type=task_type, num_classes=num_classes)
        elif model_name == "chemberta":
            model = ChemBERTaModel(task_type=task_type, num_classes=num_classes)
        else:
            return None, f"Unknown model: {model_name}"

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"


@router.post("", response_model=PredictResponse)
async def predict(request: PredictRequest) -> dict[str, Any]:
    """Predict molecular property from SMILES string.

    Returns prediction with confidence intervals via MC Dropout.
    """
    # Validate SMILES
    is_valid, error = validate_smiles(request.smiles)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {error}")

    if request.model not in config.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Available: {config.AVAILABLE_MODELS}",
        )

    # Load model
    model, error = _load_model(request.model, request.dataset)
    if error:
        # Return demo mode response
        return {
            "smiles": request.smiles,
            "model": request.model,
            "dataset": request.dataset,
            "prediction": 0.0,
            "uncertainty": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "task_type": config.DATASETS.get(request.dataset, {}).get("task_type", "regression"),
            "molecule_info": {},
            "demo_mode": True,
            "message": error,
        }

    task_type = config.DATASETS.get(request.dataset, {}).get("task_type", "regression")

    try:
        with torch.no_grad():
            if request.model in ("gcn", "gat"):
                data = smiles_to_graph(request.smiles)
                if data is None:
                    raise HTTPException(status_code=400, detail="Could not parse SMILES into graph")

                data = data.to(config.DEVICE)
                out = model(data.x, data.edge_index, data.edge_attr, torch.zeros(data.x.size(0), dtype=torch.long, device=config.DEVICE))

                if task_type == "classification":
                    pred = float(torch.sigmoid(out).item())
                else:
                    pred = float(out.item())

                # MC Dropout uncertainty
                uncertainty_result = model.predict_with_uncertainty(
                    data.x, data.edge_index, data.edge_attr,
                    torch.zeros(data.x.size(0), dtype=torch.long, device=config.DEVICE),
                    num_samples=20,
                )
            else:  # chemberta
                tokens = smiles_to_tokens([request.smiles])
                input_ids = tokens["input_ids"].to(config.DEVICE)
                out = model(input_ids)

                if task_type == "classification":
                    pred = float(torch.sigmoid(out).item())
                else:
                    pred = float(out.item())

                uncertainty_result = model.predict_with_uncertainty(
                    input_ids, num_samples=20,
                )

        std_val = float(uncertainty_result["std"].item())
        ci_lower = float(uncertainty_result["ci_lower"].item())
        ci_upper = float(uncertainty_result["ci_upper"].item())

        if task_type == "classification":
            ci_lower = max(0.0, min(1.0, ci_lower))
            ci_upper = max(0.0, min(1.0, ci_upper))

        from app.utils.helpers import get_molecule_info
        mol_info = get_molecule_info(request.smiles)

        return {
            "smiles": request.smiles,
            "model": request.model,
            "dataset": request.dataset,
            "prediction": pred,
            "uncertainty": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "task_type": task_type,
            "molecule_info": mol_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch")
async def predict_batch(requests: list[PredictRequest]) -> list[dict[str, Any]]:
    """Batch prediction for multiple molecules."""
    results = []
    for req in requests:
        try:
            result = await predict(req)
            results.append(result)
        except HTTPException as e:
            results.append({"smiles": req.smiles, "error": e.detail})
        except Exception as e:
            results.append({"smiles": req.smiles, "error": str(e)})
    return results
