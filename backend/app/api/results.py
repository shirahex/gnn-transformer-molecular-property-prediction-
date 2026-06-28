"""Results/benchmark endpoints for model comparison."""

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/results", tags=["results"])


@router.get("")
async def get_all_results() -> dict[str, Any]:
    """Get benchmark results for all trained models across all datasets.

    Returns:
        Dictionary with results table and summary statistics.
    """
    results = []

    for dataset_name in config.DATASETS:
        for model_name in config.AVAILABLE_MODELS:
            model_dir = config.MODEL_DIR / dataset_name
            checkpoint_path = model_dir / f"{model_name}_best.pt"

            if checkpoint_path.exists():
                try:
                    import torch
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    task_type = checkpoint.get("task_type", "regression")

                    # Look for test metrics file
                    test_metrics = checkpoint.get("test_metrics", checkpoint.get("final_metrics", {}))

                    entry = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "task_type": task_type,
                        "epochs_trained": checkpoint.get("epoch", 0),
                        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
                        "metrics": test_metrics,
                    }
                    results.append(entry)
                except Exception as e:
                    logger.warning("Could not load checkpoint %s: %s", checkpoint_path, e)
                    results.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "error": str(e),
                    })
            else:
                # No trained model yet
                task_type = config.DATASETS[dataset_name]["task_type"]
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "task_type": task_type,
                    "status": "not_trained",
                    "metrics": {},
                })

    return {
        "results": results,
        "datasets": list(config.DATASETS.keys()),
        "models": config.AVAILABLE_MODELS,
        "count": len(results),
    }


@router.get("/{dataset_name}")
async def get_dataset_results(dataset_name: str) -> dict[str, Any]:
    """Get results for a specific dataset."""
    if dataset_name not in config.DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    results = []
    for model_name in config.AVAILABLE_MODELS:
        model_dir = config.MODEL_DIR / dataset_name
        checkpoint_path = model_dir / f"{model_name}_best.pt"

        if checkpoint_path.exists():
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                test_metrics = checkpoint.get("test_metrics", checkpoint.get("final_metrics", {}))
                results.append({
                    "model": model_name,
                    "metrics": test_metrics,
                    "epochs_trained": checkpoint.get("epoch", 0),
                })
            except Exception as e:
                results.append({"model": model_name, "error": str(e)})
        else:
            results.append({"model": model_name, "status": "not_trained"})

    return {
        "dataset": dataset_name,
        "task_type": config.DATASETS[dataset_name]["task_type"],
        "results": results,
    }
