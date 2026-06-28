"""Dataset information endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
async def list_datasets() -> dict[str, Any]:
    """List all available datasets with statistics."""
    datasets = {}
    for key, ds_config in config.DATASETS.items():
        try:
            from app.data.loader import MoleculeNetDataLoader
            loader = MoleculeNetDataLoader(key, model_type="gcn")
            loader.load()
            stats = loader.get_stats()
            datasets[key] = {
                **ds_config,
                **stats,
            }
        except Exception as e:
            logger.warning("Could not load stats for %s: %s", key, e)
            datasets[key] = {
                **ds_config,
                "train_size": 0,
                "val_size": 0,
                "test_size": 0,
                "error": str(e),
            }

    return {"datasets": datasets, "count": len(datasets)}


@router.get("/{dataset_name}")
async def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """Get detailed information about a specific dataset."""
    if dataset_name not in config.DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    try:
        from app.data.loader import MoleculeNetDataLoader
        loader = MoleculeNetDataLoader(dataset_name, model_type="gcn")
        loader.load()
        stats = loader.get_stats()
    except Exception as e:
        logger.warning("Could not load stats: %s", e)
        stats = {"error": str(e)}

    return {
        **config.DATASETS[dataset_name],
        **stats,
    }
