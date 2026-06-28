"""Training endpoints for model training jobs."""

import logging
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.models.gat import GATModel
from app.models.gcn import GCNModel
from app.models.chemberta import ChemBERTaModel
from app.data.loader import MoleculeNetDataLoader
from app.training.trainer import Trainer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/train", tags=["training"])

# In-memory job store (use Redis in production)
_training_jobs: dict[str, dict] = {}


class TrainRequest(BaseModel):
    model_name: str = Field(..., description="Model: gcn, gat, chemberta")
    dataset_name: str = Field(..., description="Dataset: esol, bbbp, freesolv, clintox")
    epochs: int = Field(default=config.EPOCHS, ge=1, le=500)
    batch_size: int = Field(default=config.BATCH_SIZE, ge=1, le=512)
    learning_rate: float = Field(default=config.LEARNING_RATE, gt=0)


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


def _run_training(
    job_id: str,
    model_name: str,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Background training task."""
    try:
        _training_jobs[job_id]["status"] = "running"
        _training_jobs[job_id]["message"] = "Loading dataset..."

        # Load data
        loader = MoleculeNetDataLoader(dataset_name, model_name)
        loader.load()

        task_type = loader.task_type
        num_classes = 1 if task_type == "classification" else None

        # Create model
        if model_name == "gcn":
            model = GCNModel(task_type=task_type, num_classes=num_classes)
        elif model_name == "gat":
            model = GATModel(task_type=task_type, num_classes=num_classes)
        elif model_name == "chemberta":
            model = ChemBERTaModel(task_type=task_type, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(config.DEVICE)

        # Create trainer
        trainer = Trainer(
            model=model,
            task_type=task_type,
            epochs=epochs,
            learning_rate=learning_rate,
            model_name=model_name,
            dataset_name=dataset_name,
            job_id=job_id,
        )

        # Get data loaders
        train_loader = loader.get_train_loader(batch_size=batch_size)
        val_loader = loader.get_val_loader(batch_size=batch_size)

        _training_jobs[job_id]["message"] = f"Training {model_name} on {dataset_name}..."

        # Train
        result = trainer.train(train_loader, val_loader)

        # Reload the BEST checkpoint before final test evaluation — `model`
        # currently holds the LAST epoch's weights, not necessarily the best.
        import torch
        best_checkpoint_path = config.MODEL_DIR / dataset_name / f"{model_name}_best.pt"
        if best_checkpoint_path.exists():
            ckpt = torch.load(best_checkpoint_path, map_location=config.DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                "Reloaded best checkpoint (epoch %s) for test evaluation",
                ckpt.get("epoch"),
            )

        # Evaluate on test set using the BEST model
        test_loader = loader.get_test_loader(batch_size=batch_size)
        from app.training.evaluator import Evaluator
        evaluator = Evaluator(model, task_type=task_type)
        test_metrics = evaluator.evaluate(test_loader)

        # Save test metrics to the best checkpoint
        if best_checkpoint_path.exists():
            ckpt["test_metrics"] = test_metrics
            torch.save(ckpt, best_checkpoint_path)

        _training_jobs[job_id].update({
            "status": "completed",
            "message": f"Training completed. Test metrics: {test_metrics}",
            "result": {
                **result,
                "test_metrics": test_metrics,
            },
            "progress": 1.0,
        })

    except Exception as e:
        logger.error("Training job %s failed: %s", job_id, e, exc_info=True)
        _training_jobs[job_id].update({
            "status": "failed",
            "message": str(e),
        })


@router.post("/{model_name}/{dataset_name}")
async def start_training(
    model_name: str,
    dataset_name: str,
    background_tasks: BackgroundTasks,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: Optional[float] = None,
) -> TrainResponse:
    """Start a training job in the background."""
    if model_name not in config.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Available: {config.AVAILABLE_MODELS}",
        )
    if dataset_name not in config.DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{dataset_name}'. Available: {list(config.DATASETS.keys())}",
        )

    # Use a model-appropriate learning rate if the caller didn't explicitly
    # set one — ChemBERTa needs a much smaller fine-tuning LR than GCN/GAT.
    learning_rate = config.MODEL_LEARNING_RATES.get(model_name, config.LEARNING_RATE)

    job_id = str(uuid.uuid4())[:8]
    _training_jobs[job_id] = {
        "job_id": job_id,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "status": "queued",
        "message": "Job queued",
        "progress": 0.0,
        "created_at": time.time(),
    }

    background_tasks.add_task(
        _run_training,
        job_id,
        model_name,
        dataset_name,
        epochs,
        batch_size,
        learning_rate,
    )

    return TrainResponse(
        job_id=job_id,
        status="queued",
        message=f"Training job {job_id} started: {model_name} on {dataset_name}",
    )


@router.get("/status/{job_id}")
async def get_training_status(job_id: str) -> dict[str, Any]:
    """Get the status of a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _training_jobs[job_id]


@router.get("/jobs")
async def list_training_jobs() -> dict[str, Any]:
    """List all training jobs."""
    return {"jobs": list(_training_jobs.values()), "count": len(_training_jobs)}


@router.get("/history/{model_name}/{dataset_name}")
async def get_training_history(model_name: str, dataset_name: str) -> dict[str, Any]:
    """Get training history if available."""
    job = next(
        (
            j for j in _training_jobs.values()
            if j.get("model_name") == model_name and j.get("dataset_name") == dataset_name
        ),
        None,
    )
    if job is None:
        raise HTTPException(status_code=404, detail="No training history found")
    return job