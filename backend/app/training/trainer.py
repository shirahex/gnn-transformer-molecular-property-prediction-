"""Training loop with early stopping and optional W&B logging."""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from app import config
from app.models.base import BaseMolecularModel
from app.training.evaluator import Evaluator
from app.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer with early stopping, LR scheduling, and checkpointing."""

    def __init__(
        self,
        model: BaseMolecularModel,
        task_type: str = "regression",
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        epochs: int = config.EPOCHS,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        device: str = config.DEVICE,
        model_name: str = "model",
        dataset_name: str = "dataset",
        job_id: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.task_type = task_type
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.job_id = job_id or f"{model_name}_{dataset_name}_{int(time.time())}"

        # Optimizer & scheduler
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR,
        )

        # Loss function
        if task_type == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()
        self.evaluator = Evaluator(model, task_type, device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": [],
            "learning_rate": [],
        }
        self.status = "idle"  # idle, running, completed, failed
        self.progress = 0.0
        self.status_message = ""

    def train_epoch(self, train_loader: PyGDataLoader | DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            if hasattr(batch, "x"):
                # GNN batch
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                if self.task_type == "classification":
                    # Binary classification: model outputs [batch, 1], target is [batch, 1] float
                    target = batch.y.view(-1, 1).float()
                else:
                    target = batch.y.view_as(out)
            else:
                # Transformer batch
                input_ids = batch["input_ids"].to(self.device)
                out = self.model(input_ids)
                
                if self.task_type == "classification":
                    target = batch["labels"].to(self.device).view(-1, 1).float()
                else:
                    target = batch["labels"].to(self.device).view(-1, 1)

            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: PyGDataLoader | DataLoader,
        val_loader: PyGDataLoader | DataLoader,
    ) -> dict[str, Any]:
        """Run full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dictionary with training history and final metrics.
        """
        self.status = "running"
        self.status_message = "Training started"

        try:
            for epoch in range(1, self.epochs + 1):
                self.current_epoch = epoch
                self.progress = epoch / self.epochs
                self.status_message = f"Epoch {epoch}/{self.epochs}"

                # Training
                train_loss = self.train_epoch(train_loader)
                self.history["train_loss"].append(train_loss)
                self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

                # Validation
                val_metrics = self.evaluator.evaluate(val_loader)
                val_loss = val_metrics.get("rmse", val_metrics.get("loss", train_loss))
                if isinstance(val_loss, float):
                    self.history["val_loss"].append(val_loss)

                main_metric_key = "r2" if self.task_type == "regression" else "roc_auc"
                main_metric = val_metrics.get(main_metric_key, 0.0)
                self.history["val_metric"].append(main_metric)

                # LR scheduling
                self.scheduler.step(val_loss)

                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_rmse=%.4f | val_%s=%.4f | lr=%.2e",
                    epoch,
                    self.epochs,
                    train_loss,
                    val_loss,
                    main_metric_key,
                    main_metric,
                    self.optimizer.param_groups[0]["lr"],
                )

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best", val_metrics)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.status_message = f"Early stopping at epoch {epoch}"
                        logger.info(self.status_message)
                        break

                # Save latest
                if epoch % 5 == 0:
                    self.save_checkpoint("latest", val_metrics)

            self.status = "completed"
            self.status_message = "Training completed"
            self.progress = 1.0

        except Exception as e:
            self.status = "failed"
            self.status_message = f"Training failed: {str(e)}"
            logger.error(self.status_message, exc_info=True)
            raise

        return {
            "job_id": self.job_id,
            "status": self.status,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "final_metrics": val_metrics,
            "test_metrics": test_metrics if 'test_metrics' in locals() else val_metrics,
            "epochs_trained": self.current_epoch,
        }

    def save_checkpoint(self, suffix: str = "latest", final_metrics: dict = None) -> str:
        """Save model checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        model_dir = config.MODEL_DIR / self.dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{self.model_name}_{suffix}.pt"

        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                "task_type": self.task_type,
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "final_metrics": final_metrics if final_metrics else {},  # <-- ADD THIS
            },
            path,
        )
        return str(path)

    @classmethod
    def load_checkpoint(cls, path: str, model: BaseMolecularModel) -> "Trainer":
        """Load a trainer from checkpoint.

        Args:
            path: Path to checkpoint file.
            model: Model instance to load weights into.

        Returns:
            Trainer instance with restored state.
        """
        checkpoint = torch.load(path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(
            model=model,
            task_type=checkpoint.get("task_type", "regression"),
            model_name=checkpoint.get("model_name", "model"),
            dataset_name=checkpoint.get("dataset_name", "dataset"),
        )
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint.get("epoch", 0)
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        trainer.history = checkpoint.get("history", trainer.history)

        return trainer
