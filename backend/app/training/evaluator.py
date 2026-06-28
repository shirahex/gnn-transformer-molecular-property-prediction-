"""Evaluation utilities for trained models."""

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from app import config
from app.models.base import BaseMolecularModel
from app.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Model evaluator that handles both GNN and Transformer models."""

    def __init__(
        self,
        model: BaseMolecularModel,
        task_type: str = "regression",
        device: str = config.DEVICE,
    ) -> None:
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device

    @torch.no_grad()
    def evaluate(self, data_loader: PyGDataLoader | DataLoader) -> dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            data_loader: PyG DataLoader for GNNs or PyTorch DataLoader for transformers.

        Returns:
            Dictionary of computed metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            if hasattr(batch, "x"):
                # GNN batch
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                labels = batch.y
            else:
                # Transformer batch
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                out = self.model(input_ids)

            if self.task_type == "classification":
                preds = torch.sigmoid(out).cpu().numpy()
            else:
                preds = out.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        metrics = compute_metrics(all_labels, all_preds, self.task_type)
        return metrics

    @torch.no_grad()
    def predict_single(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Predict on a single sample with uncertainty estimation.

        Returns:
            Dictionary with 'prediction', 'uncertainty', 'ci_lower', 'ci_upper'.
        """
        self.model.eval()

        if hasattr(batch, "x"):
            batch = batch.to(self.device)
            # Single forward pass for speed
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            x = x.to(self.device)
            out = self.model(x)

        if self.task_type == "classification":
            pred = float(torch.sigmoid(out).item())
        else:
            pred = float(out.item())

        # Uncertainty via MC dropout
        uncertainty_result = self.model.predict_with_uncertainty(
            x, edge_index, edge_attr, batch, num_samples=config.MC_DROPOUT_PASSES
        )

        std_val = float(uncertainty_result["std"].item())
        ci_lower = float(uncertainty_result["ci_lower"].item())
        ci_upper = float(uncertainty_result["ci_upper"].item())

        if self.task_type == "classification":
            ci_lower = max(0.0, min(1.0, ci_lower))
            ci_upper = max(0.0, min(1.0, ci_upper))

        return {
            "prediction": pred,
            "uncertainty": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
