"""Base model class for all molecular property predictors."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from app import config


class BaseMolecularModel(nn.Module, ABC):
    """Abstract base class for all molecular prediction models.

    Provides a unified interface for GNN and Transformer-based models,
    including forward passes with optional MC dropout for uncertainty.
    """

    def __init__(
        self,
        task_type: str = "regression",
        num_tasks: int = 1,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self._device = torch.device(config.DEVICE)

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass — must be implemented by subclasses."""
        ...

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        num_samples: int = config.MC_DROPOUT_PASSES,
    ) -> dict[str, torch.Tensor]:
        """Monte Carlo Dropout for uncertainty quantification.

        Performs multiple forward passes with dropout enabled to estimate
        prediction mean and confidence intervals.

        Args:
            x: Node features or tokenized input.
            edge_index: Graph edge indices (GNN models).
            edge_attr: Edge attributes (GNN models).
            batch: Batch vector (GNN models).
            num_samples: Number of MC dropout forward passes.

        Returns:
            Dictionary with 'mean', 'std', 'ci_lower', 'ci_upper' tensors.
        """
        self.eval()  # keep BatchNorm (and everything else) in eval mode
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # only Dropout layers become stochastic

        predictions: list[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(num_samples):
                out = self.forward(x, edge_index, edge_attr, batch)
                if self.task_type == "classification" and self.num_classes:
                    out = torch.softmax(out, dim=-1)
                predictions.append(out)

        self.eval()  # fully restore eval mode for any subsequent calls

        preds = torch.stack(predictions)  # (num_samples, batch, ...)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)

        # Confidence interval (normal approximation)
        z = 1.96  # 95% confidence
        ci_lower = mean - z * std
        ci_upper = mean + z * std

        return {
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "task_type": self.task_type,
                "num_tasks": self.num_tasks,
                "num_classes": self.num_classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "BaseMolecularModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=config.DEVICE)
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        return model