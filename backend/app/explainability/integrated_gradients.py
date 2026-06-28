"""Integrated Gradients for node feature importance attribution."""

import logging
from typing import Optional

import numpy as np
import torch

from app import config
from app.data.preprocessor import smiles_to_graph

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """Integrated Gradients attribution for molecular graphs.

    Attributes the prediction to input node features by computing
    the path integral of gradients along the straight-line path
    from a baseline (zero features) to the input.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = config.DEVICE,
        n_steps: int = 50,
    ) -> None:
        self.model = model
        self.device = device
        self.n_steps = n_steps

    def explain(
        self,
        smiles: str,
        target_idx: Optional[int] = None,
    ) -> dict:
        """Compute Integrated Gradients for a molecule.

        Args:
            smiles: SMILES string.
            target_idx: Target output index for attribution.

        Returns:
            Dictionary with attribution scores per atom.
        """
        data = smiles_to_graph(smiles)
        if data is None:
            return {"error": "Invalid SMILES"}

        data = data.to(self.device)
        original_features = data.x.clone().detach()

        # Baseline: zero features
        baseline = torch.zeros_like(original_features)

        # Compute interpolated features and gradients
        attributions = torch.zeros_like(original_features)

        self.model.eval()
        for i in range(self.n_steps):
            alpha = (i + 1) / self.n_steps
            interpolated = baseline + alpha * (original_features - baseline)
            interpolated.requires_grad_(True)

            out = self.model(
                interpolated,
                data.edge_index,
                data.edge_attr,
                torch.zeros(data.x.size(0), dtype=torch.long, device=self.device),
            )

            if target_idx is not None:
                target = out[0, target_idx]
            else:
                target = out[0]

            self.model.zero_grad()
            target.backward()

            if interpolated.grad is not None:
                attributions += interpolated.grad.detach()

        # Scale by input difference
        integrated_gradients = (original_features - baseline) * (attributions / self.n_steps)

        # Sum across feature dimension for per-atom importance
        atom_importance = integrated_gradients.abs().sum(dim=1).cpu().numpy()

        # Normalize
        if atom_importance.max() > atom_importance.min():
            atom_importance = (atom_importance - atom_importance.min()) / (
                atom_importance.max() - atom_importance.min()
            )

        atom_scores = {i: float(v) for i, v in enumerate(atom_importance)}

        return {
            "atom_scores": atom_scores,
            "num_atoms": data.x.size(0),
            "smiles": smiles,
            "method": "integrated_gradients",
        }
