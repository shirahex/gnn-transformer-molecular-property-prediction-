"""Attention visualization for GAT and ChemBERTa models."""

import logging
from typing import Any, Optional

import numpy as np
import torch
from rdkit import Chem

from app import config
from app.data.preprocessor import smiles_to_graph

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Extract and format attention weights for visualization."""

    def __init__(self, model_type: str, device: str = config.DEVICE) -> None:
        self.model_type = model_type
        self.device = device

    def extract_gat_attention(
        self,
        model: torch.nn.Module,
        smiles: str,
    ) -> dict[str, Any]:
        """Extract attention weights from GAT model.

        Args:
            model: GAT model instance.
            smiles: SMILES string.

        Returns:
            Dictionary with atom indices, scores, and edge attentions.
        """
        model.eval()
        data = smiles_to_graph(smiles)
        if data is None:
            return {"error": "Invalid SMILES"}

        data = data.to(self.device)

        with torch.no_grad():
            _ = model(data.x, data.edge_index, None, torch.zeros(data.x.size(0), dtype=torch.long, device=self.device))

        # Get attention weights from model
        attention_weights = model.attention_weights if hasattr(model, "attention_weights") else []

        if not attention_weights:
            return {"error": "No attention weights captured"}

        # Average attention across layers and heads
        atom_scores = {}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Could not parse SMILES for atom mapping"}

        num_atoms = mol.GetNumAtoms()

        # Use the last layer's attention (most semantic)
        last_attn = attention_weights[-1] if attention_weights else None
        if last_attn is not None:
            # Aggregate attention per atom (sum of incoming attentions)
            attn_np = last_attn.cpu().numpy().flatten()
            edge_index = data.edge_index.cpu().numpy()

            for i in range(num_atoms):
                # Find edges where i is the target
                mask = edge_index[1] == i
                if mask.any():
                    atom_scores[i] = float(attn_np[mask].mean())
                else:
                    atom_scores[i] = 0.0

        # Normalize scores to [0, 1]
        if atom_scores:
            max_score = max(atom_scores.values())
            min_score = min(atom_scores.values())
            if max_score > min_score:
                for k in atom_scores:
                    atom_scores[k] = (atom_scores[k] - min_score) / (max_score - min_score)

        return {
            "atom_scores": atom_scores,
            "num_atoms": num_atoms,
            "num_layers": len(attention_weights),
        }

    def extract_chemberta_attention(
        self,
        model: torch.nn.Module,
        smiles: str,
    ) -> dict[str, Any]:
        """Extract attention from ChemBERTa model.

        Args:
            model: ChemBERTa model instance.
            smiles: SMILES string.

        Returns:
            Dictionary with token attentions.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.CHEMBERTA_MODEL_NAME)
        encoded = tokenizer(smiles, return_tensors="pt").to(self.device)

        model.eval()
        with torch.no_grad():
            outputs = model.encoder(
                **encoded,
                output_attentions=True,
            )

        # Get last layer's attention (layer, batch, head, seq, seq)
        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            return {"error": "No attentions returned"}

        last_layer_attn = attentions[-1]  # (batch, heads, seq, seq)
        # Average over heads and batch, take CLS attention to all tokens
        avg_attn = last_layer_attn.mean(dim=1)  # (batch, seq, seq)
        cls_attn = avg_attn[0, 0, :].cpu().numpy()  # CLS token's attention to all

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

        # Map token attention back to approximate atom positions
        token_scores = {}
        for i, (token, score) in enumerate(zip(tokens, cls_attn)):
            token_scores[i] = {
                "token": token,
                "score": float(score),
            }

        # Normalize
        scores = [v["score"] for v in token_scores.values()]
        min_s, max_s = min(scores), max(scores)
        if max_s > min_s:
            for k in token_scores:
                token_scores[k]["score"] = (token_scores[k]["score"] - min_s) / (max_s - min_s)

        return {
            "token_scores": token_scores,
            "tokens": tokens,
            "num_tokens": len(tokens),
        }

    def get_attention(
        self,
        model: torch.nn.Module,
        smiles: str,
    ) -> dict[str, Any]:
        """Route to the appropriate attention extractor.

        Args:
            model: Model instance.
            smiles: SMILES string.

        Returns:
            Attention visualization data.
        """
        if self.model_type == "gat":
            return self.extract_gat_attention(model, smiles)
        elif self.model_type == "chemberta":
            return self.extract_chemberta_attention(model, smiles)
        elif self.model_type == "gcn":
            # GCN doesn't have attention; use gradient-based importance
            return self._gradient_importance(model, smiles)
        else:
            return {"error": f"Unknown model type: {self.model_type}"}

    def _gradient_importance(
        self,
        model: torch.nn.Module,
        smiles: str,
    ) -> dict[str, Any]:
        """Compute node importance via gradient magnitude (for GCN)."""
        model.eval()
        data = smiles_to_graph(smiles)
        if data is None:
            return {"error": "Invalid SMILES"}

        data = data.to(self.device)
        data.x.requires_grad_(True)

        out = model(data.x, data.edge_index, None, torch.zeros(data.x.size(0), dtype=torch.long, device=self.device))
        out.backward()

        if data.x.grad is not None:
            grad_magnitude = data.x.grad.abs().mean(dim=1).cpu().numpy()
            # Normalize
            if grad_magnitude.max() > grad_magnitude.min():
                grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (
                    grad_magnitude.max() - grad_magnitude.min()
                )
            atom_scores = {i: float(v) for i, v in enumerate(grad_magnitude)}
        else:
            atom_scores = {}

        return {
            "atom_scores": atom_scores,
            "num_atoms": data.x.size(0),
            "method": "gradient",
        }
