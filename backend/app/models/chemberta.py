"""ChemBERTa-2 Transformer model for molecular property prediction."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from app import config
from app.models.base import BaseMolecularModel


class ChemBERTaModel(BaseMolecularModel):
    """ChemBERTa-2 wrapper with LoRA-style parameter-efficient fine-tuning.

    Uses a pre-trained RoBERTa model trained on 10M SMILES strings from ZINC.
    The final hidden state of the [CLS] token is fed into a prediction MLP.
    """

    def __init__(
        self,
        model_name: str = config.CHEMBERTA_MODEL_NAME,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        task_type: str = "regression",
        num_tasks: int = 1,
        num_classes: Optional[int] = None,
    ) -> None:
        # tokenizer loads before super().__init__() to avoid nn.Module issues
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        super().__init__(task_type=task_type, num_tasks=num_tasks, num_classes=num_classes)

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Pre-trained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        # Freeze bottom layers for parameter-efficient tuning
        self._freeze_base_layers()

        # Prediction head
        if task_type == "classification" and num_classes:
            self.predictor = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_tasks),
            )

        self.dropout = nn.Dropout(dropout)

    def _freeze_base_layers(self, num_unfrozen: int = 4) -> None:
        """Freeze all but the last N transformer layers for efficient fine-tuning."""
        # Freeze all parameters first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last num_unfrozen layers
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
            for layer in layers[-num_unfrozen:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Always keep pooler trainable if it exists
        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for SMILES token sequences.

        Args:
            x: Token IDs of shape (batch_size, seq_len).
            edge_index: Unused (for API compatibility).
            edge_attr: Unused.
            batch: Unused.

        Returns:
            Predictions of shape (batch_size, num_tasks) or (batch_size, num_classes).
        """
        # x is expected to be token_ids here
        if x.dtype != torch.long:
            x = x.long()

        attention_mask = (x != self.tokenizer.pad_token_id).long()

        outputs = self.encoder(input_ids=x, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        # Predictor head
        out = self.predictor(cls_embedding)

        return out

    def encode_smiles(self, smiles_list: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=config.MAX_SMILES_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(config.DEVICE),
            "attention_mask": encoded["attention_mask"].to(config.DEVICE),
        }
