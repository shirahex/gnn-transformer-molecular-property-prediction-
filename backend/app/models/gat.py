"""Graph Attention Network (GAT) with multi-head attention for molecules."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from app import config
from app.models.base import BaseMolecularModel


class GATModel(BaseMolecularModel):
    """GAT with multi-head attention and readout pooling.

    Architecture:
        - 3 GAT layers with multi-head attention
        - Batch normalization after each layer
        - Dropout for regularization
        - Global mean pooling (readout)
        - 2-layer MLP predictor head

    Attention weights are stored for explainability visualization.
    """

    def __init__(
        self,
        node_feature_dim: int = 30,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = config.DROPOUT,
        task_type: str = "regression",
        num_tasks: int = 1,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__(task_type=task_type, num_tasks=num_tasks, num_classes=num_classes)

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.attention_weights: list[torch.Tensor] = []

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: input_dim -> hidden_dim * num_heads
        self.convs.append(
            GATConv(
                node_feature_dim,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Last layer: average heads instead of concatenating
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Predictor head
        if task_type == "classification" and num_classes:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_tasks),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with attention weight capture.

        Args:
            x: Node features (num_nodes, node_feature_dim).
            edge_index: Edge connectivity (2, num_edges).
            edge_attr: Optional edge attributes.
            batch: Batch vector.

        Returns:
            Predictions with attention weights stored in self.attention_weights.
        """
        self.attention_weights = []
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GAT layers with batch norm
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x, attn = conv(x, edge_index, return_attention_weights=True)
            self.attention_weights.append(attn[1] if isinstance(attn, tuple) else attn)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Predictor
        out = self.predictor(x)

        return out
