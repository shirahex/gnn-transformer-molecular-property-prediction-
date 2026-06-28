"""Graph Convolutional Network (GCN) for molecular property prediction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from app import config
from app.models.base import BaseMolecularModel


class GCNModel(BaseMolecularModel):
    """GCN baseline with batch normalization and dropout.

    Architecture:
        - 3 GCN layers with ReLU activation
        - Batch normalization after each layer
        - Dropout for regularization
        - Global mean pooling for graph-level representation
        - 2-layer MLP predictor head
    """

    def __init__(
        self,
        node_feature_dim: int = 30,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = 3,
        dropout: float = config.DROPOUT,
        task_type: str = "regression",
        num_tasks: int = 1,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__(task_type=task_type, num_tasks=num_tasks, num_classes=num_classes)

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
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
        """Forward pass.

        Args:
            x: Node features of shape (num_nodes, node_feature_dim).
            edge_index: Edge connectivity (2, num_edges).
            edge_attr: Optional edge attributes.
            batch: Batch vector assigning nodes to graphs.

        Returns:
            Predictions of shape (num_graphs, num_tasks) or (num_graphs, num_classes).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GCN layers with batch norm and dropout
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling: mean over nodes in each graph
        x = global_mean_pool(x, batch)

        # Predictor head
        out = self.predictor(x)

        return out
