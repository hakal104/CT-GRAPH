import random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast

import torch_geometric
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from report_gen.graph.graph_utils import get_organs, get_h_dict
  
class BaselineGAT(torch.nn.Module):
    """
    Baseline Graph Attention Network (GAT) used for the random baseline.

    Applies layer normalization, a projection MLP, a single multi-head GAT layer,
    and a residual projection.

    Args:
        in_features (int): Number of input node features.
        hidden_dim_proj (int): Hidden size of MLP.
        att_dim (int): Hidden size per attention head.
        out_dim_proj (int): Output size of the projection MLP.
        num_heads (int): Number of attention heads.

    Returns:
        Tensor: Node embeddings of shape (N, hidden_dim * num_heads) from `forward`.
    """
    def __init__(self, in_features, hidden_dim_proj, att_dim, out_dim_proj, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.init_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim_proj),
            nn.ReLU(),
            nn.Linear(hidden_dim_proj, out_dim_proj)
        )
        self.res_proj = nn.Linear(out_dim_proj, att_dim * num_heads)
        self.gat_1 = GATConv(
            in_channels=out_dim_proj,
            out_channels=att_dim,
            heads=num_heads,
            concat=True,         # output dim: hidden_dim * num_heads
            add_self_loops=True,
            dropout=0.2
        )

    def forward(self, x, edge_index):
        """
        Forward pass through the BaselineGAT.

        Args:
            x (Tensor): Node feature matrix of shape (N, 1488).
            edge_index (LongTensor): Graph edges of shape (2, E).

        Returns:
            Tensor: Node embeddings of shape (N, hidden_dim * num_heads).
        """
        print(x.shape)
        with autocast():
            x = self.norm(x)
            x = self.init_proj(x)
            x_residual = self.res_proj(x)
            x = F.elu(self.gat_1(x, edge_index))
            x = x + x_residual
        return x
    

