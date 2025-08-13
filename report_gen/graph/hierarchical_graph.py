import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
from torch.cuda.amp import autocast

class HierarchicalGAT(torch.nn.Module):
    """
    Implementation of the HierarchicalGAT used in the main CT-GRAPH method.
    """
    
    def __init__(self, in_features, hidden_dim_proj, att_dim, out_dim_proj, num_heads, llm_dim):
        super().__init__()
        # Fine → Coarse GAT (bipartite)
        self.norm_fine = nn.LayerNorm(in_features)
        self.norm_coarse = nn.LayerNorm(att_dim*num_heads)
        self.norm_global = nn.LayerNorm(att_dim*num_heads)
        
        self.init_proj = nn.Sequential(nn.Linear(in_features, hidden_dim_proj), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim_proj, out_dim_proj))

        self.gat_fine_to_coarse = GATConv(
            in_channels=out_dim_proj, 
            out_channels=att_dim, 
            heads=num_heads,
            concat=True,
            add_self_loops=True,
            dropout=0.2
        )
        
        # Coarse → Global GAT (bipartite)
        self.gat_coarse_to_global = GATConv(
            in_channels=att_dim*num_heads,
            out_channels=out_dim_proj,
            heads=num_heads//2,
            concat=True,
            add_self_loops=True,
            dropout=0.2
        )
        
        self.global_proj = torch.nn.Linear(num_heads//2*out_dim_proj, llm_dim)
        self.coarse_proj = torch.nn.Linear(att_dim*num_heads, llm_dim)
        self.fine_proj = nn.Sequential(nn.Linear(out_dim_proj, 2048), 
                                 nn.ReLU(), 
                                 nn.Linear(2048, llm_dim))

    def forward(self, x_fine, x_coarse, x_global, edge_index_fine_coarse, edge_index_coarse_global):
        """
        x_fine: Features of fine-grained nodes.
        x_coarse: Features of coarse nodes.
        x_global: Features of global nodes.
        edge_index_fine_coarse: Edges from fine nodes to coarse nodes.
        edge_index_coarse_global: Edges from coarse nodes to the global node.
        """
        
        with autocast():  
            B = x_global.shape[0]
            
            # Normalization and initial projection
            x_fine = self.norm_fine(x_fine)
            x_coarse = self.norm_fine(x_coarse)
            x_fine = self.init_proj(x_fine)
            x_coarse = self.init_proj(x_coarse)
            x_global = self.norm_global(x_global)

            # Fine --> Coarse aggregation
            x_coarse = self.gat_fine_to_coarse(
                x=(x_fine, x_coarse),  # (src, dst)
                edge_index=edge_index_fine_coarse
            )

            # Coarse --> Global aggregation
            x_coarse = F.relu(self.norm_coarse(x_coarse))
            x_global = self.gat_coarse_to_global(
                x=(x_coarse, x_global), 
                edge_index=edge_index_coarse_global
            ) + x_global
            
            # Projection to LLM space
            fine_dim, coarse_dim, global_dim = x_fine.shape[1],x_coarse.shape[1], x_global.shape[1]
 
            x_global = x_global.view(B,-1,global_dim)
            x_coarse=x_coarse.view(B,-1,coarse_dim)
            x_fine = x_fine.view(B,-1,fine_dim)
            
            x_global = self.global_proj(x_global)
            x_coarse = self.coarse_proj(x_coarse)
            x_fine = self.fine_proj(x_fine)
            
            out = torch.cat([x_global,x_coarse,x_fine], dim=1)
        
        return out 
