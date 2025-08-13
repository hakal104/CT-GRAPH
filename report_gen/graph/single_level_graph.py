import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
from torch.cuda.amp import autocast

class SingleLevelGraph(torch.nn.Module):
    """
    Implementation of the graph used in the ablation "Single-Level Hierarchy Graph"
    """
    
    def __init__(self, in_features, hidden_dim_proj, att_dim, num_heads, llm_dim):
        super().__init__()
        # Fine → Coarse GAT (bipartite)
        self.norm_fine = nn.LayerNorm(in_features)
        self.norm_global = nn.LayerNorm(att_dim*num_heads)
        
        self.init_proj = nn.Sequential(nn.Linear(in_features, hidden_dim_proj), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim_proj, att_dim*num_heads))
        
        self.gat_fine_to_global = GATConv(
            in_channels=att_dim*num_heads, 
            out_channels=att_dim, 
            heads=num_heads,
            concat=True,
            add_self_loops=True,
            dropout=0.2
        )
        
        self.global_proj = torch.nn.Linear(att_dim*num_heads, llm_dim)
        self.fine_proj = torch.nn.Linear(att_dim*num_heads, llm_dim)

    def forward(self, x_fine, x_global, edge_index_fine_global):
        """
        x_fine: Features of fine-level nodes.
        x_global: Features of global nodes.
        edge_index_fine_global: Edges from fine nodes to the global node.
        """
        
        with autocast():  
            B = x_global.shape[0]
            
            # Normalization of nodes
            x_fine = self.norm_fine(x_fine)
            x_fine = self.init_proj(x_fine)  
            x_global = self.norm_global(x_global)

            
            # Fine --> Global aggregation
            x_global = self.gat_fine_to_global(
                x=(x_fine, x_global),  # (src, dst)
                edge_index=edge_index_fine_global
            ) + x_global
            
            fine_dim, global_dim = x_fine.shape[1],x_global.shape[1]
 
            x_global = x_global.view(B,-1,global_dim)
            x_fine = x_fine.view(B,-1,fine_dim)
            
            #Projection to LLM dimension
            x_global = self.global_proj(x_global)
            x_fine = self.fine_proj(x_fine)
            
            out = torch.cat([x_global,x_fine], dim=1)
        
        return out 