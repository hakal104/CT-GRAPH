from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

from torch_geometric.data import Data, Batch
import torch_geometric
import torch.nn.functional as F
from einops import rearrange

from feature_extraction.models import get_models
from report_gen.graph.baseline_graph import BaselineGAT
from report_gen.graph.hierarchical_graph import HierarchicalGAT
from report_gen.graph.single_level_graph import SingleLevelGraph
from report_gen.graph.graph_utils import get_random_batch, get_hierarchical_batch, get_single_level_batch

#The following variants/ablations reflect the main ablations from the official paper.
# Set the desired variant by updating the "mode" field in the config file 
# (must match one of the strings below).

VARIANT_REGISTRY = {}
variants = ["use_global",
            "use_global_multiple",
            "use_local",
            "use_global_local",
            "use_random_graph",
            "use_single_level_graph",
            "use_ctgraph"]

for variant in variants:
    VARIANT_REGISTRY[variant] = None


class BaseModelVariant(nn.Module, ABC):
    """
    Abstract base class for model variants implementing the main method and its ablations.

    Provides a common interface and shared attributes for all variants, including:
    - Default node and filtering parameters.
    - Abstract image encoding method to be implemented by subclasses.
    - Method to get saveable modules (non-peft).
    """
    
    def __init__(self, config):
        """
        Initialize the base model variant with common settings.
        The number of nodes "num_nodes" represent the number of features that 
        are fed to the LLM. "use_filter" is needed for preprocessing of the nodes 
        to remain the correct anatomical ordering.

        Args:
            config: Main model config.
        """
        
        super().__init__()
        self.config = config
        self.num_nodes = 1
        self.use_filter = False

    @abstractmethod
    def encode_features(self, local_features, global_features):
        """
        Fuse and project local and global features.

        Args:
            local_features (torch.Tensor): Features of shape [B, N, D_local].
            global_features (torch.Tensor): Features of shape [B, D_global].

        Returns:
            Projected embeddings for report generation.
        """
        pass

    def get_saveable_modules(self):
        """
        Returns dictionary of modules that are fully trained and (should be) saved.
        """
        
        return {}

class GlobalVariant(BaseModelVariant):
    """
    This variant represents the ablation named "global".
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.projector = nn.Sequential(
                                         nn.Linear(32*self.config.feat_dims[-1], config.global_proj_dim), 
                                         nn.ReLU(), 
                                         nn.Linear(config.global_proj_dim, config.llama_dim))

        self.num_nodes = 1
        self.use_filter = False


    def eencode_features(self, local_features, global_features):

        B = global_features.shape[0]
        output_size = (4,4,2)
        x= nn.AdaptiveAvgPool3d(output_size)(global_features) 
        x = x.view(B,-1)
        out = self.projector(x)
        out = out.unsqueeze(1)

        return out

    def get_modules(self):
        return {"projector": self.projector}
    
class GlobalMultipleVariant(BaseModelVariant):
    """
    This variant represents the ablation named "global (multiple)".
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.projector = nn.Sequential(nn.Linear(self.config.feat_dims[-1], config.global_proj_dim), 
                                         nn.ReLU(), 
                                         nn.Linear(config.global_proj_dim, 4096))

        self.num_nodes = 33
        self.use_filter = False


    def encode_features(self, local_features, global_features):

        B = global_features.shape[0]
        output_size = (4,4,2)
        x= nn.AdaptiveAvgPool3d(output_size)(global_features) 
        x = x.reshape(B, 768, -1)        
        x = x.permute(0, 2, 1)
        out = self.projector(x)

        return out

    def get_modules(self):
        return {"projector": self.projector}
    
class LocalVariant(BaseModelVariant):
    """
    This variant represents the ablation named "local".
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        input_dim = np.sum(np.array(self.config.feat_dims))
        self.projector = nn.Sequential(nn.Linear(input_dim, 2048), 
                                 nn.ReLU(), 
                                 nn.Linear(2048,config.llama_dim))     

        self.num_nodes = 32
        self.use_filter = False


    def encode_features(self, local_features, global_features):

        out = self.projector(local_features)

        return out

    def get_modules(self):
        return {"projector": self.projector}
    
class GlobalLocalVariant(BaseModelVariant):
    """
    This variant represents the ablation named "global + local".
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        input_dim = np.sum(np.array(self.config.feat_dims))
        self.projector_global = nn.Sequential(nn.Linear(32*self.config.feat_dims[-1],config.global_proj_dim), 
                             nn.ReLU(), 
                             nn.Linear(config.global_proj_dim, config.llama_dim))

        self.projector = nn.Sequential(nn.Linear(input_dim, 2048), 
                             nn.ReLU(), 
                             nn.Linear(2048,config.llama_dim))
        self.num_nodes = 34
        self.use_filter = False


    def encode_features(self, local_features, global_features):

        B = global_features.shape[0]
        output_size = (4,4,2)

        x= nn.AdaptiveAvgPool3d(output_size)(global_features) 
        x = x.view(B,-1)
        out_global = self.projector_global(x).unsqueeze(1)
        local = self.projector(local_features)
        global_local = torch.concat([out_global,local], dim=1)
        out = global_local
        
        return out

    def get_modules(self):
        return {"projector": self.projector,
                "projector_global": self.projector_global}

class RandomGraphVariant(BaseModelVariant):
    """
    This variant represents the ablation named "random graph".
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        concat_dim = np.sum(np.array(self.config.feat_dims))
        self.projector_global =  nn.Sequential(nn.Linear(32*self.config.feat_dims[-1], config.global_proj_dim), 
                                         nn.ReLU(), 
                                         nn.Linear(config.global_proj_dim, concat_dim))
        
        self.graph_module = BaselineGAT(in_features=concat_dim, 
                                        hidden_dim_proj=config.hidden_dim_proj, 
                                        att_dim=config.att_dim, 
                                        out_dim_proj=config.out_dim_proj, 
                                        num_heads=config.num_heads, 
                                        llm_dim=config.llama_dim)
        
        self.llm_projector = nn.Sequential(nn.Linear(config.att_dim*num_heads,config.llama_dim))
        self.num_nodes = 34
        self.use_filter = False

    def encode_features(self, local_features, global_features):

        B = global_features.shape[0]
        output_size = (4,4,2)
        x= nn.AdaptiveAvgPool3d(output_size)(global_features)
        x = x.view(B,-1)
        out_global = self.projector_global(x).unsqueeze(1)
        global_local = torch.concat([out_global,local_features.squeeze(1)], dim=1)
        
        gnn_batch = get_random_batch(global_local)
        gnn_feats = self.graph_module(gnn_batch.x, gnn_batch.edge_index.to('cuda'))
        gnn_feats = gnn_feats.view(B,-1,gnn_feats.shape[1])
        out = self.llm_projector(gnn_feats)  
        
        return out

    def get_modules(self):
        return {"projector_global": self.projector_global,
                "graph_module": self.graph_module,
                "llm_projector": self.llm_projector}        

class SingleLevelGraphVariant(BaseModelVariant):
    """
    This variant represents the ablation named "single-level hierarchy graph".
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        concat_dim = np.sum(np.array(self.config.feat_dims))
        self.projector_global =  nn.Sequential(nn.Linear(32*self.config.feat_dims[-1], config.hidden_dim_proj), 
                             nn.ReLU(), 
                             nn.Linear(config.hidden_dim_proj, 2048))
        
        self.graph_module = SingleLevelGraph(in_features=concat_dim, 
                                             hidden_dim_proj=config.hidden_dim_proj, 
                                             att_dim=config.att_dim, 
                                             num_heads=config.num_heads, 
                                             llm_dim=config.llama_dim)
        
        self.num_nodes = 34
        self.use_filter = False


    def encode_features(self, local_features, global_features):

        B = global_features.shape[0]
        output_size = (4,4,2)
        x= nn.AdaptiveAvgPool3d(output_size)(global_features)
        x = x.view(B,-1)
        out_global = self.projector_global(x)
        gnn_batch = get_single_level_batch(local_features,out_global)
        gnn_feats = self.graph_module(gnn_batch.x_fine, gnn_batch.x_global, gnn_batch.edge_index_fine_global.to('cuda'))
        out = gnn_feats
        
        return out

    def get_modules(self):
        return {"projector_global": self.projector_global,
                "graph_module": self.graph_module}        

class CTGraph(BaseModelVariant):
    """
    This variant represents the main method named "CT-GRAPH".
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        concat_dim=np.sum(np.array(config.feat_dims))
        self.projector_global = nn.Sequential(
                                     nn.Linear(32*config.feat_dims[-1], config.global_proj_dim), 
                                     nn.ReLU(), 
                                     nn.Linear(config.global_proj_dim, config.num_heads//2*config.out_dim_proj))

        self.graph_module = HierarchicalGAT(in_features=concat_dim, 
                                                  hidden_dim_proj=config.hidden_dim_proj, 
                                                  att_dim=config.att_dim,
                                                  out_dim_proj=config.out_dim_proj, 
                                                  num_heads=config.num_heads,
                                                  llm_dim=config.llama_dim)
        self.num_nodes = 38
        self.use_filter = True


    def encode_features(self, local_features, global_features):

        B,C,H,W,D = global_features.shape
        output_size = (4,4,2)
        x= nn.AdaptiveAvgPool3d(output_size)(global_features)
        x = x.view(B,-1)
        out_global = self.projector_global(x)
        batch =  get_hierarchical_batch(local_features,out_global)
        gnn_feats = self.graph_module(batch.x_fine, batch.x_coarse, batch.x_global, batch.edge_index_fine_coarse.to('cuda'), batch.edge_index_coarse_global.to('cuda'))
        out = gnn_feats

        return out

    def get_modules(self):
        return {"projector_global": self.projector_global,
                "graph_module": self.graph_module}    

VARIANT_REGISTRY["use_global"] = GlobalVariant
VARIANT_REGISTRY["use_global_multiple"] = GlobalMultipleVariant
VARIANT_REGISTRY["use_local"] = LocalVariant
VARIANT_REGISTRY["use_global_local"] = GlobalLocalVariant
VARIANT_REGISTRY["use_random_graph"] = RandomGraphVariant
VARIANT_REGISTRY["use_single_level_graph"] = SingleLevelGraphVariant
VARIANT_REGISTRY["use_ctgraph"] = CTGraph
    
