import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from torch_geometric.data import Data, Batch
from report_gen.graph.anatomy import get_organs, get_h_dict

def get_random_indices(n_nodes: int = 34,
                     n_edges: int = 50,
                     seed: int = 0) -> torch.LongTensor:
    """
    Create `n_edges` random directed edges among `n_nodes` nodes.
    
    Args:
    - n_nodes (int): Number of input nodes for the graph.
    - n_edges (int): Number of edges between the nodes.
    - seed (int): Random seed.

    Returns:
    - PyG Batch object with stacked nodes & random edge indices.
    """
    
    random.seed(seed)
    edges = set()
    while len(edges) < n_edges:
        src = random.randrange(n_nodes)
        dst = random.randrange(n_nodes)
        if src != dst:
            edges.add((src, dst))        # keep only unique pairs

    # convert to tensor [2, E]
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index

def get_random_batch(all_features):
    """
    Creates a batched PyG `Batch` object for GAT processing.
    
    Args:
    - all_features (Tensor): All node features.

    Returns:
    - PyG Batch object with stacked nodes & random edge indices.
    """
    
    data_list = []
    edge_index = get_random_indices()
    print(edge_index)

    for i in range(len(all_features)):
        
        # Create PyG Data object
        data = Data(
            x=all_features[i],  # Node features
            edge_index=edge_index
        )

        # Append to batch list
        data_list.append(data)

    # Create the batch from multiple graphs
    batch = Batch.from_data_list(data_list)

    return batch

def get_simple_indices():
    """
    Creates edge indices for fine-to-coarse and coarse-to-global aggregration based on anatomy.
    """
    
    organs = get_organs()
    hierarchical_region_dict = get_h_dict()
    
    remove_indices = torch.tensor([26,41,35,25])
    x = [elem for i, elem in enumerate(organs) if i not in remove_indices]
    coarse_nodes = x[1:9]
    fine_nodes = x[9:]
    
    global_idx = 0
    coarse_nodes_to_idx = {name: idx for idx, name in enumerate(coarse_nodes)}
    fine_nodes_to_idx = {name: idx for idx, name in enumerate(fine_nodes)}
    coarse_regions = ['abdomen', 'bone', 'esophagus', 'heart', 'mediastinum', 'trachea', 'lung', 'thyroid']

    # Fine --> coarse edges
    fine_coarse_edges = []
    for coarse_region, children in hierarchical_region_dict.items():
        coarse_idx = coarse_nodes_to_idx[coarse_region]
        for child in children.keys():
            child_idx = fine_nodes_to_idx[child]
            fine_coarse_edges.append([child_idx, coarse_idx])

    fine_coarse_edge_index = torch.tensor(fine_coarse_edges, dtype=torch.long).t().contiguous()
    
    # Coarse --> global edges
    coarse_global_edges = []
    for region in coarse_regions:
        region_idx = coarse_nodes_to_idx[region]
        coarse_global_edges.append([region_idx, global_idx])

    coarse_global_edge_index = torch.tensor(coarse_global_edges, dtype=torch.long).t().contiguous() 
    
    return fine_coarse_edge_index, coarse_global_edge_index
    
def create_data(fine_features,coarse_features,global_feature,fine_coarse_edge_index,coarse_global_edge_index):
    """
    Returns Data object using node features and edge indices.
    """
    
    x_fine = torch.tensor(fine_features) # 30 fine-grained nodes
    x_coarse = torch.tensor(coarse_features)  # 8 coarse + global nodes
    x_global = global_feature.unsqueeze(0)

    return Data(x_fine=x_fine, 
                x_coarse=x_coarse,
                x_global = x_global,
                edge_index_fine_coarse=fine_coarse_edge_index,
                edge_index_coarse_global=coarse_global_edge_index)

def get_hierarchical_batch(inputs, global_inputs):
    """
    Creates a batched PyG `Batch` object for hierarchical GAT processing.
    
    Args:
    - inputs (Tensor): All local node features.
    - global inputs (Tensor): Global node features. 

    Returns:
    - PyG Batch object with stacked nodes & edge indices.
    """
    
    batch_size = inputs.shape[0]
    coarse_features, fine_features = inputs[:,:8,:], inputs[:,8:,:]
    coarse_fine_edge_index, global_coarse_edge_index = get_simple_indices()
    
    data_list = [create_data(fine_features[i],coarse_features[i],global_inputs[i],coarse_fine_edge_index,global_coarse_edge_index) for i in range(batch_size)] 

    num_fine_nodes = data_list[0].x_fine.shape[0]  
    num_coarse_nodes = data_list[0].x_coarse.shape[0]  

    all_fine_features = torch.cat([data.x_fine for data in data_list], dim=0)
    all_coarse_features = torch.cat([data.x_coarse for data in data_list], dim=0)
    global_features = torch.cat([data.x_global for data in data_list], dim=0)

    # Adjust edge indices for fine-to-coarse edges and coarse-to-global edges
    edge_index_fine_coarse_batch = []
    edge_index_coarse_global_batch = []
    
    for i, data in enumerate(data_list):
        # Shift node indices by the number of nodes in previous graphs for each level (fine, coarse, global) individually
        
        shift_fine = i * num_fine_nodes
        shift_coarse = i * num_coarse_nodes
        shift_global = i*1
        
        edge_index_fine_coarse_batch.append(data.edge_index_fine_coarse + torch.tensor([[shift_fine], [shift_coarse]]).repeat(1, data.edge_index_fine_coarse.shape[1]))
        edge_index_coarse_global_batch.append(data.edge_index_coarse_global + torch.tensor([[shift_coarse], [shift_global]]).repeat(1, data.edge_index_coarse_global.shape[1]))

    # Stack all edge indices and features
    edge_index_fine_coarse_batch = torch.cat(edge_index_fine_coarse_batch, dim=1)
    edge_index_coarse_global_batch = torch.cat(edge_index_coarse_global_batch, dim=1)

    batch = Batch()
    batch.x = torch.cat([all_fine_features, all_coarse_features], dim=0)
    batch.x_fine = all_fine_features
    batch.x_coarse = all_coarse_features
    batch.x_global = global_features
    batch.edge_index_fine_coarse = edge_index_fine_coarse_batch
    batch.edge_index_coarse_global = edge_index_coarse_global_batch
    
    return batch


def get_single_level_indices():
    """
    Returns an edge index that connects all local (fine-level and coarse-level) nodes to the global node.
    """
    
    fine_ids   = torch.arange(0,33)                  
    global_id  = torch.full((33,), 0)         
    src = fine_ids
    dst = global_id

    edge_index = torch.stack([src, dst], 0)

    return edge_index 

def create_data_single_level(fine_features, global_feature,edge_index_fine_global):
    """
    Returns Data object using node features and "single-level" edge indices.
    """
    
    x_fine = torch.tensor(fine_features) # 30 fine-grained nodes
    x_global = global_feature.unsqueeze(0)
    
    return Data(x_fine=x_fine, 
                x_global = x_global,
                edge_index_fine_global=edge_index_fine_global)

def get_single_level_batch(inputs, global_inputs):
    """
    Creates a batched PyG `Batch` object for "single-level" GAT processing.
    
    Args:
    - inputs (Tensor): All local node features.
    - global inputs (Tensor): Global node features. 

    Returns:
    - PyG Batch object with stacked nodes & edge indices.
    """
    
    batch_size = inputs.shape[0]
    fine_global_edge_index = get_single_level_indices()
    
    data_list = [create_data_single_level(inputs[i],global_inputs[i],fine_global_edge_index) for i in range(batch_size)] 

    num_fine_nodes = data_list[0].x_fine.shape[0] 
    
    all_fine_features = torch.cat([data.x_fine for data in data_list], dim=0)
    global_features = torch.cat([data.x_global for data in data_list], dim=0)

    edge_index_fine_global_batch = []
    
    for i, data in enumerate(data_list):
        shift_fine = i * num_fine_nodes
        shift_global = i*1
        
        edge_index_fine_global_batch.append(data.edge_index_fine_global + torch.tensor([[shift_fine], [shift_global]]).repeat(1, data.edge_index_fine_global.shape[1]))

    edge_index_fine_global_batch = torch.cat(edge_index_fine_global_batch, dim=1)

    batch = Batch()
    batch.x = torch.cat([all_fine_features], dim=0)
    batch.x_fine = all_fine_features
    batch.x_global = global_features
    batch.edge_index_fine_global = edge_index_fine_global_batch
    
    return batch
