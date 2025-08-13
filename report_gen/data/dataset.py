from torch.utils.data import Dataset
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils
from torch.optim import Adam

import os
from smart_open import open
import boto3
import botocore
import io
import itertools
import glob
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from utils.utils import load_np_from_s3

class CTDataset(Dataset):
    """
    Custom dataset for CT report generation.

    Loads local and global CT image features (optionally from S3) and the 
    corresponding report. Prepares input IDs, labels, and attention masks 
    for multimodal language modeling.

    Returns a dict with:
        - local_features (torch.Tensor)
        - global_features (torch.Tensor)
        - input_ids (torch.LongTensor)
        - labels (torch.LongTensor)
        - attention_mask (torch.LongTensor)
    """
    
    def __init__(self, local_paths: str, global_paths: str, reports: list, tokenizer, input_prompt, config, mode, client=None):
        
        self.config = config
        self.local_paths = local_paths
        self.global_paths = global_paths
        self.reports=reports
        self.client= client
        self.tokenizer = tokenizer
        self.input_prompt = input_prompt
        self.image_tokens = "<vis_token>" * config.num_nodes
        self.mode = mode 

    def __len__(self):
        return len(self.local_paths)

    def __getitem__(self, idx):
     
        local_path = self.local_paths[idx]
        global_path = self.global_paths[idx]
        report = self.reports[idx]

        global_features = load_np_from_s3(global_path, self.client).unsqueeze(0)
        
        local_features=[]
        for layer in range(len(self.config.feat_dims)):
            feat_path = local_path.replace('layer_0',f'layer_{layer}')
            local_features.append(load_np_from_s3(feat_path, self.client).unsqueeze(0))
        local_features = np.concatenate(local_features, axis=2)
        local_features = torch.tensor(local_features)
        
        if self.config.use_filter:
            local_features = self.filter_local_features(local_features)
        else:
            local_features = local_features.squeeze(0)[9:]

        inputs = '<image>' + self.image_tokens + '</image>' + ' ' + self.input_prompt
        full_text = self.tokenizer(inputs +' '+ '<start_report>' + ' ' + report, max_length=self.config.max_length, truncation=True, padding="max_length", return_tensors='pt')

        all_tokens = full_text["input_ids"][0]
        attention_mask = full_text["attention_mask"][0]

        valid_len = torch.sum(attention_mask)
        if valid_len < len(all_tokens):
            all_tokens[valid_len] = self.tokenizer.eos_token_id
            
        input_only = self.tokenizer(inputs,max_length=self.config.max_length, truncation=True, padding="max_length", return_tensors='pt')
        input_len = torch.sum(input_only["attention_mask"][0])

        labels = all_tokens.clone()
        labels[:input_len] = -100
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            labels[labels == self.tokenizer.pad_token_id] = -100
            if valid_len < len(labels):
                labels[valid_len] = self.tokenizer.eos_token_id
        else:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        if self.mode == 'valid':
            tok_input = self.tokenizer(inputs, truncation=True, padding=True, return_tensors='pt').to('cuda')
            all_tokens = tok_input["input_ids"][0]
            attention_mask = tok_input["attention_mask"][0]       

        out = {'local_features': local_features,
               'global_features': global_features,
               'input_ids': all_tokens,
               'labels': labels,
               'attention_mask': attention_mask
              }
        
        return out
    
    def filter_local_features(self, local_features):
        """
        This method is used to filter/remove specific elements from the local features.
        This is needed, since coarse-level nodes are included twice in the organ list (see get_organs() from .graph/anatomy.py).
        The specific indices are hard-coded, and this function must be changed if the anatomical ordering changes.
        """        
        
        remove_indices = torch.tensor([26,41,35,25])
        mask = torch.ones(local_features.shape[1], dtype=torch.bool)  
        mask[remove_indices] = False 
        local_features = local_features[0,mask, :]

        return local_features[1:]   

class DataCollator:
    """
    Collator class for CTDataset.
    """
    
    def __call__(self, batch: list) -> dict:
        feats, global_feats, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch] for key in ('local_features', 'global_features', 'input_ids', 'labels', 'attention_mask')
        )

        local_features = torch.cat([_.unsqueeze(0) for _ in feats], dim=0)
        global_features = torch.cat([_ for _ in global_feats], dim=0)
        input_ids = torch.cat([torch.Tensor(_).unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([torch.Tensor(_).unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([torch.Tensor(_).unsqueeze(0) for _ in attention_mask], dim=0)

        return_dict = dict(
            local_features=local_features,
            global_features=global_features,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )

        return return_dict