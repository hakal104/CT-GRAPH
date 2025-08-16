import os
import boto3
import zipfile
from io import BytesIO
import tempfile
import io
import shutil
import glob
import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from smart_open import open

import pandas as pd
import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, MetaTensor

from models import get_models
from utils.utils import get_client, load_np_from_s3
from feature_extraction.anatomy_mapping import interpolate_array, map_to_fine_level, map_to_coarse_level
from monai.data import MetaTensor

class MaskFeaturesDataset(Dataset):
    """
    Dataset for paired CT volumes and anatomy masks from S3.

    Loads CT ('image') and anatomy mask files ('ana') in parallel, and applies 
    corresponding transforms based on the architecture of the feature extractor.

    Returns:
        dictionary with 'image', 'ana', and 'paths'.
    """
    
    def __init__(self, ct_paths, ana_paths, arch, transform, use_s3):
        
        self.ct_paths = ct_paths
        self.ana_paths = ana_paths
        self.transform = transform
        self.arch = arch

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        
        ct_path = self.ct_paths[idx]
        ana_path = self.ana_paths[idx]
        
        s3_paths=[ct_path,ana_path]
        # Load images in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tensors = list(executor.map(lambda path: load_np_from_s3(path, client,compressed=True), s3_paths))
            
        img = tensors[0]
        ana = tensors[0]
        
        if self.arch == 'ctfm':
            img = img.permute(2, 1, 0)         # (X, Y, Z) → (Z, Y, X)
            img = torch.flip(img, dims=(1, 2))

            ana = ana.permute(2, 1, 0)
            ana = torch.flip(ana, dims=(1, 2))

        d = {
            'image': img,
            'ana': ana
         } 

        d['paths'] = ct_path.replace('.nii.gz','.npy')
        out =self.transform(d)
        
        return out
    

def write_single_feature(feats, batch_paths, arch, layer_idx, j):
    """
    Uploads features to S3.
    """
    
    print("final feats: ", feats.shape)
    buffer = io.BytesIO()
    np.save(buffer, feats)
    buffer.seek(0)

    # Construct save path and upload to S3
    save_path = batch_paths[j].replace('dataset', f'features/{arch}/downsampling/ip/layer_{layer_idx}')
    print(save_path)
    #client.upload_fileobj(buffer, 'hamza-kalisch', save_path)
    buffer.close()

    return f'Successfully uploaded feature vector {j} to {save_path}'
                
def get_features(model, dataset, arch, batch_size, roi, save_dir, use_s3):
    """
    Computes global features for given model and images via sliding window inference.
    Local features are then computed by mask-pooling and are saved.
    """
    
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    model = model.to('cuda')
    
    for batch in data_loader:

        images = batch["image"].unsqueeze(1).to('cuda')  # Access images
        ana_labels= batch["ana"].unsqueeze(1)

        with torch.no_grad(), torch.cuda.amp.autocast():
            features =sliding_window_inference(inputs=images, 
                                     roi_size=roi,
                                     sw_batch_size=8,
                                     predictor=model,
                                     overlap=0.25,
                                     mode='gaussian')

        del images
        
        fine_level_labels, fine_level_idx, num_fine_nodes = map_to_fine_level(ana_labels)
        max_length = len(features)-1
        
        for i, feat in enumerate(features):
            all_features=[]   
            B,C,H,W,D = feat.shape
            target_size = (B,1,H,W,D)

            fine_arr = interpolate_array(fine_level_labels, target_size)
            coarse_arr, num_coarse_nodes = map_to_coarse_level(fine_arr, fine_level_idx)
      
            ana_features = ana_mask_pooling(feat, coarse_arr, fine_arr,num_coarse_nodes, num_fine_nodes).cpu().numpy()
            all_features.append(ana_features)
 
            if i==max_length:
                all_features.append(feat.cpu().numpy())
            
            for k, ana_feats in enumerate(all_features):
                volume_id = batch["paths"]
                if k==1:
                    volume_id = [elm.replace('.npy','_full.npy') for elm in volume_id]
                
                if use_s3:
                    with ProcessPoolExecutor(max_workers=8) as executor:
                        futures = [
                                executor.submit(
                                write_single_feature, 
                                ana_feats[j], 
                                volume_id, 
                                arch,
                                i, 
                                j
                            )
                            for j in range(ana_features.shape[0])
                        ]
                else:
                    for j in range(ana_features.shape[0]):
                        save_path = volume_id[j].replace('dataset', os.path.join(save_dir, f'features/{arch}/downsampling/ip/layer_{layer_idx}'))
                        np.save(ana_feats[j],save_path)
            
            del ana_features, all_features
            
        del features, ana_labels
        torch.cuda.empty_cache()  
    

def ana_mask_pooling(features, coarse_labels, fine_labels, num_coarse, num_fine):
    """
    Implementation of mask pooling, which is applied for global, coarse-level 
    and fine-level features.
    """
    
    B, C, D, H, W = features.shape
    
    num_global =1
    num_features = num_global + num_coarse + num_fine
    ana_features = torch.zeros((B, num_features, C), device=features.device)
    running_label = 0
    
    ana_features[:,0,:] = torch.mean(features, axis=[2, 3, 4])
    with torch.no_grad():
        for ana_labels, num_labels in [(coarse_labels,num_coarse), (fine_labels,num_fine)]:
            for label in range(1, num_labels+1):

                label_mask = (ana_labels == label)  # Shape: (B, D, H, W)
                running_label +=1
                
                for b in range(B):

                    #Select feature vector based on given label mask
                    selected_features = features[b, :, label_mask[b].squeeze()].reshape(C, -1)  # Shape: (C, N) where N is the count of label points
                    
                    #For non-empty features apply mean pooling 
                    if selected_features.numel() > 0:

                        mean_feature = selected_features.mean(dim=1)  # Shape: (C,)
                        ana_features[b, running_label, :] = mean_feature


            # Clean up to reduce memory usage per iteration
            del label_mask, selected_features
    
    return ana_features
                                       
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract features from CT images via pretrained 3D SSL models.")
    parser.add_argument("--weight_path", type=str, default=None, help="Path to model weights.")
    parser.add_argument("--ctrate_dir", type=str, default=None, help="Path to CT-RATE directory.")
    parser.add_argument("--arch", type=str, default=None, help="Type of feature extractor.")
    parser.add_argument("--arch_size", type=str, default='B', help="Size of architecture.")
    parser.add_argument("--mode", type=str, default="", help="Training or validation features.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--use_s3", type=bool, default=False, help="Whether to use S3 storage.")
    
    args = parser.parse_args()

    client=get_client()
    reports_df = pd.read_csv(os.path.join(args.ctrate_dir, 'train_reports.csv')).drop_duplicates('Findings_EN')
    names = list(reports_df['VolumeName'])   

    ct_paths = [f'dataset/{args.mode}/{name[:-11]}/{name[:-9]}/{name}'.replace('.nii.gz','.npy') for name in names]
    ana_paths = [f'dataset/{args.mode}/{name[:-11]}/{name[:-9]}/ana_{name}'.replace('.nii.gz','.npy') for name in names]

    if not args.use_s3:
        ct_paths = [os.path.join(args.ctrate_dir,path) for path in ct_paths]
        ana_paths = [os.path.join(args.ctrate_dir,path) for path in ana_paths]
        
    is_voco = 'voco' in args.arch
    model,transforms,_,roi = get_models(arch=args.arch, weight_path =args.weight_path, is_voco=is_voco)
    ds=MaskFeaturesDataset(ct_paths,ana_paths,arch=args.arch,transform=transforms)
    get_features(model,ds,args.arch,args.batch_size,roi,args.ctrate_dir,args.use_s3)