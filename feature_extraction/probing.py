import copy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils
from torch.optim import Adam, AdamW

import boto3
import argparse
import os
from monai.transforms import MapTransform
from smart_open import open
import boto3
import botocore
import io
import itertools
import nibabel as nib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.utils import get_client, load_np_from_s3
from feature_extraction.models import get_models
from feature_extraction.anatomy_mapping import count_level_nodes

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class CTDataset(Dataset):
    """
    Custom dataset for classification or linear probing.
    Provides local features based on selected layer configuration
    (single layer or layer fusion) and specified pooling level.
    """
    
    def __init__(self, local_paths, labels, concat_feats, num_nodes, pool_level, num_layers, client):
        
        self.local_paths = local_paths
        self.labels = labels
        self.concat_feats=concat_feats
        self.num_nodes=num_nodes
        self.pool_level = pool_level
        self.num_layers = num_layers
        self.client = client

    def __len__(self):
        return len(self.local_paths)
    
    def __getitem__(self, idx):
        
        local_path = self.local_paths[idx]
        label = self.labels[idx]

        feat_slices = {}
        num_global, num_coarse, num_fine = self.num_nodes['global'], self.num_nodes['coarse'], self.num_nodes['fine']
        feat_slices['global'] = (0,num_global)
        feat_slices['coarse'] = (num_global,num_coarse+num_global)
        feat_slices['fine'] = (num_coarse+num_global,num_fine+num_coarse+num_global)
        
        if self.concat_feats:
            features=[]
            for layer in range(self.num_layers):
                feat_path = local_path.replace('layer_0',f'layer_{layer}')
                features.append(load_np_from_s3(feat_path, self.client).unsqueeze(0))
            features = np.concatenate(features,axis=2)
        else:
            features = load_np_from_s3(local_path, self.client).unsqueeze(0)
            
        if self.pool_level != 'all':
            features = features[:, feat_slices[self.pool_level][0]:feat_slices[self.pool_level][1], :]
        else:
            #filter out to avoid duplicate coarse features
            remove_indices = torch.tensor([26,41,35,25])
            mask = torch.ones(num_global+num_coarse+num_fine, dtype=torch.bool)  
            mask[remove_indices] = False 
            
            features = features[:,mask, :]
            
        features = torch.tensor(features).squeeze(0)
        
        return features, label
    
class Classifier(nn.Module):
    """Simple classifier for linear probing of input features."""
    
    def __init__(self, n_outputs, feature_dim, num_feats):
        super(Classifier, self).__init__()        
        
        self.simple_classifier = nn.Linear(num_feats*feature_dim, n_outputs)
        
    def forward(self, x):

        x = x.view(x.shape[0],-1)
        out = self.simple_classifier(x)

        return out

    
def train_epoch(model, data_loader, optimizer, scaler, scheduler, criterion, device):
    """Run one training epoch and return loss and evaluation metrics."""
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []
    
    for local_inputs, labels in data_loader:
        
        local_inputs = local_inputs.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(local_inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping (critical for stability)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=1.0,  # Start with 1.0, tune between 0.5-5.0
            norm_type=2
        )
        
        scaler.step(optimizer)
        if scheduler != None:
            scheduler.step()

        scaler.update()
        running_loss += loss.item() * local_inputs.size(0)
        all_labels.append(labels.cpu().detach())
        all_preds.append(outputs.cpu().detach())

    epoch_loss = running_loss / len(data_loader.dataset)
    all_labels = torch.cat(all_labels)
    all_preds = (torch.cat(all_preds) > 0.5).float()  # Threshold for multilabel
    
    metrics = compute_metrics(all_labels, all_preds, epoch_loss)
    
    return metrics

def val_epoch(model, data_loader, criterion, device):
    """Evaluate the model for one epoch and return loss and metrics."""
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for local_inputs, labels in data_loader:
            local_inputs, labels = local_inputs.to(device), labels.float().to(device)
            with torch.cuda.amp.autocast():
                outputs = model(local_inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * local_inputs.size(0)
            all_labels.append(labels.cpu())
            all_preds.append((outputs.cpu() > 0.5).float())

    epoch_loss = running_loss / len(data_loader.dataset)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    
    metrics = compute_metrics(all_labels, all_preds, epoch_loss)
    
    return metrics


def compute_metrics(all_labels, all_preds, loss):
    """Return loss, F1, precision, and recall for micro, macro, and weighted averages."""
    
    averages = ["micro", "macro", "weighted"]
    metrics = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score
    }

    results = {"loss": loss}

    for avg in averages:
        for name, func in metrics.items():
            results[f"{name}_{avg}"] = func(all_labels, all_preds, average=avg)

    return results


def get_data(mode, ctrate_data_path, radgenome_data_path, use_s3, arch, feat_dims, layer):
    """
    Defines the paths for the local and global features. Provides the reports and abnormality labels.
    """
    
    # Filter samlpes based on duplicates of original reports from CT-RATE
    reports_path = os.path.join(ctrate_data_path, f'{mode}_reports.csv')
    ct_rate_df = pd.read_csv(reports_path).drop_duplicates('Findings_EN')
    names = list(ct_rate_df['VolumeName'])     
    
    region_reports_path = os.path.join(radgenome_data_path, f'{mode}_region_report.csv')
    df_rr=pd.read_csv(region_reports_path)
    radgenome_df=df_rr[df_rr['Anatomy'].isna()].sort_values('Volumename')
    filtered_df = radgenome_df[radgenome_df['Volumename'].isin(names)]
    filtered_df['Volumename'] = pd.Categorical(filtered_df['Volumename'], categories=names, ordered=True)
    filtered_df = filtered_df.sort_values('Volumename')
    names = list(filtered_df['Volumename'])   
    
    # Define the exact paths for local and global features. Adapt paths accordingly. 
    ct_paths = [f'dataset/{mode}/{name[:-11]}/{name[:-9]}/{name}' for name in names]
    local_paths = [elem.replace('dataset', f'features/{arch}/downsampling/ip/layer_{layer}').replace('.nii.gz',f'.npy') for elem in ct_paths]

    # If S3 is not used, features are assumed to be located at following paths: 
    if not use_s3:
        local_paths = [os.path.join(ctrate_data_path,path) for path in local_paths]
        
    abnormality_path = os.path.join(ctrate_data_path, f'dataset_multi_abnormality_labels_{mode}_predicted_labels.csv')
    abnormality_df=pd.read_csv(abnormality_path)
    abnormality_df = abnormality_df[abnormality_df['VolumeName'].isin(ct_rate_df['VolumeName'])]
    abnormality_labels = np.column_stack([abnormality_df[col].values for col in abnormality_df.columns if col != 'VolumeName'])

    return local_paths, abnormality_labels

def load_and_split_data(args, feat_dims):
    """
    Load training and test data and perform a 90/10 train-validation split on the training data.
    """
    # train set
    train_local_paths, train_labels = get_data(
        mode='train',
        ctrate_data_path=args.ctrate_data_path,
        radgenome_data_path=args.radgenome_data_path,
        use_s3=args.use_s3,
        arch=args.arch,
        feat_dims=feat_dims,
        layer=args.layer
    )

    data = list(zip(train_local_paths, train_labels))
    train_data, val_data = train_test_split(
        data, test_size=0.1, shuffle=True, random_state=42
    )
    train_local_paths, train_labels = map(list, zip(*train_data))
    val_local_paths, val_labels     = map(list, zip(*val_data))

    test_local_paths, test_labels = get_data(
        mode='valid',
        ctrate_data_path=args.ctrate_data_path,
        radgenome_data_path=args.radgenome_data_path,
        use_s3=args.use_s3,
        arch=args.arch,
        feat_dims=feat_dims,
        layer=args.layer
    )

    return train_local_paths, train_labels, val_local_paths, val_labels, test_local_paths, test_labels
    

def build_loaders(args, client, num_layers,
                   train_local_paths, train_labels,
                   val_local_paths,   val_labels,
                   test_local_paths,  test_labels):
    """
    Build CTDataset objects and prepare dataloaders.
    """
    
    num_nodes = count_level_nodes()

    train_dataset = CTDataset(
        local_paths=train_local_paths, labels=train_labels,
        concat_feats=args.concat_feats, num_nodes=num_nodes,
        pool_level=args.pool_level, num_layers=num_layers, client=client
    )
    val_dataset = CTDataset(
        local_paths=val_local_paths, labels=val_labels,
        concat_feats=args.concat_feats, num_nodes=num_nodes,
        pool_level=args.pool_level, num_layers=num_layers, client=client
    )
    test_dataset = CTDataset(
        local_paths=test_local_paths, labels=test_labels,
        concat_feats=args.concat_feats, num_nodes=num_nodes,
        pool_level=args.pool_level, num_layers=num_layers, client=client
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=16)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=16)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=16)
    
    return train_loader, val_loader, test_loader, num_nodes


def train_and_evaluate(args, model, optimizer, scaler, scheduler, criterion,
                        train_loader, val_loader, test_loader, device):
    """
    Run the training, validation, and test loops for all epochs and return evaluation metrics.
    """
    
    metric_names = ["f1_macro", "f1_micro", "precision_macro", "precision_micro",
                    "recall_macro", "recall_micro", "loss"]
    splits = ["train", "val", "test"]
    all_metrics = {split: {name: [] for name in metric_names} for split in splits}

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, scaler, scheduler, criterion, device)
        val_metrics   = val_epoch(model, val_loader,   criterion, device)
        test_metrics  = val_epoch(model, test_loader,  criterion, device)

        results = {"train": train_metrics, "val": val_metrics, "test": test_metrics}

        print(
            f"Train Loss: {results['train']['loss']:.4f} | "
            f"Train micro F1: {results['train']['f1_micro']:.4f} | "
            f"Train macro F1: {results['train']['f1_macro']:.4f}"
        )
        print(
            f"Val Loss: {results['val']['loss']:.4f} | "
            f"Val micro F1: {results['val']['f1_micro']:.4f} | "
            f"Val macro F1: {results['val']['f1_macro']:.4f}"
        )
        print(
            f"Test Loss: {results['test']['loss']:.4f} | "
            f"Test micro F1: {results['test']['f1_micro']:.4f} | "
            f"Test macro F1: {results['test']['f1_macro']:.4f}"
        )

        for split in splits:
            for name in metric_names:
                all_metrics[split][name].append(results[split][name])

    return all_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform multilabel classification on CT images.")
    parser.add_argument("--ctrate_data_path", type=str, default='', help="Data directory for the CT-RATE dataset.")
    parser.add_argument("--radgenome_data_path", type=str, default='', help="Data directory for the Radgenome dataset.")
    parser.add_argument("--save_dir", type=str, default='', help="Directory path for saving results.")    
    parser.add_argument("--use_s3", type=bool, default=False, help="Whether to use S3 storage.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--wd", type=float, default=0.0001, help="Weight decay for the optimizer.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--layer", type=int, default=0, help="Layer to extract features from.")
    parser.add_argument("--arch", type=str, default=None, help="Type of feature extractor.")
    parser.add_argument("--concat_feats", type=bool, default=False, help="Whether to use all feats or not.")
    parser.add_argument("--pool_level", type=str, choices=["all","global","coarse","fine"], default="all", help="Pooling level.")
   
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client = get_client()

    feat_dims =  get_models(arch=args.arch,weight_path=None,feature_dims_only=True)
    feat_dim = np.sum(np.array(feat_dims)) if args.concat_feats else feat_dims[args.layer]
    num_layers = len(feat_dims)

    (train_local_paths, train_labels,
       val_local_paths,   val_labels,
      test_local_paths,  test_labels) = load_and_split_data(args, feat_dims)

    train_loader, val_loader, test_loader, num_nodes = build_loaders(args, client, num_layers,
                                                                train_local_paths, train_labels,
                                                                val_local_paths,   val_labels,
                                                                test_local_paths,  test_labels)
    
    #set up model
    model = Classifier(n_outputs=np.array(train_labels).shape[1], 
                       feature_dim=feat_dim,
                       num_feats=num_nodes[args.pool_level]).to('cuda')
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.2
    )
    criterion = nn.BCEWithLogitsLoss()  

    #training and validation
    all_metrics =  train_and_evaluate(
        args, model, optimizer, scaler, scheduler, criterion,
        train_loader, val_loader, test_loader, device
    )
    
    #save results
    layer='concat' if args.concat_feats else args.layer
    save_metrics = f'layer_{layer}_pool_{args.pool_level}.json'
    save_path = os.path.join(args.save_dir, save_metrics)
    with open(save_path, 'w') as f:
        json.dump(all_metrics, f)