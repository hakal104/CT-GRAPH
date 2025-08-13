"""
Code to run multi-abnormality classifier on generated reports. 
Copied and adapted from https://github.com/ibrahimethemhamamci/CT-CLIP/blob/main/text_classifier/infer.py. 
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.optim import AdamW
import time
from scipy.special import expit

class ModelTrainer:
    def __init__(self,
                 model,
                 dataloaders,
                 num_class,
                 epochs,
                 optimizer,
                 scheduler,
                 device,
                 save_path,
                 test_label_cols,
                 save_in=10,
                 early_stop=100,
                 threshold = 0.5):
        
        self.model = model
        self.dataloaders = dataloaders
        self.num_class = num_class
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.threshold = threshold
        self.early_stop = early_stop
        self.save_in = save_in
        self.test_label_cols = test_label_cols
    
    def infer(self):
        self.model.eval()
        pred_logits = np.zeros(self.num_class).reshape(1,self.num_class)

        with torch.no_grad():
            for input_ids,attention_mask in tqdm(self.dataloaders['test']):
                input_ids = torch.squeeze(input_ids)
                attention_mask = torch.squeeze(attention_mask)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                logits = self.model(input_ids,attention_mask)
                
                pred_logits = np.concatenate((pred_logits,logits.detach().cpu().numpy()),axis=0)
                
        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        
        pred_labels[pred_labels>=self.threshold]=1
        pred_labels[pred_labels<self.threshold]=0
        
        return pred_labels
    
class CTDataset(Dataset):
    
    def __init__(self, data_files, class_count, label_cols, max_length, augment = False ,infer=False):
        
        
        self.data_files = data_files
        self.class_count = class_count
        
        self.tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m',do_lower_case=True)
        self.max_length = 512
        self.label_cols = label_cols
        self.infer = infer
        self.augment = augment

        if self.augment:
            self.txt_augment = TextAugment()
        
        
    def __len__(self):
        return len(self.data_files)
    
    
    def __getitem__(self,idx):
        text_comment = self.data_files[idx]
        text_comment = str(text_comment) if not isinstance(text_comment, str) else text_comment
        if pd.isna(text_comment):
            text_comment = " "

        encodings = self.tokenizer(text_comment, return_tensors='pt',max_length=self.max_length,padding='max_length',truncation=True)
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask'] 
        
        return input_ids, attention_mask           
    
class RadBertClassifier(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
    
        self.config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        self.model = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=self.config)
    
        self.classifier=nn.Linear(self.model.config.hidden_size,n_classes) 
        
    def forward(self,input_ids, attn_mask):
        output = self.model(input_ids=input_ids,attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)
              
        return output

def get_abn_labels(final_reports, model_path):

    label_cols = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
    num_labels = len(label_cols)
    print('Label columns: ', label_cols)
    print('\nNumber of test data: ',len(final_reports))

    # Create dataloader

    dataloaders = {}

    max_length = 512
    num_workers = 4

    batch_size = 30

    test_data = CTDataset(data_files = final_reports, class_count=num_labels, label_cols=label_cols, max_length=512)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=True, shuffle=False)
    dataloaders['test'] = test_dataloader


    save_path = os.path.dirname(model_path)
    device='cuda'

    model = RadBertClassifier(n_classes=num_labels)
    model.load_state_dict(torch.load(model_path),strict=False)
    model = model.to(device)
    print(model.eval())

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
    scheduler = None
    # optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization
    epochs = 0


    trainer = ModelTrainer(model,
                           dataloaders,
                           num_labels,
                           epochs,
                           optimizer,
                           scheduler,
                           device,
                           save_path,
                           label_cols)

    start = time.time()
    print('----------------------Starting Inferring----------------------')
    predicted_labels = trainer.infer()

    finish = time.time()
    print('---------------------------------------------------------------')
    print('Inferring Complete')
    print('Infer time: ',finish-start)

    columns = ['AccessionNo','Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']

    ce_labels = pd.DataFrame()
    #inferred_data[columns[0]] = df['AccessionNo']
    #inferred_data['report_text'] = train_reports[:100]

    for col,i in zip(columns[1:],range(num_labels)):
        ce_labels[col] = predicted_labels[:,i]

    return ce_labels
    
    
if __name__ == '__main__':
    
    final_df,reports,impressions, abn = get_data()
    #get_batch_f1(reports)