import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

import argparse
import os
import glob

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import json
from pathlib import Path

from feature_extraction.models import get_models
from report_gen.data.dataset import DataCollator, CTDataset
from report_gen.data.get_data import get_data
from report_gen.config import ModelConfig, TrainingArguments, DataConfig
from report_gen.model.CTRepgen import CTLlamaForCausalLM, CTConfig
from report_gen.training.trainer import RepGenTrainer

from utils.utils import get_client

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def compute_metrics(eval_preds):
    """
    Compute token-level next-token accuracy from Trainer outputs.
    """
    y_true_all = eval_preds.label_ids
    y_pred_all = eval_preds.predictions

    # align targets/preds (shift)
    gt = y_true_all[:, 1:]
    pred = y_pred_all[:, :-1]

    # ignore padding labels
    ignore_idx = -100
    valid_mask = (gt != ignore_idx)

    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]

    correct = np.sum(pred_valid == gt_valid)
    total = gt_valid.size
    acc = correct / total

    return {"accuracy": acc}


def find_all_linear_names(model, ignore_keywords):
    """
    Collect names of all torch.nn.Linear submodules not matching any ignore keyword.
    """
    linear_cls = torch.nn.Linear
    target_names = set()
    for qual_name, submod in model.named_modules():
        if any(skip in qual_name for skip in ignore_keywords):
            continue
        if isinstance(submod, linear_cls):
            target_names.add(qual_name)
    return list(target_names)


def preprocess_logits_for_metrics(logits, labels):
    """Map logits to predicted token ids (argmax over last dim)."""
    _ = labels  # unused, kept for signature compatibility
    return torch.argmax(logits, dim=-1)


def save_args(args, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    args_dict = {
        k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
        for k, v in vars(args).items()
    }
    with path.open("w") as f:
        json.dump(args_dict, f, indent=4)

if __name__ == "__main__":

    parser = transformers.HfArgumentParser((TrainingArguments, ModelConfig, DataConfig))
    training_args,model_args,data_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())

    client = get_client() if data_args.use_s3 else None

    tokenizer = LlamaTokenizer.from_pretrained(data_args.tokenizer_path,
                                             model_max_length=model_args.max_length,
                                             padding_side="right",
                                             use_fast=False)
    special_token_list = ["<pad>", "<eos>", "<vis_token>", "<image>", "</image>", "<start_report>"]
    special_token = {"additional_special_tokens": special_token_list}

    tokenizer.add_special_tokens(special_token)
    tokenizer.eos_token="<eos>"
    tokenizer.pad_token="<pad>"
    
    feature_dims = get_models(arch=model_args.arch,weight_path=None,feature_dims_only=True)
    config = CTConfig(**vars(model_args))
    config.feat_dims = feature_dims
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    #Initialize main model
    model = CTLlamaForCausalLM.from_pretrained(
                    data_args.tokenizer_path,
                    config = config)    
    
    model.tokenizer = tokenizer
    model.resize(tokenizer)
    
    variant = model.get_model().variant
    ft_modules=list(variant.get_modules().keys())
    config.num_nodes = variant.num_nodes
    config.use_filter = variant.use_filter

    #Define peft model and trainable modules
    base_keywords = ['embed_tokens', 'lm_head']
    ignore_keywords = base_keywords + ft_modules
    lora_modules = find_all_linear_names(model, ignore_keywords)
    
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=lora_modules,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config).to('cuda')

    for n, p in model.named_parameters():
        if any(
                [x in n for x in ignore_keywords]
        ):
            p.requires_grad =True
            
    #Define training (and evaluation) dataset
    input_prompt = "Generate a medical report based on the visual information of the given CT image."
    
    train_local_paths, train_global_paths, train_reports = get_data(mode='train', model_config = config, data_args=data_args)
    test_local_paths, test_global_paths, test_reports, _ = get_data(mode='valid', model_config = config, data_args=data_args)
    
    train_dataset=CTDataset(local_paths= train_local_paths, global_paths = train_global_paths, reports = train_reports, tokenizer=tokenizer, input_prompt=input_prompt, config=config, mode='train', client=client)
    eval_dataset = CTDataset(local_paths= test_local_paths, global_paths = test_global_paths, reports = test_reports, tokenizer=tokenizer, input_prompt=input_prompt, config=config, mode='valid', client=client)

    data_collator = DataCollator()

    #save training, data and model parameters
    save_args(training_args, f"{training_args.output_dir}/training_config.json")
    save_args(data_args, f"{training_args.output_dir}/data_config.json")
    config.to_json_file(f"{training_args.output_dir}/model_config.json")

    trainer = RepGenTrainer(
                        model=model,
                        args=training_args,
                        model_args=model_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics,      
                      )

    trainer.train()
    trainer.save_state()
    