import os
import numpy as np
from types import SimpleNamespace
import argparse
import json
from utils.utils import load_np_from_s3, get_client

import torch
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from report_gen.model.CTRepgen import CTLlamaForCausalLM, CTConfig
from report_gen.data.dataset import CTDataset, DataCollator
from report_gen.data.get_data import get_data

def get_tokenizer(config, data_args):
    """
    Returns initialized tokenizer.
    """
    
    tokenizer = LlamaTokenizer.from_pretrained(data_args.tokenizer_path,
                                             model_max_length=config.max_length,
                                             padding_side="right",
                                             use_fast=False)
    special_token_list = ["<pad>", "<eos>", "<vis_token>", "<image>", "</image>", "<start_report>"]
    
    special_token = {"additional_special_tokens": special_token_list}
    tokenizer.add_special_tokens(special_token)
    tokenizer.eos_token="<eos>"
    tokenizer.pad_token="<pad>"
    
    return tokenizer

# Function to load the model
def load_model(weights_dir, config, data_args):
    """
    Loads a pretrained CTLlama model and restores its embedding, LM head, and 
    all trainable submodule weights. Finally, integrates the PEFT layers and 
    returns the fully initialized model.

    Args:
        weights_dir (str): Directory containing all saved weight files.
        config: Model configuration object.
        data_args: Data Arguments.

    Returns:
        PeftModel: The fully restored model ready for inference or training.
    """
    
    model = CTLlamaForCausalLM.from_pretrained(
                    data_args.tokenizer_path,
                    config=config
            )
    model.tokenizer = tokenizer
    model.resize(tokenizer)
    model.config.eos_token_id = tokenizer.eos_token_id

    model.model.embed_tokens.load_state_dict(torch.load(os.path.join(weights_dir, 'embed_tokens.pt')))
    model.lm_head.load_state_dict(torch.load(os.path.join(weights_dir, 'lm_head.pt')))

    for module_name, module in model.model.variant.get_modules().items():
        state_path = os.path.join(weights_dir,module_name)
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Expected weight file not found: {state_path}")
        state_dict = torch.load(state_path)
        module.load_state_dict(state_dict)
        
    final_model = PeftModel.from_pretrained(model, weights_dir)
    
    return final_model

def generate_reports(model, data_loader, tokenizer):
    """
    Generate text reports from a dataset using a pretrained CTLlama model.
    """
    
    all_gen_reports = []
                
    for batch in data_loader:
        
        local_features = batch["local_features"].to("cuda")
        global_features = batch["global_features"].to("cuda")
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")
        attention_masks = batch["attention_mask"].to("cuda")
        
        with torch.no_grad():
        # Call the generate function for the entire batch
            output = model.generate(
                local_features=local_features,
                global_features=global_features,# Batch of node features
                inputs=input_ids,
                attention_mask=attention_masks,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        prompt_lengths = attention_masks.sum(dim=1)

        report_ids = [
            output[i, prompt_lengths[i]:]  # skip the prompt tokens
            for i in range(len(output))
        ]

        generated_reports = [
            tokenizer.decode(report, skip_special_tokens=True).lstrip()
            for report in report_ids
        ]
        
        all_gen_reports += generated_reports
        
        print(generated_reports[0])

    return all_gen_reports


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate reports.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights.")
    parser.add_argument("--batch_size", type=int, default=14, help="Batch size.")

    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(args.model_path),'model_config.json')
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CTConfig.from_dict(config_dict) 
    
    data_config_path = os.path.join(os.path.dirname(args.model_path),'data_config.json')
    with open(data_config_path) as f:
        data_args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    tokenizer = get_tokenizer(config, data_args)
    
    client = get_client() if data_args.use_s3 else None
    val_local_paths, val_global_paths, val_reports, _ = get_data(mode='valid', model_config=config, data_args=data_args)
    input_prompt = "Generate a medical report based on the visual information of the given CT image."
    
    val_dataset = CTDataset(local_paths= val_local_paths, global_paths = val_global_paths, reports = val_reports, tokenizer=tokenizer, input_prompt=input_prompt, config=config, mode='valid', client=client)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=DataCollator())
    
    model = load_model(args.model_path, config, data_args)
    model= model.to('cuda').eval()
    
    gen_reports = generate_reports(model=model, 
                             data_loader=val_loader,
                             tokenizer=tokenizer)
    
    reports_save_path = os.path.join(os.path.dirname(args.model_path), 'reports.json')
    with open(reports_save_path, "w", encoding="utf-8") as f:
        json.dump(gen_reports, f, ensure_ascii=False, indent=2)
