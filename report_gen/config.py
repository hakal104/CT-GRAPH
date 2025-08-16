import huggingface
from dataclasses import dataclass
import transformers
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class ModelConfig:
    llama_dim: int = 4096
    vocab_size: int = 32000
    max_length: int = 640
    arch: str = "swinunetr"
    feat_dims :  List[int] = field(default_factory=lambda: [512])
    layer: int = 0
    mode: str = "use_ctgraph"
    hidden_dim_proj: int = 1024
    att_dim: int = 256
    out_dim_proj: int = 512
    num_heads: int = 8
    global_proj_dim: int = 9192
    
@dataclass
class DataConfig:
    
    tokenizer_path: str = '/home/jovyan/ct-repgen-datavol-1/.llama/checkpoints/Llama-2-7b-hf/'
    ctrate_data_path: str = '/home/jovyan/ct-repgen-datavol-1/datasets/ct-rate/'
    use_s3: bool = True
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=640,  # Maximum sequence length
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    optim: str = field(default="adamw_torch")
    bf16: bool = True
    fp16: bool = False
    output_dir: str = "/home/jovyan/ct-repgen-datavol-1/ctgraph/report_gen/VLM_runs/swinunetr"
    num_train_epochs: float = 6
    per_device_train_batch_size: int = 8 
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1 #
    evaluation_strategy: str = "no"
    eval_accumulation_steps: Optional[int] = None  
    eval_steps: Optional[float] = None
    save_strategy: str = "epoch"
    save_steps: int = 3
    save_total_limit: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "constant"
    logging_steps: float = 10
    gradient_checkpointing: bool = True  
    dataloader_pin_memory: bool = True  
    dataloader_num_workers: int = 8 
    report_to: str = "tensorboard"  
    disable_tqdm: bool = False
    max_grad_norm: float = 1.0
