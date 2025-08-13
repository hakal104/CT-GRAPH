import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class RepGenTrainer(Trainer):
    """Trainer subclass that stores `model_args` and customizes checkpoint saving."""
    
    def __init__(self, model, args, model_args, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        self.model_args = model_args
        
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save a checkpoint: HF model/tokenizer, training args, and extra components (embed_tokens, lm_head, variant modules)."""
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()
    
        base = self.model.base_model.model
        variant = base.model.variant
        
        torch.save(base.model.embed_tokens.state_dict(), os.path.join(output_dir, 'embed_tokens.pt'))
        torch.save(base.lm_head.state_dict(), os.path.join(output_dir, 'lm_head.pt'))
        
        for module_name, module in variant.get_modules().items():
            torch.save(module.state_dict(), os.path.join(output_dir, module_name))

        self.model.save_pretrained(output_dir, state_dict=state_dict)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))